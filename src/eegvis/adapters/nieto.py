"""Nieto 2022 adapter. Dataset-specific mapping lives here, not in core math."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from eegvis.adapters.base import AdapterError
from eegvis.types import EpochBatch

CLASS_NAMES = {0: "up", 1: "down", 2: "right", 3: "left"}
CONDITION_IDS = {
    "pronounced": 0,
    "pron": 0,
    "inner": 1,
    "in": 1,
    "visualized": 2,
    "vis": 2,
}

# Paper / derivatives layout. Not used by cwt/riemann.
CLASS_COLUMN = 1
CONDITION_COLUMN = 2
SESSION_COLUMN = 3


def _subject_name(subject: int) -> str:
    return f"sub-{subject:02d}"


def _montage_xy(ch_names: list[str]) -> np.ndarray | None:
    try:
        import mne
    except ImportError:
        return None
    mont = mne.channels.make_standard_montage("biosemi128")
    pos = mont.get_positions()["ch_pos"]
    xy = []
    for name in ch_names:
        if name not in pos:
            return None
        xyz = pos[name]
        xy.append([xyz[0], xyz[1]])
    return np.asarray(xy, dtype=np.float64)


def load_nieto(
    subject: int,
    *,
    condition: str = "inner",
    root: str | Path | None = None,
    sessions: list[int] | None = None,
) -> EpochBatch:
    """Load one subject as ``EpochBatch``.

    Tries local OpenNeuro-style derivatives first, then MOABB ``Nieto2022``.
    """
    cond_key = condition.strip().lower()
    if cond_key not in CONDITION_IDS:
        raise AdapterError(f"unknown condition {condition!r}")
    cond_id = CONDITION_IDS[cond_key]
    root = Path(root) if root is not None else _default_root()
    if root is not None and (root / "derivatives").exists():
        return _load_derivatives(root, subject, cond_id, cond_key, sessions)
    return _load_moabb(subject, cond_id, cond_key, sessions)


def _default_root() -> Path | None:
    env = os.environ.get("EEGVIS_NIETO_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for cand in (
        Path.cwd() / "Dataset",
        here.parents[4] / "Dataset",
        here.parents[3] / "Dataset",
    ):
        if (cand / "derivatives").exists():
            return cand
    return None


def _load_derivatives(
    root: Path,
    subject: int,
    cond_id: int,
    cond_key: str,
    sessions: list[int] | None,
) -> EpochBatch:
    try:
        import mne
    except ImportError as exc:
        raise AdapterError("install eegvis[nieto] (mne) to load local derivatives") from exc

    name = _subject_name(subject)
    session_ids = sessions or [1, 2, 3]
    xs, ys = [], []
    info = None
    tmin = None
    for ses in session_ids:
        base = root / "derivatives" / name / f"ses-0{ses}" / f"{name}_ses-0{ses}"
        fif = Path(str(base) + "_eeg-epo.fif")
        events_path = Path(str(base) + "_events.dat")
        if not fif.exists() or not events_path.exists():
            continue
        epochs = mne.read_epochs(fif, verbose="ERROR")
        events = np.load(events_path, allow_pickle=True)
        data = epochs.get_data(copy=True)
        if data.shape[0] != events.shape[0]:
            raise AdapterError(f"trial count mismatch in {fif}")
        keep = events[:, CONDITION_COLUMN] == cond_id
        if not np.any(keep):
            continue
        xs.append(data[keep])
        ys.append(events[keep])
        info = epochs.info
        if tmin is None:
            tmin = float(epochs.tmin)
    if not xs or info is None or tmin is None:
        raise AdapterError(f"no inner/selected trials for subject {subject} under {root}")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    ch_names = list(info.ch_names)
    return EpochBatch(
        data=x,
        sfreq=float(info["sfreq"]),
        ch_names=ch_names,
        tmin=tmin,
        labels=y[:, CLASS_COLUMN].astype(int),
        sessions=y[:, SESSION_COLUMN].astype(int),
        montage_xy=_montage_xy(ch_names),
        class_names=dict(CLASS_NAMES),
        subject_id=name,
        condition=cond_key,
        dataset="nieto2022",
    )


def _load_moabb(
    subject: int,
    cond_id: int,
    cond_key: str,
    sessions: list[int] | None,
) -> EpochBatch:
    try:
        from moabb.datasets import Nieto2022
        import mne
    except ImportError as exc:
        raise AdapterError(
            "Nieto data not found locally. Set EEGVIS_NIETO_ROOT to the dataset "
            "folder (with derivatives/) or pip install eegvis[moabb]."
        ) from exc

    dataset = Nieto2022()
    try:
        raws = dataset.get_data(subjects=[subject])
    except Exception as exc:
        raise AdapterError(f"MOABB could not fetch Nieto2022 subject {subject}: {exc}") from exc

    # MOABB layout: data[subject][session][run] -> Raw
    sub_dict = raws.get(subject) or raws.get(str(subject))
    if not sub_dict:
        raise AdapterError("unexpected MOABB payload for Nieto2022")

    picked = []
    for session_key, runs in sub_dict.items():
        if sessions is not None:
            digits = "".join(ch for ch in str(session_key) if ch.isdigit())
            if digits and int(digits) not in sessions:
                continue
        for raw in runs.values():
            picked.append(raw)

    if not picked:
        raise AdapterError("no MOABB runs matched the requested sessions")

    # Epoch on directional cues when annotations/events exist; otherwise fail clearly.
    event_id = dict(up=31, down=32, right=33, left=34)
    epochs_list = []
    for raw in picked:
        events = mne.find_events(raw, verbose="ERROR") if raw.info.get("chs") else np.empty((0, 3))
        try:
            epo = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=-0.5,
                tmax=4.0,
                preload=True,
                baseline=None,
                verbose="ERROR",
            )
        except Exception:
            continue
        epochs_list.append(epo)
    if not epochs_list:
        raise AdapterError(
            "MOABB Raw loaded but directional events were not found. "
            "Use local derivatives (recommended for this adapter)."
        )
    epochs = mne.concatenate_epochs(epochs_list, verbose="ERROR")
    data = epochs.get_data(copy=True)
    code = {31: 0, 32: 1, 33: 2, 34: 3}
    try:
        labels = np.array([code[int(v)] for v in epochs.events[:, 2]])
    except KeyError as exc:
        raise AdapterError("unexpected MOABB event codes; prefer local derivatives") from exc
    ch_names = list(epochs.ch_names)
    return EpochBatch(
        data=data,
        sfreq=float(epochs.info["sfreq"]),
        ch_names=ch_names,
        tmin=float(epochs.tmin),
        labels=labels,
        sessions=np.ones(len(labels), dtype=int),
        montage_xy=_montage_xy(ch_names),
        class_names=dict(CLASS_NAMES),
        subject_id=_subject_name(subject),
        condition=cond_key,
        dataset="nieto2022",
    )


def fetch_nieto(
    dest: str | Path,
    *,
    subject: int = 1,
    sessions: list[int] | None = None,
) -> Path:
    """Download Nieto 2022 derivatives for one subject via openneuro-py.

    The full dataset is tens of GB. This pulls only ``derivatives/sub-XX``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import openneuro
    except ImportError as exc:
        raise AdapterError(
            "pip install openneuro-py to download ds003626, or set EEGVIS_NIETO_ROOT "
            "to a local OpenNeuro copy that already has derivatives/."
        ) from exc

    name = _subject_name(subject)
    session_ids = sessions or [1, 2, 3]
    include = [f"derivatives/{name}/ses-0{ses}/**" for ses in session_ids]
    openneuro.download(dataset="ds003626", target_dir=str(dest), include=include)
    nested = dest / "ds003626" / "derivatives" / name
    if nested.exists() and not (dest / "derivatives" / name).exists():
        return dest / "ds003626"
    return dest
