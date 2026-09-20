"""Load any MNE Epochs FIF into ``EpochBatch``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from epochlens.adapters.base import AdapterError
from epochlens.types import EpochBatch

_SESSION_COLUMNS = ("session", "sessions", "sess", "run", "recording", "block")


def _montage_xy(ch_names: list[str], montage) -> np.ndarray | None:
    if montage is None:
        return None
    pos = montage.get_positions().get("ch_pos") or {}
    xy = np.full((len(ch_names), 2), np.nan, dtype=np.float64)
    n_located = 0
    for i, name in enumerate(ch_names):
        if name not in pos:
            continue
        xyz = np.asarray(pos[name], dtype=np.float64).reshape(-1)
        if xyz.size < 2 or not np.isfinite(xyz[:2]).all():
            continue
        xy[i, 0] = float(xyz[0])
        xy[i, 1] = float(xyz[1])
        n_located += 1
    if n_located == 0:
        return None
    return xy


def _sessions_from_metadata(epochs, n_trials: int) -> np.ndarray | None:
    metadata = getattr(epochs, "metadata", None)
    if metadata is None:
        return None
    columns = getattr(metadata, "columns", None)
    if columns is None:
        return None
    by_lower = {str(name).lower(): name for name in columns}
    chosen = next((by_lower[key] for key in _SESSION_COLUMNS if key in by_lower), None)
    if chosen is None:
        return None
    values = np.asarray(metadata[chosen])
    if values.shape[0] != n_trials:
        return None
    _codes, inverse = np.unique(values, return_inverse=True)
    return np.asarray(inverse, dtype=int)


def load_epochs_fif(path: str | Path) -> EpochBatch:
    """Read a ``*-epo.fif`` (or equivalent) produced by MNE."""
    path = Path(path)
    if not path.exists():
        raise AdapterError(f"epochs file not found: {path}")
    try:
        import mne
    except ImportError as exc:
        raise AdapterError("install mne to load epochs FIF files") from exc

    try:
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    except Exception as exc:
        raise AdapterError(f"could not read epochs: {path}: {exc}") from exc

    eeg_idx = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
    if eeg_idx.size == 0:
        raise AdapterError("no EEG channels in epochs file")
    epochs.pick(eeg_idx)

    data = epochs.get_data(copy=True)
    ch_names = list(epochs.ch_names)
    raw_labels = np.asarray(epochs.events[:, 2], dtype=int)
    codes = np.unique(raw_labels)
    remap = {int(code): i for i, code in enumerate(codes)}
    labels = np.array([remap[int(v)] for v in raw_labels], dtype=int)
    inv_event = {int(v): str(k) for k, v in (epochs.event_id or {}).items()}
    class_names = {i: inv_event.get(int(code), str(code)) for code, i in remap.items()}
    try:
        montage = epochs.get_montage()
    except Exception:
        montage = None
    return EpochBatch(
        data=data,
        sfreq=float(epochs.info["sfreq"]),
        ch_names=ch_names,
        tmin=float(epochs.tmin),
        labels=labels,
        sessions=_sessions_from_metadata(epochs, data.shape[0]),
        montage_xy=_montage_xy(ch_names, montage),
        class_names=class_names,
        subject_id=path.stem,
        condition=None,
        dataset="mne-epochs",
    )
