"""Load any MNE Epochs FIF into ``EpochBatch``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from epochlens.adapters.base import AdapterError
from epochlens.types import EpochBatch


def _montage_xy(ch_names: list[str], montage) -> np.ndarray | None:
    if montage is None:
        return None
    pos = montage.get_positions().get("ch_pos") or {}
    xy = []
    for name in ch_names:
        if name not in pos:
            return None
        xyz = pos[name]
        xy.append([float(xyz[0]), float(xyz[1])])
    return np.asarray(xy, dtype=np.float64)


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
        sessions=None,
        montage_xy=_montage_xy(ch_names, montage),
        class_names=class_names,
        subject_id=path.stem,
        condition=None,
        dataset="mne-epochs",
    )
