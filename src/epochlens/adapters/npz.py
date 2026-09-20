"""Save / load ``EpochBatch`` as a compressed NPZ. No MNE required."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from epochlens.adapters.base import AdapterError
from epochlens.types import EpochBatch


def save_epochs(batch: EpochBatch, path: str | Path) -> Path:
    path = Path(path)
    payload: dict[str, object] = {
        "data": batch.data,
        "sfreq": np.float64(batch.sfreq),
        "tmin": np.float64(batch.tmin),
        "ch_names": np.asarray(batch.ch_names, dtype=object),
        "class_keys": np.asarray(list(batch.class_names.keys()), dtype=np.int64),
        "class_values": np.asarray(list(batch.class_names.values()), dtype=object),
        "subject_id": np.asarray(batch.subject_id or ""),
        "condition": np.asarray(batch.condition or ""),
        "dataset": np.asarray(batch.dataset or "npz"),
    }
    if batch.labels is not None:
        payload["labels"] = batch.labels
    if batch.sessions is not None:
        payload["sessions"] = batch.sessions
    if batch.montage_xy is not None:
        payload["montage_xy"] = batch.montage_xy
    np.savez_compressed(path, **payload)
    return path


def load_epochs(path: str | Path) -> EpochBatch:
    path = Path(path)
    if not path.exists():
        raise AdapterError(f"epochs file not found: {path}")
    try:
        with np.load(path, allow_pickle=True) as z:
            keys = set(z.files)
            if "data" not in keys or "sfreq" not in keys or "ch_names" not in keys:
                raise AdapterError(f"npz is not an EpochBatch: {path}")
            class_names: dict[int, str] = {}
            if "class_keys" in keys and "class_values" in keys:
                class_names = {int(k): str(v) for k, v in zip(z["class_keys"], z["class_values"])}
            sid = str(z["subject_id"]) if "subject_id" in keys else path.stem
            cond = str(z["condition"]) if "condition" in keys else ""
            dset = str(z["dataset"]) if "dataset" in keys else "npz"
            return EpochBatch(
                data=z["data"],
                sfreq=float(z["sfreq"]),
                ch_names=[str(n) for n in z["ch_names"]],
                tmin=float(z["tmin"]) if "tmin" in keys else 0.0,
                labels=z["labels"] if "labels" in keys else None,
                sessions=z["sessions"] if "sessions" in keys else None,
                montage_xy=z["montage_xy"] if "montage_xy" in keys else None,
                class_names=class_names,
                subject_id=sid or None,
                condition=cond or None,
                dataset=dset or "npz",
            )
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"could not read npz: {path}: {exc}") from exc
