"""Tiny disk cache for expensive representations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def cache_root() -> Path:
    return Path(os.environ.get("EEGVIS_CACHE", Path.home() / ".eegvis" / "cache"))


def _key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def array_fingerprint(data: np.ndarray) -> dict:
    d = np.asarray(data)
    return {
        "shape": list(d.shape),
        "mean": float(d.mean()),
        "std": float(d.std()),
        "first": float(d.reshape(-1)[0]),
        "last": float(d.reshape(-1)[-1]),
    }


def save_npz(name: str, payload: dict, arrays: dict[str, np.ndarray]) -> Path:
    path = cache_root()
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{name}_{_key(payload)}.npz"
    np.savez_compressed(file, **arrays)
    return file


def load_npz(name: str, payload: dict) -> dict[str, np.ndarray] | None:
    file = cache_root() / f"{name}_{_key(payload)}.npz"
    if not file.exists():
        return None
    with np.load(file) as z:
        return {k: z[k] for k in z.files}
