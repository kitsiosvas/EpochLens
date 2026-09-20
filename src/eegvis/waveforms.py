"""Time-domain traces. Windows are caller-supplied seconds."""

from __future__ import annotations

import numpy as np

from eegvis.types import EpochBatch
from eegvis.windows import sample_span


def class_mean_sem(
    batch: EpochBatch,
) -> tuple[dict[int | None, np.ndarray], dict[int | None, np.ndarray]]:
    """Trial-mean and SEM per class, shape ``(n_channels, n_times)``.

    Unlabeled batches use key ``None`` for the grand average.
    """
    means: dict[int | None, np.ndarray] = {}
    sems: dict[int | None, np.ndarray] = {}
    if batch.labels is None:
        means[None] = batch.data.mean(axis=0)
        sems[None] = _sem(batch.data)
        return means, sems
    for cls in np.unique(batch.labels):
        x = batch.data[batch.labels == cls]
        key = int(cls)
        means[key] = x.mean(axis=0)
        sems[key] = _sem(x)
    return means, sems


def _sem(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n < 2:
        return np.zeros(x.shape[1:], dtype=np.float64)
    return x.std(axis=0, ddof=1) / np.sqrt(n)


def rms_channel_score(batch: EpochBatch, window: tuple[float, float]) -> np.ndarray:
    """RMS over trials and time in ``window``, one score per channel."""
    sl = sample_span(batch.n_times, batch.sfreq, batch.tmin, window[0], window[1])
    x = batch.data[:, :, sl]
    return np.sqrt((x * x).mean(axis=(0, 2)))


def baseline_zscore(batch: EpochBatch, baseline: tuple[float, float]) -> EpochBatch:
    """Per-trial, per-channel z-score using the baseline window."""
    sl = sample_span(batch.n_times, batch.sfreq, batch.tmin, baseline[0], baseline[1])
    base = batch.data[:, :, sl]
    mu = base.mean(axis=-1, keepdims=True)
    sd = np.maximum(base.std(axis=-1, keepdims=True), 1e-12)
    return batch.copy_with(data=(batch.data - mu) / sd)
