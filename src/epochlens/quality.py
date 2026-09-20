"""Cheap channel-quality flags. Dataset-agnostic."""

from __future__ import annotations

import numpy as np

from epochlens.types import EpochBatch


def flag_bad_channels(
    batch: EpochBatch,
    *,
    mad_z: float = 8.0,
    trial_ratio: float = 20.0,
) -> np.ndarray:
    """True for dead, amplitude-outlier, or single-trial-dominated channels."""
    rms = np.sqrt(np.mean(batch.data * batch.data, axis=-1))
    ch_rms = np.median(rms, axis=0)
    dead = ch_rms < 1e-18
    peak = rms.max(axis=0)
    noisy = peak / np.maximum(ch_rms, 1e-18) > trial_ratio
    log_rms = np.log10(np.maximum(ch_rms, 1e-18))
    med = np.median(log_rms)
    mad = np.median(np.abs(log_rms - med)) * 1.4826
    # Floor MAD so a tight noise cloud does not mark slightly stronger sensors.
    extreme = np.abs(log_rms - med) > mad_z * max(float(mad), 0.15)
    return dead | noisy | extreme
