"""Time-window helpers. Windows are always caller-supplied seconds."""

from __future__ import annotations

import numpy as np


def time_mask(times: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    return (times >= tmin) & (times <= tmax)


def sample_span(n_times: int, sfreq: float, tmin_epoch: float, tmin: float, tmax: float) -> slice:
    start = int(np.round((tmin - tmin_epoch) * sfreq))
    stop = int(np.round((tmax - tmin_epoch) * sfreq))
    start = max(0, start)
    stop = min(n_times, max(start + 1, stop))
    return slice(start, stop)
