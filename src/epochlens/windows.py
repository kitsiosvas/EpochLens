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


def default_windows(
    dataset: str | None, tmin: float, tmax: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Heuristic analysis and baseline windows from the epoch span.

    The synthetic demo uses a window around the planted rhythm. Any other
    epochs get a 20% baseline and the rest as the analysis window.
    """
    if dataset == "synthetic":
        return (0.6, 1.4), (tmin, min(tmin + 0.5, tmin + 0.4 * (tmax - tmin)))
    span = max(tmax - tmin, 1e-6)
    baseline = (tmin, tmin + 0.2 * span)
    window = (tmin + 0.2 * span, tmax)
    return window, baseline
