"""Pairwise class discriminability. Vectorized across features."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind


def _as_trials_features(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    if x.ndim < 2:
        raise ValueError("expected (n_trials, ...features)")
    n_trials = x.shape[0]
    feat_shape = x.shape[1:]
    return x.reshape(n_trials, -1), feat_shape


def pairwise_maps(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "wilcoxon",
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Absolute pairwise statistic for every feature.

    Parameters
    ----------
    features
        ``(n_trials, ...)``; remaining axes are flattened then restored.
    labels
        Shape ``(n_trials,)``.
    method
        ``"wilcoxon"`` (Mann–Whitney U z, abs) or ``"ttest"`` (abs t).

    Returns
    -------
    maps
        ``(n_pairs, *feature_shape)``.
    pairs
        List of ``(class_a, class_b)`` in the same order.
    """
    labels = np.asarray(labels)
    flat, feat_shape = _as_trials_features(np.asarray(features, dtype=np.float64))
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("need at least two classes")
    pairs = list(combinations(classes.tolist(), 2))
    maps = np.empty((len(pairs), flat.shape[1]), dtype=np.float64)
    for i, (a, b) in enumerate(pairs):
        xa = flat[labels == a]
        xb = flat[labels == b]
        if xa.shape[0] < 2 or xb.shape[0] < 2:
            maps[i] = 0.0
            continue
        if method == "wilcoxon":
            res = mannwhitneyu(xa, xb, axis=0, alternative="two-sided")
            n1, n2 = xa.shape[0], xb.shape[0]
            # Convert U to |z| under the null (tie-corrected via scipy's p if needed).
            mu = n1 * n2 / 2.0
            sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
            z = (res.statistic - mu) / np.maximum(sigma, 1e-12)
            maps[i] = np.abs(z)
        elif method == "ttest":
            res = ttest_ind(xa, xb, axis=0, equal_var=False, nan_policy="omit")
            maps[i] = np.abs(np.nan_to_num(res.statistic, nan=0.0))
        else:
            raise ValueError("method must be 'wilcoxon' or 'ttest'")
    return maps.reshape((len(pairs),) + feat_shape), [(int(a), int(b)) for a, b in pairs]


def aggregate_pairs(maps: np.ndarray, how: str = "mean") -> np.ndarray:
    if how == "mean":
        return maps.mean(axis=0)
    if how == "max":
        return maps.max(axis=0)
    raise ValueError("how must be 'mean' or 'max'")


def peak_map(ave: np.ndarray, times: np.ndarray) -> tuple[float, int, int]:
    """Channel and time of the maximum mean |z|.

    ``ave`` is ``(n_channels, n_times)``. Returns ``(peak_time, channel_index,
    time_index)``. Non-finite entries are ignored.
    """
    ave = np.asarray(ave, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if ave.ndim != 2:
        raise ValueError("ave must be (n_channels, n_times)")
    if times.shape != (ave.shape[1],):
        raise ValueError("times must match ave's time axis")
    if not np.any(np.isfinite(ave)):
        raise ValueError("ave has no finite values")
    filled = np.where(np.isfinite(ave), ave, -np.inf)
    ch, t = np.unravel_index(int(np.argmax(filled)), ave.shape)
    return float(times[int(t)]), int(ch), int(t)
