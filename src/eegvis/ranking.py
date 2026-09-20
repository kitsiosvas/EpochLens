"""Channel scores from maps, plus cross-subject voting."""

from __future__ import annotations

import numpy as np


def score_channels(
    channel_map: np.ndarray,
    *,
    reduce_axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Reduce a per-channel map to one score per channel.

    ``channel_map`` must have channels on axis 0. Remaining axes are averaged
    unless ``reduce_axes`` is given (relative to the full array).
    """
    arr = np.asarray(channel_map, dtype=np.float64)
    if arr.ndim < 1:
        raise ValueError("channel_map must include a channel axis")
    if reduce_axes is None:
        if arr.ndim == 1:
            return arr
        return arr.mean(axis=tuple(range(1, arr.ndim)))
    return arr.mean(axis=reduce_axes)


def top_channels(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    k = min(int(k), scores.size)
    if k <= 0:
        return np.array([], dtype=int)
    return np.argsort(scores)[::-1][:k]


def vote_channels(top_index_lists: list[np.ndarray], n_channels: int) -> np.ndarray:
    """Histogram of how often each channel appears in a subject's top set."""
    votes = np.zeros(n_channels, dtype=np.int64)
    for picks in top_index_lists:
        idx = np.asarray(picks, dtype=int)
        idx = idx[(idx >= 0) & (idx < n_channels)]
        votes[idx] += 1
    return votes
