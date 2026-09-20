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


def top_channels(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).copy()
    if exclude is not None:
        ex = np.asarray(exclude)
        if ex.dtype == bool:
            scores[ex] = -np.inf
        else:
            scores[ex.astype(int)] = -np.inf
    finite = np.isfinite(scores)
    k = min(int(k), int(finite.sum()))
    if k <= 0:
        return np.array([], dtype=int)
    return np.argsort(scores)[::-1][:k]


def prepare_ranking(
    batch,
    window: tuple[float, float],
    baseline: tuple[float, float],
    top_k: int,
    *,
    time_bins: int = 100,
):
    """Z-score, drop bad channels, rank. Returns explorer-ready pieces."""
    from epochlens.discriminability import aggregate_pairs, pairwise_maps
    from epochlens.quality import flag_bad_channels
    from epochlens.waveforms import baseline_zscore, rms_channel_score
    from epochlens.windows import time_mask

    bad = flag_bad_channels(batch)
    zbatch = baseline_zscore(batch, baseline)
    k = min(int(top_k), int((~bad).sum()) or batch.n_channels)
    if batch.labels is None:
        scores = rms_channel_score(zbatch, window)
        scores[bad] = -np.inf
        picks = top_channels(scores, k)
        if picks.size == 0:
            picks = np.arange(min(int(top_k), batch.n_channels))
        return zbatch, scores, picks, None, None, [], bad
    idx = np.flatnonzero(time_mask(zbatch.times, window[0], window[1]))
    if idx.size == 0:
        raise ValueError("analysis window empty")
    step = max(1, idx.size // int(time_bins))
    idx = idx[::step]
    maps, pairs = pairwise_maps(zbatch.data[:, :, idx], batch.labels, method="wilcoxon")
    ave = aggregate_pairs(maps, "mean")
    scores = score_channels(ave)
    scores[bad] = -np.inf
    picks = top_channels(scores, k)
    if picks.size == 0:
        picks = np.arange(min(int(top_k), batch.n_channels))
    return zbatch, scores, picks, ave, zbatch.times[idx], pairs, bad


def vote_channels(top_index_lists: list[np.ndarray], n_channels: int) -> np.ndarray:
    """Histogram of how often each channel appears in a subject's top set."""
    votes = np.zeros(n_channels, dtype=np.int64)
    for picks in top_index_lists:
        idx = np.asarray(picks, dtype=int)
        idx = idx[(idx >= 0) & (idx < n_channels)]
        votes[idx] += 1
    return votes
