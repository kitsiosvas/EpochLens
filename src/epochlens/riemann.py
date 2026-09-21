"""SPD covariance geometry via pyRiemann, plus classical MDS for the explorer."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from epochlens.types import EpochBatch
from epochlens.windows import sample_span

_METRICS = {"logeuclid", "riemann"}


def _pairwise_distance(covs: np.ndarray, metric: str) -> np.ndarray:
    try:
        from pyriemann.geometry.distance import pairwise_distance
    except ImportError:  # pyRiemann < 0.12
        from pyriemann.utils.distance import pairwise_distance

    return pairwise_distance(covs, metric=metric)


def trial_covariances(
    batch: EpochBatch,
    window: tuple[float, float],
    ridge: float = 0.0,
    center: bool = True,
    estimator: str = "lwf",
) -> np.ndarray:
    """Trial covariances via :class:`pyriemann.estimation.Covariances`.

    Default estimator is Ledoit–Wolf shrinkage. ``ridge`` is an optional extra
    diagonal after estimation (usually leave at 0).
    """
    from pyriemann.estimation import Covariances

    sl = sample_span(batch.n_times, batch.sfreq, batch.tmin, window[0], window[1])
    x = np.asarray(batch.data[:, :, sl], dtype=np.float64)
    if x.shape[-1] < 2:
        raise ValueError("covariance window too short")
    if center:
        x = x - x.mean(axis=-1, keepdims=True)
    covs = np.asarray(Covariances(estimator=estimator).fit_transform(x), dtype=np.float64)
    if ridge:
        covs = covs + float(ridge) * np.eye(batch.n_channels)
    return covs


def pairwise_distances(covs: np.ndarray, metric: str = "logeuclid") -> np.ndarray:
    """Full distance matrix via pyRiemann (vectorized log-Euclid / Riemann)."""
    if metric not in _METRICS:
        raise ValueError("metric must be 'logeuclid' or 'riemann'")
    covs = np.asarray(covs, dtype=np.float64)
    return np.asarray(_pairwise_distance(covs, metric), dtype=np.float64)


def session_whiten(
    covs: np.ndarray,
    sessions: np.ndarray,
    metric: str = "logeuclid",
) -> np.ndarray:
    """Per-session mean removal via :class:`pyriemann.preprocessing.Whitening`."""
    from pyriemann.preprocessing import Whitening

    if metric not in _METRICS:
        raise ValueError("metric must be 'logeuclid' or 'riemann'")
    covs = np.asarray(covs, dtype=np.float64)
    sessions = np.asarray(sessions)
    out = np.empty_like(covs)
    for sid in np.unique(sessions):
        idx = np.flatnonzero(sessions == sid)
        out[idx] = Whitening(metric=metric).fit_transform(covs[idx])
    return out


def embed_mds(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical MDS (MATLAB ``cmdscale`` equivalent). Not a pyRiemann call."""
    d = np.asarray(distances, dtype=np.float64)
    n = d.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ (d**2) @ h
    w, v = eigh(b)
    order = np.argsort(w)[::-1]
    w = np.maximum(w[order[:n_components]], 0.0)
    return v[:, order[:n_components]] * np.sqrt(w)


def mds_stress(distances: np.ndarray, xy: np.ndarray) -> float:
    """Kruskal stress-1 of a classical MDS embedding.

    ``distances`` is the original pairwise matrix; ``xy`` is the 2-D (or
    ``n``-D) embedding. Returns a value in ``[0, inf)``.
    """
    d = np.asarray(distances, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distances must be a square matrix")
    if xy.ndim != 2 or xy.shape[0] != d.shape[0]:
        raise ValueError("xy rows must match distances")
    n = d.shape[0]
    if n < 2:
        return 0.0
    delta = np.sqrt(np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1))
    iu = np.triu_indices(n, k=1)
    num = float(np.sum((d[iu] - delta[iu]) ** 2))
    den = float(np.sum(d[iu] ** 2))
    if den <= 0.0:
        return 0.0 if num <= 0.0 else float("inf")
    return float(np.sqrt(num / den))
