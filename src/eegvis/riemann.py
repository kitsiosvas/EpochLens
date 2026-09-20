"""SPD covariance geometry: log-Euclidean (fast) and affine-invariant Riemann."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, expm, logm, sqrtm

from eegvis.types import EpochBatch
from eegvis.windows import sample_span


def _sym(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x + np.swapaxes(x, -1, -2))


def _logm(c: np.ndarray) -> np.ndarray:
    return _sym(np.real(logm(c)))


def _expm(c: np.ndarray) -> np.ndarray:
    return _sym(np.real(expm(c)))


def _invsqrtm(c: np.ndarray) -> np.ndarray:
    w, v = eigh(_sym(c))
    w = np.clip(w, 1e-12, None)
    return (v * (1.0 / np.sqrt(w))) @ v.T


def trial_covariances(
    batch: EpochBatch,
    window: tuple[float, float],
    ridge: float = 1e-4,
    center: bool = True,
) -> np.ndarray:
    """Trace-normalized covariances, shape ``(n_trials, n_channels, n_channels)``."""
    sl = sample_span(batch.n_times, batch.sfreq, batch.tmin, window[0], window[1])
    x = batch.data[:, :, sl]
    if x.shape[-1] < 2:
        raise ValueError("covariance window too short")
    if center:
        x = x - x.mean(axis=-1, keepdims=True)
    cov = np.einsum("tci,tdi->tcd", x, x, optimize=True) / x.shape[-1]
    traces = np.trace(cov, axis1=1, axis2=2)
    cov = cov / np.maximum(traces, 1e-12)[:, None, None]
    eye = np.eye(batch.n_channels)
    return cov + ridge * eye


def pairwise_distances(covs: np.ndarray, metric: str = "logeuclid") -> np.ndarray:
    """Full distance matrix ``(n_trials, n_trials)``."""
    covs = np.asarray(covs, dtype=np.float64)
    n = covs.shape[0]
    if metric == "logeuclid":
        logs = np.stack([_logm(c) for c in covs], axis=0)
        flat = logs.reshape(n, -1)
        # Frobenius distance on matrix logs.
        grams = flat @ flat.T
        diag = np.diag(grams)
        d2 = np.maximum(diag[:, None] + diag[None, :] - 2.0 * grams, 0.0)
        return np.sqrt(d2)
    if metric == "riemann":
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                w = eigh(covs[i], covs[j], eigvals_only=True)
                w = np.clip(w, 1e-12, None)
                d = float(np.sqrt(np.sum(np.log(w) ** 2)))
                dist[i, j] = dist[j, i] = d
        return dist
    raise ValueError("metric must be 'logeuclid' or 'riemann'")


def mean_spd(covs: np.ndarray, metric: str = "logeuclid", max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    covs = np.asarray(covs, dtype=np.float64)
    if metric == "logeuclid":
        logs = np.stack([_logm(c) for c in covs], axis=0)
        return _expm(logs.mean(axis=0))
    if metric == "riemann":
        mu = _sym(covs.mean(axis=0))
        for _ in range(max_iter):
            isqrt = _invsqrtm(mu)
            tangents = np.stack([_logm(isqrt @ c @ isqrt) for c in covs], axis=0)
            tmean = tangents.mean(axis=0)
            if np.linalg.norm(tmean, ord="fro") < tol:
                break
            sqrt_mu = _sym(np.real(sqrtm(mu)))
            mu = _sym(sqrt_mu @ _expm(tmean) @ sqrt_mu)
        return mu
    raise ValueError("metric must be 'logeuclid' or 'riemann'")


def session_whiten(
    covs: np.ndarray,
    sessions: np.ndarray,
    metric: str = "logeuclid",
) -> np.ndarray:
    """Per-session mean removal: ``C <- R^{-1/2} C R^{-1/2}``."""
    covs = np.asarray(covs, dtype=np.float64)
    sessions = np.asarray(sessions)
    out = np.empty_like(covs)
    for sid in np.unique(sessions):
        idx = np.flatnonzero(sessions == sid)
        ref = mean_spd(covs[idx], metric=metric)
        isqrt = _invsqrtm(ref)
        for i in idx:
            out[i] = _sym(isqrt @ covs[i] @ isqrt)
    return out


def embed_mds(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Classical MDS (MATLAB ``cmdscale`` equivalent)."""
    d = np.asarray(distances, dtype=np.float64)
    n = d.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ (d ** 2) @ h
    w, v = eigh(b)
    order = np.argsort(w)[::-1]
    w = np.maximum(w[order[:n_components]], 0.0)
    return v[:, order[:n_components]] * np.sqrt(w)
