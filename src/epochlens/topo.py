"""2-D scalp interpolation. Coordinates come from the batch, not a named montage."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator


def interpolate_topo(
    xy: np.ndarray,
    values: np.ndarray,
    *,
    n: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Xi, Yi, Zi)`` grids. Points outside a circular head are NaN."""
    xy = np.asarray(xy, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (n_channels, 2)")
    if values.shape != (xy.shape[0],):
        raise ValueError("values must be (n_channels,)")
    if xy.shape[0] < 3:
        raise ValueError("need at least three sensors")
    span = float(np.max(np.ptp(xy, axis=0)))
    pad = 0.15 * max(span, 1e-6)
    lo = xy.min(axis=0) - pad
    hi = xy.max(axis=0) + pad
    xi = np.linspace(lo[0], hi[0], n)
    yi = np.linspace(lo[1], hi[1], n)
    Xi, Yi = np.meshgrid(xi, yi)
    scale = float(np.ptp(values))
    smoothing = 0.0 if scale <= 0 else 1e-6 * scale
    rbf = RBFInterpolator(xy, values, kernel="thin_plate_spline", smoothing=smoothing)
    Zi = rbf(np.column_stack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    center = xy.mean(axis=0)
    radius = float(np.max(np.linalg.norm(xy - center, axis=1)) * 1.15)
    Zi[np.sqrt((Xi - center[0]) ** 2 + (Yi - center[1]) ** 2) > radius] = np.nan
    return Xi, Yi, Zi
