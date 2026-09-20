import numpy as np

from eegvis.topo import interpolate_topo


def test_interpolate_topo_peaks_near_source():
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    values = np.zeros(12)
    values[0] = 1.0
    Xi, Yi, Zi = interpolate_topo(xy, values, n=40)
    assert Zi.shape == Xi.shape
    peak = np.nanargmax(Zi)
    py, px = np.unravel_index(peak, Zi.shape)
    dist = np.hypot(Xi[py, px] - xy[0, 0], Yi[py, px] - xy[0, 1])
    assert dist < 0.6
    center = xy.mean(axis=0)
    inside = np.hypot(Xi - center[0], Yi - center[1]) < 1.05
    assert np.mean(np.isfinite(Zi[inside])) > 0.9


def test_flat_map_has_no_spoke_artifacts():
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    values = np.ones(16)
    Xi, Yi, Zi = interpolate_topo(xy, values, n=50)
    center = xy.mean(axis=0)
    inside = np.hypot(Xi - center[0], Yi - center[1]) < 1.0
    field = Zi[inside]
    assert np.mean(np.isfinite(field)) > 0.9
    field = field[np.isfinite(field)]
    assert np.std(field) / np.mean(field) < 0.08
