import numpy as np
import pytest

from epochlens.topo import can_draw_scalp, interpolate_topo, located_mask


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


def test_located_mask_and_scalp_gate():
    xy = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan], [-1.0, 0.0]])
    mask = located_mask(xy)
    np.testing.assert_array_equal(mask, [True, True, False, True])
    assert can_draw_scalp(xy)
    assert not can_draw_scalp(None)
    assert not can_draw_scalp(np.full((4, 2), np.nan))
    two = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan], [np.nan, 0.0]])
    assert not can_draw_scalp(two)


def test_interpolate_topo_skips_nan_sensors():
    xy = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [np.nan, np.nan]])
    values = np.array([0.0, 0.0, 0.0, 100.0])
    Xi, Yi, Zi = interpolate_topo(xy, values, n=30)
    assert np.any(np.isfinite(Zi))
    finite = Zi[np.isfinite(Zi)]
    assert np.std(finite) < 1.0


def test_interpolate_topo_requires_three_located():
    xy = np.array([[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan], [np.nan, np.nan]])
    with pytest.raises(ValueError, match="need at least three sensors"):
        interpolate_topo(xy, np.ones(4))

