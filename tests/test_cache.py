from pathlib import Path

import numpy as np

from epochlens.adapters.synthetic import make_synthetic
from epochlens.cwt import class_relative_scalograms, mean_cwt_power


def test_pick_preserves_trials_and_montage():
    batch = make_synthetic(n_channels=8, trials_per_class=3, seed=8)
    sub = batch.pick([0, 3, 5])
    assert sub.n_channels == 3
    assert sub.n_trials == batch.n_trials
    assert sub.ch_names == [batch.ch_names[i] for i in (0, 3, 5)]
    assert sub.montage_xy.shape == (3, 2)
    np.testing.assert_allclose(sub.data, batch.data[:, [0, 3, 5]])


def test_mean_cwt_cache_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EPOCHLENS_CACHE", str(tmp_path))
    batch = make_synthetic(n_channels=6, trials_per_class=4, seed=9)
    a, fa, ta = mean_cwt_power(batch, voices_per_octave=6, decim=4, use_cache=True)
    files = list(tmp_path.glob("mean_cwt_*.npz"))
    assert files
    b, fb, tb = mean_cwt_power(batch, voices_per_octave=6, decim=4, use_cache=True)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(fa, fb)
    np.testing.assert_allclose(ta, tb)


def test_class_relative_cache_roundtrip_distinct_from_mean_cwt(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EPOCHLENS_CACHE", str(tmp_path))
    batch = make_synthetic(n_channels=4, trials_per_class=3, seed=11)
    baseline = (0.0, 0.4)
    kwargs = dict(show_n=2, voices_per_octave=4, decim=4, use_cache=True)
    rel_a, ta, fa, na = class_relative_scalograms(batch, baseline, **kwargs)
    assert list(tmp_path.glob("mean_cwt_*.npz")) == []
    rel_files = list(tmp_path.glob("class_rel_cwt_*.npz"))
    assert rel_files
    picked = batch.pick([0, 1])
    mean_cwt_power(picked, voices_per_octave=4, decim=4, use_cache=True)
    mean_files = list(tmp_path.glob("mean_cwt_*.npz"))
    assert mean_files
    assert {p.name for p in mean_files}.isdisjoint({p.name for p in rel_files})
    rel_b, tb, fb, nb = class_relative_scalograms(batch, baseline, **kwargs)
    assert na == nb
    np.testing.assert_allclose(ta, tb)
    np.testing.assert_allclose(fa, fb)
    assert set(rel_a) == set(rel_b)
    for cls in rel_a:
        np.testing.assert_allclose(rel_a[cls], rel_b[cls])
