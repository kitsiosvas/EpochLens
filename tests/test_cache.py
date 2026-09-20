from pathlib import Path

import numpy as np

from epochlens.adapters.synthetic import make_synthetic
from epochlens.cwt import mean_cwt_power


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
