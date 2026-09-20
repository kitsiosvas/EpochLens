import numpy as np

from eegvis.adapters.synthetic import make_synthetic
from eegvis.waveforms import class_mean_sem, rms_channel_score, baseline_zscore


def test_class_mean_sem_shapes():
    batch = make_synthetic(n_channels=6, trials_per_class=8, seed=11)
    means, sems = class_mean_sem(batch)
    assert set(means) == {0, 1, 2, 3}
    for cls in means:
        assert means[cls].shape == (6, batch.n_times)
        assert sems[cls].shape == (6, batch.n_times)
        assert np.all(sems[cls] >= 0)


def test_planted_rhythm_visible_in_class_mean():
    batch = make_synthetic(n_channels=8, trials_per_class=12, seed=12)
    means, _ = class_mean_sem(batch)
    t = batch.times
    burst = (t >= 0.6) & (t <= 1.4)
    assert np.abs(means[0][0, burst]).mean() > np.abs(means[0][0, ~burst]).mean()


def test_rms_prefers_active_channels():
    batch = make_synthetic(n_channels=8, trials_per_class=10, seed=13)
    scores = rms_channel_score(batch, (0.6, 1.4))
    assert scores.shape == (8,)
    assert scores[0] > scores[5]


def test_sem_is_zero_for_single_trial():
    batch = make_synthetic(n_classes=2, trials_per_class=1, n_channels=4, seed=19)
    means, sems = class_mean_sem(batch)
    assert set(means) == {0, 1}
    for cls in means:
        assert np.all(sems[cls] == 0)
        assert np.all(np.isfinite(sems[cls]))


def test_baseline_zscore_zeros_baseline_mean():
    batch = make_synthetic(n_channels=6, trials_per_class=8, seed=16)
    z = baseline_zscore(batch, (0.0, 0.4))
    sl = (z.times >= 0.0) & (z.times <= 0.4)
    assert np.abs(z.data[:, :, sl].mean(axis=-1)).mean() < 0.05
