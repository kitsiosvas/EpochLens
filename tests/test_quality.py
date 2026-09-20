import numpy as np

from epochlens.adapters.synthetic import make_synthetic
from epochlens.quality import flag_bad_channels
from epochlens.ranking import top_channels


def test_flag_huge_channel():
    batch = make_synthetic(n_channels=8, trials_per_class=6, seed=17)
    batch.data[:, 7] *= 1e4
    bad = flag_bad_channels(batch)
    assert bad[7]


def test_synthetic_demo_keeps_planted_channels():
    batch = make_synthetic(n_channels=16, trials_per_class=12, seed=0)
    bad = flag_bad_channels(batch)
    assert not np.any(bad)


def test_top_channels_exclude():
    scores = np.array([1.0, 9.0, 3.0, 8.0])
    picks = top_channels(scores, 2, exclude=np.array([False, True, False, False]))
    assert list(picks) == [3, 2]
