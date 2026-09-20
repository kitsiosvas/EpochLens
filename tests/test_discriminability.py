import numpy as np

from epochlens.adapters.synthetic import make_synthetic
from epochlens.discriminability import aggregate_pairs, pairwise_maps
from epochlens.ranking import score_channels, top_channels, vote_channels
from epochlens.windows import time_mask


def test_pairwise_maps_rank_planted_channel():
    batch = make_synthetic(n_channels=8, trials_per_class=20, seed=3)
    mask = time_mask(batch.times, 0.6, 1.4)
    maps, pairs = pairwise_maps(batch.data[:, :, mask], batch.labels, method="wilcoxon")
    assert maps.shape[0] == len(pairs) == 6
    ave = aggregate_pairs(maps, "mean")
    scores = score_channels(ave)
    tops = top_channels(scores, 3)
    assert 0 in tops or 1 in tops


def test_pairwise_maps_ttest_finite():
    batch = make_synthetic(n_channels=6, trials_per_class=10, seed=5)
    mask = time_mask(batch.times, 0.6, 1.4)
    maps, pairs = pairwise_maps(batch.data[:, :, mask], batch.labels, method="ttest")
    assert maps.shape[0] == len(pairs)
    assert np.all(np.isfinite(maps))
    assert np.all(maps >= 0)


def test_vote_channels():
    votes = vote_channels([np.array([0, 1, 2]), np.array([0, 3])], n_channels=5)
    assert votes[0] == 2
    assert votes[1] == 1
    assert votes[4] == 0
