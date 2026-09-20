import numpy as np

from eegvis.adapters.synthetic import make_synthetic
from eegvis.discriminability import aggregate_pairs, pairwise_maps
from eegvis.ranking import score_channels, top_channels, vote_channels
from eegvis.windows import time_mask


def test_pairwise_maps_rank_planted_channel():
    batch = make_synthetic(n_channels=8, trials_per_class=20, seed=3)
    mask = time_mask(batch.times, 0.6, 1.4)
    maps, pairs = pairwise_maps(batch.data[:, :, mask], batch.labels, method="wilcoxon")
    assert maps.shape[0] == len(pairs) == 6
    ave = aggregate_pairs(maps, "mean")
    scores = score_channels(ave)
    tops = top_channels(scores, 3)
    assert 0 in tops or 1 in tops


def test_vote_channels():
    votes = vote_channels([np.array([0, 1, 2]), np.array([0, 3])], n_channels=5)
    assert votes[0] == 2
    assert votes[1] == 1
    assert votes[4] == 0
