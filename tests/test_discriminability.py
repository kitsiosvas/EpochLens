import numpy as np

from epochlens.adapters.synthetic import make_synthetic
from epochlens.discriminability import aggregate_pairs, pairwise_maps, peak_map
from epochlens.ranking import ranking_table, score_channels, top_channels, vote_channels
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


def test_session_channel_votes_needs_two_sessions():
    from epochlens.ranking import session_channel_votes

    batch = make_synthetic(n_channels=8, trials_per_class=10, seed=3)
    votes = session_channel_votes(batch, (0.6, 1.4), (0.0, 0.4), 3)
    assert votes is not None
    assert votes.shape == (batch.n_channels,)
    assert int(votes.sum()) >= 2 * 3
    assert votes.max() >= 2

    one = batch.copy_with(sessions=np.ones(batch.n_trials, dtype=int))
    assert session_channel_votes(one, (0.6, 1.4), (0.0, 0.4), 3) is None


def test_peak_map_index():
    ave = np.zeros((3, 5))
    ave[1, 3] = 2.0
    times = np.linspace(0.0, 1.0, 5)
    t, ch, ti = peak_map(ave, times)
    assert ch == 1
    assert ti == 3
    assert t == times[3]


def test_ranking_table_sorted_and_subset_flag():
    scores = np.array([0.1, 0.5, 0.2])
    names = ["a", "b", "c"]
    picks = np.array([1])
    rows = ranking_table(scores, names, picks, votes=np.array([0, 2, 1]))
    assert [r["channel"] for r in rows] == ["b", "c", "a"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["in visualization subset"] == "yes"
    assert rows[1]["in visualization subset"] == "no"
    assert rows[0]["session votes"] == 2
    assert rows[2]["session votes"] == 0
