import numpy as np

from epochlens.adapters.synthetic import make_synthetic
from epochlens.explorer.summary import dataset_facts, facts_line
from epochlens.topo import can_draw_scalp


def test_dummy_batch_facts():
    batch = make_synthetic(n_channels=16, trials_per_class=24, duration=2.0, sfreq=256.0)
    facts = dataset_facts(batch)
    assert facts["duration"] == "2 s"
    assert facts["channels"] == "16"
    assert facts["trials"] == "96"
    assert facts["sfreq"] == "256 Hz"
    assert facts["classes"] == "4"
    assert "10 Hz" in facts["class_detail"]
    assert facts["montage"] == "yes"
    line = facts_line(batch)
    assert "2 s epochs" in line
    assert "16 channels" in line
    assert "96 trials" in line


def test_montage_fact_needs_three_located_sensors():
    batch = make_synthetic(n_channels=4, trials_per_class=2, seed=1)
    assert can_draw_scalp(batch.montage_xy)
    assert dataset_facts(batch)["montage"] == "yes"

    nan_xy = np.full((4, 2), np.nan)
    nan_batch = batch.copy_with(montage_xy=nan_xy)
    assert dataset_facts(nan_batch)["montage"] == "no"

    two = nan_xy.copy()
    two[0] = [1.0, 0.0]
    two[1] = [0.0, 1.0]
    two_batch = batch.copy_with(montage_xy=two)
    assert not can_draw_scalp(two)
    assert dataset_facts(two_batch)["montage"] == "no"

    three = two.copy()
    three[2] = [-1.0, 0.0]
    three_batch = batch.copy_with(montage_xy=three)
    assert can_draw_scalp(three)
    assert dataset_facts(three_batch)["montage"] == "yes"
