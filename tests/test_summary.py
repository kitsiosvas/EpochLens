from eegvis.adapters.synthetic import make_synthetic
from eegvis.explorer.summary import dataset_facts, facts_line


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
