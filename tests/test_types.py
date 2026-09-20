import numpy as np
import pytest

from eegvis.adapters.synthetic import make_synthetic
from eegvis.types import EpochBatch


def test_epoch_batch_rejects_bad_shape():
    with pytest.raises(ValueError):
        EpochBatch(data=np.zeros((2, 3)), sfreq=256.0, ch_names=["a", "b", "c"])


def test_times_match_tmin_sfreq():
    batch = make_synthetic(duration=1.0, sfreq=100.0, tmin=-0.5, n_channels=4, trials_per_class=2)
    assert batch.times[0] == pytest.approx(-0.5)
    assert batch.times[1] - batch.times[0] == pytest.approx(0.01)


def test_subset_trials_keeps_channels():
    batch = make_synthetic(n_channels=5, trials_per_class=3, seed=21)
    sub = batch.subset_trials(batch.labels == 0)
    assert sub.n_channels == batch.n_channels
    assert sub.n_trials == 3
    assert np.all(sub.labels == 0)
