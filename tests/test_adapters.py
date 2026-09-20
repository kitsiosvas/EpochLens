import pytest

from eegvis.adapters.base import AdapterError
from eegvis.adapters.nieto import load_nieto
from eegvis.adapters.synthetic import make_synthetic


def test_synthetic_label_balance():
    batch = make_synthetic(n_classes=4, trials_per_class=5)
    assert batch.dataset == "synthetic"
    assert batch.n_trials == 20
    assert set(batch.labels.tolist()) == {0, 1, 2, 3}


def test_nieto_unknown_condition():
    with pytest.raises(AdapterError):
        load_nieto(1, condition="whisper")
