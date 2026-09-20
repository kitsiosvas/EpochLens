from pathlib import Path

import numpy as np
import pytest

from epochlens.adapters.base import AdapterError
from epochlens.adapters.npz import load_epochs, save_epochs
from epochlens.adapters.synthetic import make_synthetic


def test_missing_npz_raises():
    with pytest.raises(AdapterError, match="not found"):
        load_epochs("no-such-file.npz")


def test_npz_roundtrip(tmp_path: Path):
    batch = make_synthetic(n_channels=6, trials_per_class=4, duration=1.0, seed=19)
    path = tmp_path / "demo.npz"
    save_epochs(batch, path)
    loaded = load_epochs(path)
    assert loaded.n_trials == batch.n_trials
    assert loaded.n_channels == batch.n_channels
    assert loaded.class_names[0] == "10 Hz"
    np.testing.assert_allclose(loaded.data, batch.data)
    np.testing.assert_array_equal(loaded.labels, batch.labels)
