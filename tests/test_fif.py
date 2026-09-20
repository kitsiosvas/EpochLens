from pathlib import Path

import numpy as np
import pytest

from eegvis.adapters.base import AdapterError
from eegvis.adapters.fif import load_epochs_fif
from eegvis.adapters.synthetic import make_synthetic


def test_missing_fif_raises():
    with pytest.raises(AdapterError, match="not found"):
        load_epochs_fif("no-such-file-epo.fif")


def test_fif_roundtrip(tmp_path: Path):
    mne = pytest.importorskip("mne")
    batch = make_synthetic(n_channels=6, trials_per_class=4, duration=1.0, seed=15)
    info = mne.create_info(batch.ch_names, batch.sfreq, ch_types="eeg")
    events = np.column_stack(
        [
            np.arange(batch.n_trials) * 20,
            np.zeros(batch.n_trials, dtype=int),
            batch.labels,
        ]
    )
    event_id = {name: i for i, name in batch.class_names.items()}
    epochs = mne.EpochsArray(batch.data, info, events=events, tmin=batch.tmin, event_id=event_id)
    path = tmp_path / "demo-epo.fif"
    epochs.save(path, overwrite=True)
    loaded = load_epochs_fif(path)
    assert loaded.n_trials == batch.n_trials
    assert loaded.n_channels == batch.n_channels
    assert loaded.dataset == "mne-epochs"
    assert set(loaded.labels.tolist()) == {0, 1, 2, 3}
    np.testing.assert_allclose(loaded.data, batch.data)
