from pathlib import Path

import numpy as np
import pytest

from epochlens.adapters.base import AdapterError
from epochlens.adapters.fif import load_epochs_fif
from epochlens.adapters.synthetic import make_synthetic
from epochlens.topo import can_draw_scalp, interpolate_topo


def _save_epochs(tmp_path: Path, epochs, name: str = "demo-epo.fif") -> Path:
    path = tmp_path / name
    epochs.save(path, overwrite=True)
    return path


def _epochs_array(mne, data, ch_names, sfreq, tmin, labels, *, ch_types="eeg", montage=None, metadata=None):
    info = mne.create_info(list(ch_names), sfreq, ch_types=ch_types)
    if montage is not None:
        info.set_montage(montage, on_missing="ignore")
    n_trials = data.shape[0]
    events = np.column_stack(
        [
            np.arange(n_trials) * 20,
            np.zeros(n_trials, dtype=int),
            labels,
        ]
    )
    event_id = {str(int(i)): int(i) for i in np.unique(labels)}
    epochs = mne.EpochsArray(data, info, events=events, tmin=tmin, event_id=event_id, verbose="ERROR")
    if metadata is not None:
        epochs.metadata = metadata
    return epochs


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
    assert loaded.sessions is None
    assert loaded.montage_xy is None


def test_fif_keeps_eeg_only(tmp_path: Path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(3)
    n_trials, n_times = 8, 64
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    data = rng.standard_normal((n_trials, 5, n_times))
    ch_names = ["C3", "C4", "Cz", "EOG", "STI 014"]
    ch_types = ["eeg", "eeg", "eeg", "eog", "stim"]
    epochs = _epochs_array(
        mne, data, ch_names, 256.0, 0.0, labels, ch_types=ch_types
    )
    loaded = load_epochs_fif(_save_epochs(tmp_path, epochs))
    assert loaded.n_channels == 3
    assert loaded.ch_names == ["C3", "C4", "Cz"]
    np.testing.assert_allclose(loaded.data, data[:, :3, :])


def test_fif_drops_meg_keeps_eeg(tmp_path: Path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(2)
    n_trials, n_times = 6, 48
    labels = np.array([0, 0, 1, 1, 0, 1])
    data = rng.standard_normal((n_trials, 4, n_times))
    ch_names = ["C3", "C4", "MEG 0111", "MEG 0112"]
    ch_types = ["eeg", "eeg", "mag", "mag"]
    epochs = _epochs_array(mne, data, ch_names, 256.0, 0.0, labels, ch_types=ch_types)
    loaded = load_epochs_fif(_save_epochs(tmp_path, epochs, "meg-eeg-epo.fif"))
    assert loaded.n_channels == 2
    assert loaded.ch_names == ["C3", "C4"]
    np.testing.assert_allclose(loaded.data, data[:, :2, :])


def test_fif_no_eeg_raises(tmp_path: Path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, 2, 32))
    labels = np.array([1, 1, 2, 2])
    epochs = _epochs_array(
        mne, data, ["EOG", "STI 014"], 256.0, 0.0, labels, ch_types=["eog", "stim"]
    )
    with pytest.raises(AdapterError, match="no EEG"):
        load_epochs_fif(_save_epochs(tmp_path, epochs, "no-eeg-epo.fif"))


def test_fif_partial_montage(tmp_path: Path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(4)
    ch_names = ["Fz", "Cz", "Pz", "Oz", "X1"]
    n_trials, n_ch, n_times = 6, len(ch_names), 48
    data = rng.standard_normal((n_trials, n_ch, n_times))
    labels = np.array([0, 0, 1, 1, 0, 1])
    ch_pos = {
        "Fz": np.array([0.0, 0.08, 0.05]),
        "Cz": np.array([0.06, 0.0, 0.09]),
        "Pz": np.array([-0.05, -0.06, 0.05]),
        "Oz": np.array([0.02, -0.10, 0.02]),
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    epochs = _epochs_array(
        mne, data, ch_names, 256.0, 0.0, labels, montage=montage
    )
    loaded = load_epochs_fif(_save_epochs(tmp_path, epochs, "partial-epo.fif"))
    assert loaded.montage_xy is not None
    assert loaded.montage_xy.shape == (n_ch, 2)
    assert np.isnan(loaded.montage_xy[-1]).all()
    assert np.isfinite(loaded.montage_xy[:-1]).all()
    assert can_draw_scalp(loaded.montage_xy)
    Xi, Yi, Zi = interpolate_topo(loaded.montage_xy, np.arange(n_ch, dtype=float))
    assert Zi.shape == Xi.shape
    assert np.any(np.isfinite(Zi))


def test_fif_zero_positions(tmp_path: Path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(5)
    data = rng.standard_normal((4, 4, 32))
    labels = np.array([0, 0, 1, 1])
    epochs = _epochs_array(mne, data, ["C3", "C4", "Cz", "Fz"], 256.0, 0.0, labels)
    loaded = load_epochs_fif(_save_epochs(tmp_path, epochs, "nomont-epo.fif"))
    assert loaded.montage_xy is None
    assert not can_draw_scalp(loaded.montage_xy)


def test_fif_fewer_than_three_located(tmp_path: Path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(6)
    ch_names = ["C3", "C4", "X1", "X2"]
    data = rng.standard_normal((4, 4, 32))
    labels = np.array([0, 0, 1, 1])
    ch_pos = {
        "C3": np.array([-0.05, 0.0, 0.08]),
        "C4": np.array([0.05, 0.0, 0.08]),
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    epochs = _epochs_array(mne, data, ch_names, 256.0, 0.0, labels, montage=montage)
    loaded = load_epochs_fif(_save_epochs(tmp_path, epochs, "two-pos-epo.fif"))
    assert loaded.montage_xy is not None
    assert np.sum(np.isfinite(loaded.montage_xy).all(axis=1)) == 2
    assert not can_draw_scalp(loaded.montage_xy)
    with pytest.raises(ValueError, match="need at least three sensors"):
        interpolate_topo(loaded.montage_xy, np.zeros(4))


def test_fif_sessions_from_run_metadata(tmp_path: Path):
    mne = pytest.importorskip("mne")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    n_trials = 8
    data = rng.standard_normal((n_trials, 3, 32))
    labels = np.array([0, 1] * 4)
    metadata = pd.DataFrame({"run": np.repeat([4, 8], n_trials // 2)})
    epochs = _epochs_array(
        mne, data, ["C3", "C4", "Cz"], 256.0, 0.0, labels, metadata=metadata
    )
    loaded = load_epochs_fif(_save_epochs(tmp_path, epochs, "runs-epo.fif"))
    assert loaded.sessions is not None
    assert loaded.sessions.shape == (n_trials,)
    assert loaded.sessions.dtype.kind in "iu"
    assert int(np.unique(loaded.sessions).size) == 2
    assert loaded.sessions[:4].min() == loaded.sessions[:4].max()
    assert loaded.sessions[4:].min() == loaded.sessions[4:].max()
    assert loaded.sessions[0] != loaded.sessions[-1]


def test_fif_sessions_from_session_metadata(tmp_path: Path):
    mne = pytest.importorskip("mne")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(9)
    n_trials = 6
    data = rng.standard_normal((n_trials, 3, 32))
    labels = np.array([0, 1, 0, 1, 0, 1])
    metadata = pd.DataFrame({"Session": ["a", "a", "b", "b", "a", "b"]})
    epochs = _epochs_array(
        mne, data, ["C3", "C4", "Cz"], 256.0, 0.0, labels, metadata=metadata
    )
    loaded = load_epochs_fif(_save_epochs(tmp_path, epochs, "session-epo.fif"))
    assert loaded.sessions is not None
    assert loaded.sessions.shape == (n_trials,)
    assert loaded.sessions.dtype.kind in "iu"
    assert int(np.unique(loaded.sessions).size) == 2
    assert loaded.sessions[0] == loaded.sessions[1]
    assert loaded.sessions[0] != loaded.sessions[2]


def test_fif_sessions_none_without_metadata(tmp_path: Path):
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(8)
    data = rng.standard_normal((4, 3, 32))
    labels = np.array([0, 0, 1, 1])
    epochs = _epochs_array(mne, data, ["C3", "C4", "Cz"], 256.0, 0.0, labels)
    loaded = load_epochs_fif(_save_epochs(tmp_path, epochs, "nometadata-epo.fif"))
    assert loaded.sessions is None


def test_fif_session_column_beats_run():
    pd = pytest.importorskip("pandas")
    from epochlens.adapters.fif import _sessions_from_metadata

    class _Epochs:
        metadata = pd.DataFrame({"run": [1, 2, 3, 4], "session": [9, 9, 8, 8]})

    sessions = _sessions_from_metadata(_Epochs(), 4)
    assert sessions is not None
    assert int(np.unique(sessions).size) == 2
    assert sessions[0] == sessions[1]
    assert sessions[0] != sessions[2]


def test_fif_sessions_none_on_metadata_length_mismatch():
    pd = pytest.importorskip("pandas")
    from epochlens.adapters.fif import _sessions_from_metadata

    class _Epochs:
        metadata = pd.DataFrame({"run": [1, 2]})

    assert _sessions_from_metadata(_Epochs(), 4) is None
