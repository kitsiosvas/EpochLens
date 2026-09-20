import numpy as np

from epochlens.adapters.synthetic import make_synthetic
import pytest

from epochlens.cwt import (
    class_relative_scalograms,
    cwt_power,
    energy_channel_score,
    mean_scalogram,
    relative_scalogram,
)


def test_cwt_shapes_and_planted_band():
    batch = make_synthetic(n_channels=8, trials_per_class=8, duration=2.0, seed=1)
    power, freqs, times = cwt_power(batch, fmin=6.0, fmax=24.0, voices_per_octave=8, decim=2)
    assert power.shape[0] == batch.n_trials
    assert power.shape[1] == batch.n_channels
    assert power.shape[2] == freqs.size
    assert power.shape[3] == times.size
    assert power.min() >= 0
    means = mean_scalogram(power, batch.labels)
    # Class 0 is planted at 10 Hz on channel 0.
    ch0 = means[0][0]
    f_idx = int(np.argmin(np.abs(freqs - 10.0)))
    t_idx = (times >= 0.6) & (times <= 1.4)
    assert ch0[f_idx, t_idx].mean() > ch0[:, ~t_idx].mean()


def test_cwt_stores_power_not_amplitude():
    batch = make_synthetic(n_channels=4, trials_per_class=4, seed=4)
    power, freqs, times = cwt_power(batch, fmin=8.0, fmax=12.0, voices_per_octave=4, decim=4)
    assert power.min() >= 0
    # A 10 Hz class-0 burst on Ch1 should dominate as energy (|coeff|^2), not merely |coeff|.
    f_idx = int(np.argmin(np.abs(freqs - 10.0)))
    t_idx = (times >= 0.6) & (times <= 1.4)
    planted = power[batch.labels == 0, 0, f_idx][:, t_idx].mean()
    quiet = power[batch.labels == 0, 0, f_idx][:, ~t_idx].mean()
    assert planted > quiet
    # Squaring stretches the contrast: peak/mean should exceed the amplitude ratio.
    amp = np.sqrt(np.maximum(power, 0.0))
    p_ratio = planted / max(float(quiet), 1e-12)
    a_ratio = (amp[batch.labels == 0, 0, f_idx][:, t_idx].mean()) / max(
        float(amp[batch.labels == 0, 0, f_idx][:, ~t_idx].mean()), 1e-12
    )
    assert p_ratio > a_ratio


def test_relative_and_energy_score():
    batch = make_synthetic(n_channels=8, trials_per_class=6, seed=2)
    power, freqs, times = cwt_power(batch, fmin=6.0, fmax=20.0, voices_per_octave=6, decim=4)
    rel = relative_scalogram(mean_scalogram(power)[None], times, (batch.tmin, 0.4))
    scores = energy_channel_score(rel, times, (0.6, 1.4))
    assert scores.shape == (batch.n_channels,)
    assert np.all(np.isfinite(scores))


def test_class_relative_scalograms_need_labels():
    batch = make_synthetic(n_channels=4, trials_per_class=3, seed=6).copy_with(labels=None)
    with pytest.raises(ValueError, match="labels"):
        class_relative_scalograms(
            batch, (0.0, 0.4), show_n=2, voices_per_octave=4, decim=4, use_cache=False
        )


def test_class_relative_scalograms_zero_in_baseline():
    batch = make_synthetic(n_channels=4, trials_per_class=4, seed=8)
    baseline = (0.0, 0.4)
    rel_by_class, times, _freqs, names = class_relative_scalograms(
        batch,
        baseline,
        show_n=2,
        fmin=6.0,
        fmax=20.0,
        voices_per_octave=4,
        decim=4,
        use_cache=False,
    )
    assert len(names) == 2
    mask = (times >= baseline[0]) & (times <= baseline[1])
    assert np.any(mask)
    for rel in rel_by_class.values():
        assert rel.shape[0] == 2
        assert rel.min() < 0
        np.testing.assert_allclose(rel[..., mask].mean(axis=-1), 0.0, atol=1e-10)
