import numpy as np
import pytest

from epochlens.adapters.synthetic import make_synthetic
from epochlens.cwt import (
    _cwt_power_array,
    _morlet_rfft,
    class_relative_scalograms,
    cwt_power,
    energy_channel_score,
    frequency_axis,
    mean_cwt_power,
    mean_scalogram,
    relative_scalogram,
)


def _cwt_power_array_ref(data, sfreq, tmin, freqs, n_cycles, decim):
    """Original per-frequency irfft loop (slow reference)."""
    n_times = data.shape[-1]
    kernels = _morlet_rfft(n_times, sfreq, freqs, n_cycles)
    spectra = np.fft.rfft(data, n=n_times, axis=-1)
    n_out = int(np.ceil(n_times / decim))
    power = np.empty((data.shape[0], data.shape[1], freqs.size, n_out), dtype=np.float64)
    for i in range(freqs.size):
        coeff = np.fft.irfft(spectra * kernels[i], n=n_times, axis=-1)
        power[:, :, i, :] = np.abs(coeff[..., ::decim]) ** 2
    times = tmin + np.arange(n_times, dtype=np.float64)[::decim] / sfreq
    if times.size > n_out:
        times = times[:n_out]
    elif times.size < n_out:
        times = np.pad(times, (0, n_out - times.size), mode="edge")
    return power, times


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


def test_cwt_power_array_matches_slow_reference():
    batch = make_synthetic(
        n_channels=3, n_classes=2, trials_per_class=2, duration=0.5, seed=3
    )
    freqs = frequency_axis(8.0, 16.0, 4)
    freqs = freqs[freqs < 0.5 * batch.sfreq]
    data = batch.data[:3, :2]
    ref, tref = _cwt_power_array_ref(data, batch.sfreq, batch.tmin, freqs, 7.0, 2)
    for fchunk in (1, 2, None):
        got, tgot = _cwt_power_array(
            data, batch.sfreq, batch.tmin, freqs, 7.0, 2, freq_chunk=fchunk
        )
        np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(tgot, tref)
    assert ref.min() >= 0.0


def test_class_relative_matches_per_class_mean_cwt():
    batch = make_synthetic(n_channels=4, trials_per_class=4, seed=8)
    baseline = (0.0, 0.4)
    kwargs = dict(
        show_n=2,
        fmin=6.0,
        fmax=20.0,
        voices_per_octave=4,
        n_cycles=7.0,
        decim=4,
        use_cache=False,
    )
    rel_by_class, times, freqs, names = class_relative_scalograms(batch, baseline, **kwargs)
    n_show = 2
    for cls in np.unique(batch.labels):
        sub = batch.subset_trials(batch.labels == cls).pick(np.arange(n_show))
        mean_power, f_ref, t_ref = mean_cwt_power(
            sub,
            fmin=kwargs["fmin"],
            fmax=kwargs["fmax"],
            voices_per_octave=kwargs["voices_per_octave"],
            n_cycles=kwargs["n_cycles"],
            decim=kwargs["decim"],
            trial_chunk=2,
            use_cache=False,
        )
        expected = relative_scalogram(mean_power, t_ref, baseline)
        np.testing.assert_allclose(rel_by_class[int(cls)], expected, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(freqs, f_ref)
        np.testing.assert_allclose(times, t_ref)
    assert names == batch.ch_names[:n_show]
