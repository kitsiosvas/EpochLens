import numpy as np

from eegvis.adapters.synthetic import make_synthetic
from eegvis.bands import band_power, window_spectrum


def test_planted_10hz_shows_in_alpha():
    batch = make_synthetic(n_channels=8, trials_per_class=10, seed=18)
    power, names = band_power(batch, (0.6, 1.4))
    assert power.shape[:2] == (batch.n_trials, batch.n_channels)
    assert "alpha" in names
    alpha = names.index("alpha")
    # Class 0 is planted at 10 Hz on channel 0.
    cls0 = power[batch.labels == 0, 0, alpha].mean()
    cls0_other = power[batch.labels == 0, 5, alpha].mean()
    assert cls0 > cls0_other


def test_window_spectrum_peaks_near_planted_hz():
    batch = make_synthetic(n_channels=8, trials_per_class=10, seed=18)
    spec, freqs = window_spectrum(batch, (0.6, 1.4))
    f_idx = int(np.argmin(np.abs(freqs - 10.0)))
    cls0 = spec[batch.labels == 0, 0, f_idx].mean()
    other = spec[batch.labels == 0, 5, f_idx].mean()
    assert cls0 > other
