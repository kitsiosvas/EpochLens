"""Batched complex-Morlet CWT. Loops only over frequencies, not trials/sensors."""

from __future__ import annotations

import numpy as np

from epochlens.cache import array_fingerprint, load_npz, save_npz
from epochlens.types import EpochBatch


def frequency_axis(
    fmin: float = 4.0,
    fmax: float = 40.0,
    voices_per_octave: int = 30,
) -> np.ndarray:
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("need 0 < fmin < fmax")
    if voices_per_octave < 1:
        raise ValueError("voices_per_octave must be >= 1")
    n_octaves = np.log2(fmax / fmin)
    n_freqs = int(np.round(n_octaves * voices_per_octave)) + 1
    freqs = fmin * (2.0 ** (np.arange(n_freqs) / voices_per_octave))
    return freqs[freqs <= fmax + 1e-9]


def _morlet_rfft(
    n_times: int,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray,
) -> np.ndarray:
    """Frequency-domain Morlet kernels, shape (n_freqs, n_rfft)."""
    freqs = np.asarray(freqs, dtype=np.float64)
    if np.isscalar(n_cycles):
        cycles = np.full(freqs.shape, float(n_cycles), dtype=np.float64)
    else:
        cycles = np.asarray(n_cycles, dtype=np.float64)
        if cycles.shape != freqs.shape:
            raise ValueError("n_cycles must be scalar or match freqs")
    fft_freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    sigma_t = cycles / (2.0 * np.pi * freqs)
    sigma_f = 1.0 / (2.0 * np.pi * sigma_t)
    gain = np.sqrt(2.0 * sigma_t)[:, None]
    gauss = np.exp(-0.5 * ((fft_freqs[None, :] - freqs[:, None]) / sigma_f[:, None]) ** 2)
    gauss[:, 0] = 0.0
    return gain * gauss


def _cwt_power_array(
    data: np.ndarray,
    sfreq: float,
    tmin: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray,
    decim: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def cwt_power(
    batch: EpochBatch,
    fmin: float = 4.0,
    fmax: float = 40.0,
    voices_per_octave: int = 30,
    n_cycles: float = 7.0,
    decim: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(power, freqs, times)``.

    ``power`` has shape ``(n_trials, n_channels, n_freqs, n_times_decim)``.
    """
    if decim < 1:
        raise ValueError("decim must be >= 1")
    freqs = frequency_axis(fmin, fmax, voices_per_octave)
    nyquist = 0.5 * batch.sfreq
    freqs = freqs[freqs < nyquist]
    if freqs.size == 0:
        raise ValueError("no frequencies below Nyquist")
    power, times = _cwt_power_array(batch.data, batch.sfreq, batch.tmin, freqs, n_cycles, decim)
    return power, freqs, times


def mean_cwt_power(
    batch: EpochBatch,
    fmin: float = 4.0,
    fmax: float = 40.0,
    voices_per_octave: int = 30,
    n_cycles: float = 7.0,
    decim: int = 1,
    trial_chunk: int = 8,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trial-averaged CWT power ``(n_channels, n_freqs, n_times)``, optionally cached."""
    freqs = frequency_axis(fmin, fmax, voices_per_octave)
    freqs = freqs[freqs < 0.5 * batch.sfreq]
    if freqs.size == 0:
        raise ValueError("no frequencies below Nyquist")
    payload = {
        "kind": "mean_cwt",
        "power": "abs2",
        "data": array_fingerprint(batch.data),
        "sfreq": batch.sfreq,
        "tmin": batch.tmin,
        "fmin": fmin,
        "fmax": fmax,
        "voices": voices_per_octave,
        "n_cycles": n_cycles,
        "decim": decim,
    }
    if use_cache:
        hit = load_npz("mean_cwt", payload)
        if hit is not None:
            return hit["power"], hit["freqs"], hit["times"]

    acc = None
    times = None
    n_trials = batch.n_trials
    for start in range(0, n_trials, trial_chunk):
        sl = slice(start, min(n_trials, start + trial_chunk))
        power, times = _cwt_power_array(
            batch.data[sl], batch.sfreq, batch.tmin, freqs, n_cycles, decim
        )
        chunk_sum = power.sum(axis=0)
        acc = chunk_sum if acc is None else acc + chunk_sum
    assert acc is not None and times is not None
    mean_power = acc / n_trials
    if use_cache:
        save_npz("mean_cwt", payload, {"power": mean_power, "freqs": freqs, "times": times})
    return mean_power, freqs, times


def mean_scalogram(
    power: np.ndarray,
    labels: np.ndarray | None = None,
) -> dict[int | None, np.ndarray]:
    """Average power over trials. Keys are class ids, or ``None`` for all trials."""
    if labels is None:
        return {None: power.mean(axis=0)}
    out: dict[int | None, np.ndarray] = {None: power.mean(axis=0)}
    for cls in np.unique(labels):
        out[int(cls)] = power[labels == cls].mean(axis=0)
    return out


def relative_scalogram(
    scalogram: np.ndarray,
    times: np.ndarray,
    baseline: tuple[float, float],
    eps: float = 1e-20,
) -> np.ndarray:
    """``(S - baseline) / baseline`` over time. ``scalogram`` is (n_ch, n_freq, n_time)."""
    tmin, tmax = baseline
    mask = (times >= tmin) & (times <= tmax)
    if not np.any(mask):
        raise ValueError("baseline window empty")
    base = scalogram[..., mask].mean(axis=-1, keepdims=True)
    return (scalogram - base) / np.maximum(base, eps)


def energy_channel_score(
    relative: np.ndarray,
    times: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """MATLAB memo3-style: mean over freq of time-variance in ``window``.

    ``relative`` is ``(n_channels, n_freqs, n_times)``.
    """
    mask = (times >= window[0]) & (times <= window[1])
    if not np.any(mask):
        raise ValueError("score window empty")
    band = relative[..., mask]
    return band.var(axis=-1).mean(axis=1)
