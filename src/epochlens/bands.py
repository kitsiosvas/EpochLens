"""Band power from the FFT of a caller-supplied time window."""

from __future__ import annotations

import numpy as np

from epochlens.types import EpochBatch
from epochlens.windows import sample_span

BANDS: tuple[tuple[str, float, float], ...] = (
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)

BAND_GLYPHS: dict[str, str] = {
    "theta": "θ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
}


def window_spectrum(
    batch: EpochBatch,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """RFFT power in ``window``, shape ``(n_trials, n_channels, n_freqs)``."""
    sl = sample_span(batch.n_times, batch.sfreq, batch.tmin, window[0], window[1])
    x = batch.data[:, :, sl]
    n = x.shape[-1]
    if n < 4:
        raise ValueError("spectrum window too short")
    spec = np.abs(np.fft.rfft(x * np.hanning(n), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / batch.sfreq)
    return spec, freqs


def band_power(
    batch: EpochBatch,
    window: tuple[float, float],
    bands: tuple[tuple[str, float, float], ...] = BANDS,
) -> tuple[np.ndarray, list[str]]:
    """Trial-wise band power, shape ``(n_trials, n_channels, n_bands)``."""
    spec, freqs = window_spectrum(batch, window)
    nyquist = 0.5 * batch.sfreq
    names: list[str] = []
    cols: list[np.ndarray] = []
    for name, lo, hi in bands:
        if lo >= nyquist:
            continue
        hi = min(hi, nyquist - 1e-9)
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            continue
        cols.append(spec[..., m].mean(axis=-1))
        names.append(name)
    if not cols:
        raise ValueError("no bands below Nyquist")
    return np.stack(cols, axis=-1), names


def class_mean_band_power(
    power: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-mean of trial-wise ``band_power``.

    ``power`` is ``(n_trials, n_channels, n_bands)``. Returns
    ``(n_classes, n_channels, n_bands)`` and the class ids in ``np.unique`` order.
    """
    power = np.asarray(power, dtype=np.float64)
    labels = np.asarray(labels)
    if power.ndim != 3:
        raise ValueError("power must be (n_trials, n_channels, n_bands)")
    if labels.shape[0] != power.shape[0]:
        raise ValueError("labels length must match n_trials")
    classes = np.unique(labels)
    means = np.stack([power[labels == cls].mean(axis=0) for cls in classes], axis=0)
    return means, classes


def band_range_caption(
    bands: tuple[tuple[str, float, float], ...] = BANDS,
) -> str:
    """Compact θ/α/β/γ Hz listing for figure captions."""
    parts = []
    for name, lo, hi in bands:
        glyph = BAND_GLYPHS.get(name, name)
        parts.append(f"{glyph} {lo:g}–{hi:g} Hz")
    return " · ".join(parts)
