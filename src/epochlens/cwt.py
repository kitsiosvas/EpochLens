"""Batched complex-Morlet CWT. Vectorizes frequencies in memory-safe chunks."""

from __future__ import annotations

import numpy as np

from epochlens.cache import array_fingerprint, load_npz, save_npz
from epochlens.types import EpochBatch

# Peak irfft workspace ~ trials × channels × freq_chunk × n_times float64 elements.
_CWT_FREQ_CHUNK_ELEMS = 8_000_000
_DEFAULT_TRIAL_CHUNK = 8


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


def _freq_chunk_size(
    n_trials: int,
    n_channels: int,
    n_times: int,
    n_freqs: int,
    freq_chunk: int | None,
) -> int:
    if n_freqs <= 0:
        return 1
    if freq_chunk is not None:
        return max(1, min(int(freq_chunk), n_freqs))
    denom = max(1, int(n_trials) * int(n_channels) * int(n_times))
    return max(1, min(n_freqs, _CWT_FREQ_CHUNK_ELEMS // denom))


def _cwt_times(tmin: float, n_times: int, sfreq: float, decim: int, n_out: int) -> np.ndarray:
    times = tmin + np.arange(n_times, dtype=np.float64)[::decim] / sfreq
    if times.size > n_out:
        return times[:n_out]
    if times.size < n_out:
        return np.pad(times, (0, n_out - times.size), mode="edge")
    return times


def _freqs_below_nyquist(
    sfreq: float,
    fmin: float,
    fmax: float,
    voices_per_octave: int,
) -> np.ndarray:
    freqs = frequency_axis(fmin, fmax, voices_per_octave)
    freqs = freqs[freqs < 0.5 * sfreq]
    if freqs.size == 0:
        raise ValueError("no frequencies below Nyquist")
    return freqs


def _cwt_power_chunks(
    data: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray,
    decim: int,
    *,
    trial_chunk: int | None = None,
    freq_chunk: int | None = None,
):
    """Yield ``(trial_slice, i0, i1, power_chunk)`` with freq-batched irfft.

    ``power_chunk`` is ``(n_slice, n_channels, i1 - i0, n_times_decim)`` and
    equals ``|coeff|^2`` after CWT, then ``decim`` along time.
    """
    n_trials, n_channels, n_times = data.shape
    kernels = _morlet_rfft(n_times, sfreq, freqs, n_cycles)
    n_freqs = int(freqs.size)
    step = n_trials if trial_chunk is None else int(trial_chunk)
    step = max(1, step)
    for start in range(0, n_trials, step):
        stop = min(n_trials, start + step)
        sl = slice(start, stop)
        spectra = np.fft.rfft(data[sl], n=n_times, axis=-1)
        fstep = _freq_chunk_size(stop - start, n_channels, n_times, n_freqs, freq_chunk)
        for i0 in range(0, n_freqs, fstep):
            i1 = min(n_freqs, i0 + fstep)
            # (trials, channels, freq_chunk, n_rfft) → irfft along time
            coeff = np.fft.irfft(spectra[..., None, :] * kernels[i0:i1], n=n_times, axis=-1)
            yield sl, i0, i1, np.abs(coeff[..., ::decim]) ** 2


def _cwt_power_array(
    data: np.ndarray,
    sfreq: float,
    tmin: float,
    freqs: np.ndarray,
    n_cycles: float | np.ndarray,
    decim: int,
    freq_chunk: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_times = data.shape[-1]
    n_out = int(np.ceil(n_times / decim))
    power = np.empty((data.shape[0], data.shape[1], freqs.size, n_out), dtype=np.float64)
    for sl, i0, i1, chunk in _cwt_power_chunks(
        data, sfreq, freqs, n_cycles, decim, freq_chunk=freq_chunk
    ):
        power[sl, :, i0:i1, :] = chunk
    times = _cwt_times(tmin, n_times, sfreq, decim, n_out)
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
    freqs = _freqs_below_nyquist(batch.sfreq, fmin, fmax, voices_per_octave)
    power, times = _cwt_power_array(batch.data, batch.sfreq, batch.tmin, freqs, n_cycles, decim)
    return power, freqs, times


def mean_cwt_power(
    batch: EpochBatch,
    fmin: float = 4.0,
    fmax: float = 40.0,
    voices_per_octave: int = 30,
    n_cycles: float = 7.0,
    decim: int = 1,
    trial_chunk: int = _DEFAULT_TRIAL_CHUNK,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trial-averaged CWT power ``(n_channels, n_freqs, n_times)``, optionally cached."""
    freqs = _freqs_below_nyquist(batch.sfreq, fmin, fmax, voices_per_octave)
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

    n_times = batch.n_times
    n_out = int(np.ceil(n_times / decim))
    acc = np.zeros((batch.n_channels, freqs.size, n_out), dtype=np.float64)
    for _sl, i0, i1, chunk in _cwt_power_chunks(
        batch.data, batch.sfreq, freqs, n_cycles, decim, trial_chunk=trial_chunk
    ):
        acc[:, i0:i1, :] += chunk.sum(axis=0)
    mean_power = acc / batch.n_trials
    times = _cwt_times(batch.tmin, n_times, batch.sfreq, decim, n_out)
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


def _class_rel_payload(
    picked: EpochBatch,
    baseline: tuple[float, float],
    *,
    show_n: int,
    fmin: float,
    fmax: float,
    voices_per_octave: int,
    n_cycles: float,
    decim: int,
) -> dict:
    assert picked.labels is not None
    return {
        "kind": "class_relative_cwt",
        "power": "abs2",
        "data": array_fingerprint(picked.data),
        "labels": array_fingerprint(picked.labels),
        "sfreq": picked.sfreq,
        "tmin": picked.tmin,
        "fmin": fmin,
        "fmax": fmax,
        "voices": voices_per_octave,
        "n_cycles": n_cycles,
        "decim": decim,
        "baseline": [float(baseline[0]), float(baseline[1])],
        "show_n": int(show_n),
    }


def _class_rel_from_cache(
    hit: dict[str, np.ndarray],
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    names = [str(x) for x in hit["names"]]
    rel_by_class = {
        int(cid): hit["rel"][i] for i, cid in enumerate(np.asarray(hit["class_ids"]))
    }
    return rel_by_class, hit["times"], hit["freqs"], names


def class_relative_scalograms(
    batch: EpochBatch,
    baseline: tuple[float, float],
    *,
    show_n: int = 4,
    fmin: float = 4.0,
    fmax: float = 40.0,
    voices_per_octave: int = 30,
    n_cycles: float = 7.0,
    decim: int = 1,
    use_cache: bool = True,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Per-class baseline-relative CWT on the first ``show_n`` channels."""
    if batch.labels is None:
        raise ValueError("class relative CWT needs labels")
    n_show = min(int(show_n), batch.n_channels)
    picked = batch.pick(np.arange(n_show))
    names = picked.ch_names
    freqs = _freqs_below_nyquist(picked.sfreq, fmin, fmax, voices_per_octave)
    payload = _class_rel_payload(
        picked,
        baseline,
        show_n=n_show,
        fmin=fmin,
        fmax=fmax,
        voices_per_octave=voices_per_octave,
        n_cycles=n_cycles,
        decim=decim,
    )
    if use_cache:
        hit = load_npz("class_rel_cwt", payload)
        if hit is not None:
            return _class_rel_from_cache(hit)

    labels = np.asarray(picked.labels)
    class_ids = np.unique(labels)
    n_times = picked.n_times
    n_out = int(np.ceil(n_times / decim))
    acc = {
        int(cid): np.zeros((n_show, freqs.size, n_out), dtype=np.float64) for cid in class_ids
    }
    counts = {int(cid): int(np.count_nonzero(labels == cid)) for cid in class_ids}
    for sl, i0, i1, chunk in _cwt_power_chunks(
        picked.data,
        picked.sfreq,
        freqs,
        n_cycles,
        decim,
        trial_chunk=_DEFAULT_TRIAL_CHUNK,
    ):
        labels_sl = labels[sl]
        for cid in class_ids:
            mask = labels_sl == cid
            if not np.any(mask):
                continue
            acc[int(cid)][:, i0:i1, :] += chunk[mask].sum(axis=0)
    times = _cwt_times(picked.tmin, n_times, picked.sfreq, decim, n_out)
    rel_by_class: dict[int, np.ndarray] = {}
    for cid in class_ids:
        key = int(cid)
        n_cls = counts[key]
        if n_cls == 0:
            continue
        rel_by_class[key] = relative_scalogram(acc[key] / n_cls, times, baseline)

    if use_cache:
        class_id_arr = np.fromiter(rel_by_class.keys(), dtype=np.int64, count=len(rel_by_class))
        save_npz(
            "class_rel_cwt",
            payload,
            {
                "rel": np.stack([rel_by_class[int(c)] for c in class_id_arr]),
                "class_ids": class_id_arr,
                "freqs": freqs,
                "times": times,
                "names": np.asarray(names),
            },
        )
    return rel_by_class, times, freqs, names


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
