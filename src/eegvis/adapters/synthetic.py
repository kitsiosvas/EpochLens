"""Synthetic 4-class EEG with a planted rhythm so plots are non-empty without a dataset."""

from __future__ import annotations

import numpy as np

from eegvis.types import EpochBatch


def make_synthetic(
    *,
    n_classes: int = 4,
    trials_per_class: int = 24,
    n_channels: int = 16,
    sfreq: float = 256.0,
    duration: float = 2.0,
    tmin: float = 0.0,
    seed: int = 0,
) -> EpochBatch:
    rng = np.random.default_rng(seed)
    n_times = int(round(duration * sfreq))
    n_trials = n_classes * trials_per_class
    times = tmin + np.arange(n_times) / sfreq
    data = 0.4 * rng.standard_normal((n_trials, n_channels, n_times))
    labels = np.repeat(np.arange(n_classes), trials_per_class)
    sessions = np.tile(np.array([1, 2, 3]), int(np.ceil(n_trials / 3)))[:n_trials]
    freqs = 10.0 + 2.0 * np.arange(n_classes)

    burst = (times >= 0.6) & (times <= 1.4)
    for cls in range(n_classes):
        wave = np.sin(2 * np.pi * freqs[cls] * times)
        ch = cls % n_channels
        idx = labels == cls
        data[idx, ch] += 2.5 * wave * burst
        if n_channels > 1:
            data[idx, (ch + 1) % n_channels] += 1.2 * wave * burst

    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    names = [f"Ch{i + 1:02d}" for i in range(n_channels)]
    return EpochBatch(
        data=data,
        sfreq=sfreq,
        ch_names=names,
        tmin=tmin,
        labels=labels,
        sessions=sessions,
        montage_xy=xy,
        class_names={i: f"{freqs[i]:.0f} Hz" for i in range(n_classes)},
        subject_id="synthetic",
        condition="demo",
        dataset="synthetic",
    )
