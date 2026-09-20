from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpochBatch:
    """Generic epoched EEG. Algorithms take this, not a named dataset.

    Parameters
    ----------
    data
        Array of shape ``(n_trials, n_channels, n_times)``.
    sfreq
        Sampling rate in Hz.
    ch_names
        Channel labels, length ``n_channels``.
    tmin
        Time in seconds of sample 0 (relative to whatever event the
        caller chose). Windows passed to algorithms are in this clock.
    labels
        Class id per trial, shape ``(n_trials,)``, or None.
    sessions
        Session id per trial, shape ``(n_trials,)``, or None.
    montage_xy
        Optional 2D scalp coordinates, shape ``(n_channels, 2)``.
    class_names
        Map from class id to display name.
    subject_id, condition, dataset
        Optional provenance for the explorer. Unused by core math.
    """

    data: np.ndarray
    sfreq: float
    ch_names: list[str]
    tmin: float = 0.0
    labels: np.ndarray | None = None
    sessions: np.ndarray | None = None
    montage_xy: np.ndarray | None = None
    class_names: dict[int, str] = field(default_factory=dict)
    subject_id: str | None = None
    condition: str | None = None
    dataset: str | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 3:
            raise ValueError("data must be (n_trials, n_channels, n_times)")
        n_trials, n_channels, n_times = self.data.shape
        if len(self.ch_names) != n_channels:
            raise ValueError("ch_names length must match n_channels")
        if n_times < 2:
            raise ValueError("need at least two time samples")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        if self.labels is not None:
            self.labels = np.asarray(self.labels)
            if self.labels.shape != (n_trials,):
                raise ValueError("labels must be shape (n_trials,)")
        if self.sessions is not None:
            self.sessions = np.asarray(self.sessions)
            if self.sessions.shape != (n_trials,):
                raise ValueError("sessions must be shape (n_trials,)")
        if self.montage_xy is not None:
            self.montage_xy = np.asarray(self.montage_xy, dtype=np.float64)
            if self.montage_xy.shape != (n_channels, 2):
                raise ValueError("montage_xy must be (n_channels, 2)")

    @property
    def n_trials(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[2])

    @property
    def times(self) -> np.ndarray:
        return self.tmin + np.arange(self.n_times, dtype=np.float64) / self.sfreq

    def copy_with(self, **kwargs) -> EpochBatch:
        payload = {
            "data": self.data,
            "sfreq": self.sfreq,
            "ch_names": list(self.ch_names),
            "tmin": self.tmin,
            "labels": None if self.labels is None else self.labels.copy(),
            "sessions": None if self.sessions is None else self.sessions.copy(),
            "montage_xy": None if self.montage_xy is None else self.montage_xy.copy(),
            "class_names": dict(self.class_names),
            "subject_id": self.subject_id,
            "condition": self.condition,
            "dataset": self.dataset,
        }
        payload.update(kwargs)
        return EpochBatch(**payload)
