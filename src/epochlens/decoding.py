"""Secondary linear check against chance. Not a BCI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from epochlens.riemann import trial_covariances
from epochlens.types import EpochBatch


@dataclass(frozen=True)
class ChanceReport:
    accuracy: float
    chance: float
    majority: float
    n_classes: int
    n_splits: int
    fold_scores: np.ndarray
    n_trials: int
    n_features: int
    method: str


def chance_level(n_classes: int) -> float:
    if n_classes < 2:
        raise ValueError("need at least two classes")
    return 1.0 / float(n_classes)


def _vectorize_logm(covs: np.ndarray) -> np.ndarray:
    from epochlens.riemann import _logm

    logs = np.stack([_logm(c) for c in covs], axis=0)
    iu = np.triu_indices(covs.shape[1])
    return logs[:, iu[0], iu[1]]


def logeuclid_lda_cv(
    batch: EpochBatch,
    window: tuple[float, float],
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> ChanceReport:
    """Stratified CV LDA on vectorized log-Euclidean covariances.

    Accuracy is a sanity check against chance, not a decoder claim.
    """
    if batch.labels is None:
        raise ValueError("labels required")
    y = np.asarray(batch.labels)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = int(classes.size)
    if n_classes < 2:
        raise ValueError("need at least two classes")
    n_splits = min(int(n_splits), int(counts.min()))
    if n_splits < 2:
        raise ValueError("need at least two trials per class for CV")

    covs = trial_covariances(batch, window)
    x = _vectorize_logm(covs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train, test in skf.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train])
        x_test = scaler.transform(x[test])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(x_train, y[train])
        scores.append(float(clf.score(x_test, y[test])))
    fold = np.asarray(scores, dtype=np.float64)
    return ChanceReport(
        accuracy=float(fold.mean()),
        chance=chance_level(n_classes),
        majority=float(counts.max() / counts.sum()),
        n_classes=n_classes,
        n_splits=n_splits,
        fold_scores=fold,
        n_trials=int(y.size),
        n_features=int(x.shape[1]),
        method="logeuclid-LDA",
    )
