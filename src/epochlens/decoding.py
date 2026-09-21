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


def _cv_covs_and_labels(
    batch: EpochBatch,
    window: tuple[float, float],
    n_splits: int,
    random_state: int,
    covs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, StratifiedKFold, int, np.ndarray, int]:
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
    if covs is None:
        covs = trial_covariances(batch, window)
    else:
        covs = np.asarray(covs, dtype=np.float64)
        if covs.shape[0] != y.size:
            raise ValueError("covs must have one matrix per trial")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return covs, y, skf, n_classes, counts, n_splits


def _logeuclid_lda_fold(
    covs_train: np.ndarray,
    covs_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, int]:
    from pyriemann.tangentspace import TangentSpace

    ts = TangentSpace(metric="logeuclid")
    scaler = StandardScaler()
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    x_train = scaler.fit_transform(ts.fit_transform(covs_train))
    x_test = scaler.transform(ts.transform(covs_test))
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test)), int(x_train.shape[1])


def _logeuclid_mdm_fold(
    covs_train: np.ndarray,
    covs_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> float:
    from pyriemann.classification import MDM

    clf = MDM(metric="logeuclid")
    clf.fit(covs_train, y_train)
    return float(clf.score(covs_test, y_test))


def logeuclid_lda_cv(
    batch: EpochBatch,
    window: tuple[float, float],
    *,
    n_splits: int = 5,
    random_state: int = 0,
    covs: np.ndarray | None = None,
) -> ChanceReport:
    """Stratified CV LDA on pyRiemann log-Euclidean tangent space.

    Accuracy is a sanity check against chance, not a decoder claim.
    Tangent-space maps are fit on the training fold only.
    Pass ``covs`` to reuse trial covariances already estimated for another check.
    """
    covs, y, skf, n_classes, counts, n_splits = _cv_covs_and_labels(
        batch, window, n_splits, random_state, covs
    )
    scores = []
    n_features = 0
    for train, test in skf.split(covs, y):
        acc, n_features = _logeuclid_lda_fold(covs[train], covs[test], y[train], y[test])
        scores.append(acc)
    fold = np.asarray(scores, dtype=np.float64)
    return ChanceReport(
        accuracy=float(fold.mean()),
        chance=chance_level(n_classes),
        majority=float(counts.max() / counts.sum()),
        n_classes=n_classes,
        n_splits=n_splits,
        fold_scores=fold,
        n_trials=int(y.size),
        n_features=n_features,
        method="logeuclid-LDA",
    )


def logeuclid_mdm_cv(
    batch: EpochBatch,
    window: tuple[float, float],
    *,
    n_splits: int = 5,
    random_state: int = 0,
    covs: np.ndarray | None = None,
) -> ChanceReport:
    """Stratified CV pyRiemann MDM (log-Euclidean). Not a BCI.

    Fit on training-fold covariances only. Pass the same ``covs`` used for
    :func:`logeuclid_lda_cv` so both checks share one geometry.
    """
    covs, y, skf, n_classes, counts, n_splits = _cv_covs_and_labels(
        batch, window, n_splits, random_state, covs
    )
    scores = [_logeuclid_mdm_fold(covs[train], covs[test], y[train], y[test]) for train, test in skf.split(covs, y)]
    fold = np.asarray(scores, dtype=np.float64)
    return ChanceReport(
        accuracy=float(fold.mean()),
        chance=chance_level(n_classes),
        majority=float(counts.max() / counts.sum()),
        n_classes=n_classes,
        n_splits=n_splits,
        fold_scores=fold,
        n_trials=int(y.size),
        n_features=0,
        method="logeuclid-MDM",
    )
