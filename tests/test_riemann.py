import numpy as np
import pytest

from epochlens.adapters.synthetic import make_synthetic
from epochlens.riemann import embed_mds, pairwise_distances, session_whiten, trial_covariances


def test_covariances_spd():
    batch = make_synthetic(n_channels=6, trials_per_class=5, duration=1.5, seed=4)
    covs = trial_covariances(batch, (0.5, 1.4), ridge=1e-3)
    assert covs.shape == (batch.n_trials, 6, 6)
    eigs = np.linalg.eigvalsh(covs[0])
    assert np.all(eigs > 0)


def test_logeuclid_zero_self_distance_and_mds():
    batch = make_synthetic(n_channels=5, trials_per_class=4, seed=5)
    covs = trial_covariances(batch, (0.5, 1.4))
    dist = pairwise_distances(covs, metric="logeuclid")
    assert dist.shape == (batch.n_trials, batch.n_trials)
    assert np.allclose(np.diag(dist), 0.0, atol=1e-6)
    xy = embed_mds(dist)
    assert xy.shape == (batch.n_trials, 2)


def test_session_whiten_runs():
    batch = make_synthetic(n_channels=5, trials_per_class=6, seed=6)
    covs = trial_covariances(batch, (0.5, 1.4))
    out = session_whiten(covs, batch.sessions, metric="logeuclid")
    assert out.shape == covs.shape


def test_affine_invariant_distance_and_whiten():
    batch = make_synthetic(n_channels=4, trials_per_class=4, duration=1.0, seed=8)
    covs = trial_covariances(batch, (0.5, 0.9))
    dist = pairwise_distances(covs, metric="riemann")
    assert dist.shape == (batch.n_trials, batch.n_trials)
    assert np.allclose(np.diag(dist), 0.0, atol=1e-6)
    assert np.all(dist >= -1e-9)
    xy = embed_mds(dist)
    assert xy.shape == (batch.n_trials, 2)
    out = session_whiten(covs, batch.sessions, metric="riemann")
    assert out.shape == covs.shape


def test_distances_match_pyriemann():
    pytest.importorskip("pyriemann")
    try:
        from pyriemann.geometry.distance import pairwise_distance
    except ImportError:
        from pyriemann.utils.distance import pairwise_distance

    batch = make_synthetic(n_channels=4, trials_per_class=3, duration=1.0, seed=9)
    covs = trial_covariances(batch, (0.5, 0.9))
    for metric in ("logeuclid", "riemann"):
        ours = pairwise_distances(covs, metric=metric)
        theirs = np.asarray(pairwise_distance(covs, metric=metric), dtype=np.float64)
        assert np.allclose(ours, theirs, atol=1e-8), metric
