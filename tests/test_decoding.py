from epochlens.adapters.synthetic import make_synthetic
from epochlens.decoding import chance_level, logeuclid_lda_cv, logeuclid_mdm_cv
from epochlens.riemann import trial_covariances


def test_chance_level_four_class():
    assert chance_level(4) == 0.25


def test_logeuclid_lda_cv_reports_against_chance():
    batch = make_synthetic(n_channels=8, trials_per_class=12, seed=14)
    report = logeuclid_lda_cv(batch, (0.6, 1.4), n_splits=4, random_state=0)
    assert report.n_classes == 4
    assert report.chance == 0.25
    assert report.n_splits == 4
    assert report.fold_scores.size == 4
    assert 0.0 <= report.accuracy <= 1.0
    assert report.n_features == 8 * 9 // 2
    assert report.method == "logeuclid-LDA"


def test_logeuclid_mdm_cv_reports_against_chance():
    batch = make_synthetic(n_channels=8, trials_per_class=12, seed=14)
    covs = trial_covariances(batch, (0.6, 1.4))
    lda = logeuclid_lda_cv(batch, (0.6, 1.4), n_splits=4, random_state=0, covs=covs)
    report = logeuclid_mdm_cv(batch, (0.6, 1.4), n_splits=4, random_state=0, covs=covs)
    assert report.n_classes == 4
    assert report.chance == lda.chance == 0.25
    assert report.majority == lda.majority == 0.25
    assert report.n_splits == 4
    assert report.fold_scores.size == 4
    assert 0.0 <= report.accuracy <= 1.0
    assert report.n_trials == batch.n_trials
    assert report.method == "logeuclid-MDM"
    assert report.n_features == 0

