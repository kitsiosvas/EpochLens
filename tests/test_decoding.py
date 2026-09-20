from eegvis.adapters.synthetic import make_synthetic
from eegvis.decoding import chance_level, logeuclid_lda_cv


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
