from epochlens.windows import default_windows


def test_generic_epochs_use_span_split():
    window, baseline = default_windows("mne-epochs", 0.0, 10.0)
    assert baseline == (0.0, 2.0)
    assert window == (2.0, 10.0)


def test_synthetic_demo_windows_cover_planted_rhythm():
    window, baseline = default_windows("synthetic", 0.0, 2.0)
    assert window == (0.6, 1.4)
    assert baseline[0] == 0.0
