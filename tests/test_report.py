from pathlib import Path

import pytest

from epochlens.adapters.synthetic import make_synthetic


def test_html_report_contains_honesty_and_figures(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from epochlens.explorer.report import HONESTY, render_report, write_report

    batch = make_synthetic(n_channels=8, trials_per_class=6, seed=7)
    html = render_report(
        batch,
        window=(0.6, 1.4),
        baseline=(0.0, 0.4),
        voices_per_octave=6,
        decim=4,
        top_k=4,
        sidecar=tmp_path / "report.json",
    )
    low = html.lower()
    assert HONESTY[:40] in html
    assert "not a classifier" in low
    assert "not a bci" in low
    assert "class-mean waveforms" in low
    assert "ranked channels" in low
    assert "<table" in low
    assert "class-mean spectra" in low
    assert "chance" in low
    assert "band-power topography" in low
    assert "session-whitened" in low
    assert "per-class relative cwt" not in low
    assert "all channels" in low or "full montage" in low
    assert "ranked-channel selection is not used" in low
    assert "full-montage" in low
    assert " vs " in low
    assert "hard fork" not in low
    assert "inner speech" not in low
    assert "pronounced" not in low
    assert "action window" not in low
    assert "analysis window" in low
    assert "24 trials" in low
    assert "8 channels" in low
    assert "mathjax" in low
    assert "\\mathrm{SEM}" in html
    assert "\\[" in html
    assert "data:image/png;base64," in html
    assert html.count("<img ") >= 5
    path = write_report(html, tmp_path / "report.html")
    assert path.stat().st_size > 10_000
    sidecar = tmp_path / "report.json"
    assert sidecar.exists()
    text = sidecar.read_text(encoding="utf-8")
    assert "ranking" in text
    assert "chance" in text
    assert "10 Hz" in text
    assert "excluded bad" not in low


def test_full_report_adds_per_class_cwt_and_extra_topomap(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from epochlens.explorer.report import render_report

    batch = make_synthetic(n_channels=8, trials_per_class=6, seed=7)
    html = render_report(
        batch,
        window=(0.6, 1.4),
        baseline=(0.0, 0.4),
        voices_per_octave=6,
        decim=4,
        top_k=4,
        full=True,
        sidecar=tmp_path / "full.json",
    )
    low = html.lower()
    assert "per-class relative cwt" in low
    assert "discriminability topography" in low
    assert "not a classifier" in low
    assert "not a bci" in low
    assert tmp_path.joinpath("full.json").exists()
