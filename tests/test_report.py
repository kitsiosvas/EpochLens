from pathlib import Path

import pytest

from eegvis.adapters.synthetic import make_synthetic


def test_html_report_contains_honesty_and_figures(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from eegvis.explorer.report import HONESTY, render_report, write_report

    batch = make_synthetic(n_channels=8, trials_per_class=6, seed=7)
    html = render_report(batch, window=(0.6, 1.4), baseline=(0.0, 0.4), voices_per_octave=6, decim=4)
    assert HONESTY[:40] in html
    assert "not a classifier" in html.lower()
    assert "hard fork" not in html.lower()
    assert "data:image/png;base64," in html
    assert html.count("<img ") >= 4
    path = write_report(html, tmp_path / "report.html")
    assert path.stat().st_size > 10_000
