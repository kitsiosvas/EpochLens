from pathlib import Path

import pytest


def test_html_flag_writes_report(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from eegvis.explorer.cli import main

    out = tmp_path / "snap.html"
    main(["--html", "--out", str(out), "--top-k", "4"])
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "<img " in text
    assert "analysis window" in text.lower()
    assert (tmp_path / "snap.json").exists()


def test_html_required_when_fif_source_has_no_path():
    from eegvis.explorer.cli import main

    with pytest.raises(SystemExit, match="--fif path is required"):
        main(["--html", "--source", "fif"])
