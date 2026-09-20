import pytest

from epochlens.explorer.mathnotes import NOTES, html_block


def test_every_note_has_display_latex():
    required = {
        "waveforms",
        "spectra",
        "cwt",
        "cwt_energy",
        "scalp",
        "disc",
        "ranking",
        "mds",
        "chance",
    }
    assert required <= set(NOTES)
    for key, note in NOTES.items():
        assert note.equations, key
        for eq in note.equations:
            assert "\\" in eq or "_" in eq or "^" in eq


def test_html_block_embeds_png_equations():
    pytest.importorskip("matplotlib")
    blob = html_block("waveforms")
    assert "<div class=\"math\">" in blob
    assert "data:image/png;base64," in blob
    assert "\\[" not in blob
    assert blob.count("<img ") == len(NOTES["waveforms"].equations)


def test_every_note_equation_renders_offline():
    pytest.importorskip("matplotlib")
    from matplotlib.mathtext import MathTextParser

    from epochlens.explorer.mathnotes import mathtext_safe

    parser = MathTextParser("path")
    for key, note in NOTES.items():
        blob = html_block(key)
        assert blob.count("data:image/png;base64,") == len(note.equations)
        for eq in note.equations:
            parser.parse("$" + mathtext_safe(eq) + "$")
