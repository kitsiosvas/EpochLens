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


def test_html_block_wraps_display_math():
    blob = html_block("waveforms")
    assert "\\[" in blob
    assert "\\mathrm{SEM}" in blob
    assert "<div class=\"math\">" in blob
