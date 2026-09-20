"""CLI: Streamlit explorer by default; optional static HTML export."""

from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

from eegvis.windows import default_windows


def _load_batch(source: str, fif: str, npz: str):
    if source == "synthetic":
        from eegvis.adapters.synthetic import make_synthetic

        return make_synthetic()
    if source == "fif":
        from eegvis.adapters.fif import load_epochs_fif

        if not fif:
            raise SystemExit("--fif path is required")
        return load_epochs_fif(fif)
    from eegvis.adapters.npz import load_epochs

    if not npz:
        raise SystemExit("--npz path is required")
    return load_epochs(npz)


def _launch_streamlit(*, fif: str = "", npz: str = "") -> None:
    if fif:
        os.environ["EEGVIS_FIF"] = fif
    if npz:
        os.environ["EEGVIS_NPZ"] = npz
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), "--browser.gatherUsageStats=false"]
    from streamlit.web import cli as stcli

    raise SystemExit(stcli.main())


def _write_html(args) -> None:
    from eegvis.explorer.report import render_report, write_report

    batch = _load_batch(args.source, args.fif.strip(), args.npz.strip())
    window, baseline = default_windows(batch.dataset, batch.tmin, float(batch.times[-1]))
    if args.window:
        parts = [float(x) for x in args.window.split(",")]
        if len(parts) != 2:
            raise SystemExit("--window must be tmin,tmax")
        window = (parts[0], parts[1])
    if args.baseline_end is not None:
        baseline = (batch.tmin, args.baseline_end)

    t0 = time.perf_counter()
    html = render_report(
        batch,
        window=window,
        baseline=baseline,
        top_k=args.top_k,
        sidecar=Path(args.out).with_suffix(".json"),
        full=args.full,
    )
    out = write_report(html, Path(args.out))
    print(f"Wrote {out.resolve()} in {time.perf_counter() - t0:.1f}s")
    if args.open_browser:
        webbrowser.open(out.resolve().as_uri())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="eegvis-explorer",
        description=(
            "Streamlit explorer for labeled EEG epochs. "
            "Pass --html to write a static matplotlib report instead."
        ),
    )
    parser.add_argument("--source", choices=["synthetic", "fif", "npz"], default="synthetic")
    parser.add_argument("--fif", default="", help="MNE *-epo.fif from any experiment")
    parser.add_argument("--npz", default="", help="eegvis NPZ epochs from any experiment")
    parser.add_argument(
        "--html",
        action="store_true",
        help="write a static HTML report instead of launching Streamlit",
    )
    parser.add_argument("--out", default="eegvis_report.html", help="HTML output path (with --html)")
    parser.add_argument("--window", default="", help="analysis window tmin,tmax in seconds (with --html)")
    parser.add_argument("--baseline-end", type=float, default=None, help="baseline end in seconds (with --html)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="channels kept for waveforms / CWT visualization (not MDS or the chance check)",
    )
    parser.add_argument("--full", action="store_true", help="HTML only: extra figures (per-class CWT, topomaps)")
    parser.add_argument("--open", action="store_true", dest="open_browser", help="open the HTML report (with --html)")
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="launch Streamlit (default; kept so older commands still work)",
    )
    args = parser.parse_args(argv)
    if args.fif.strip():
        args.source = "fif"
    if args.npz.strip():
        args.source = "npz"

    if args.html:
        _write_html(args)
        return

    _launch_streamlit(fif=args.fif.strip(), npz=args.npz.strip())


if __name__ == "__main__":
    main()
