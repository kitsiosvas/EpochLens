"""CLI: write an HTML report (default) or launch Streamlit."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="eegvis-explorer",
        description="EEG visualization explorer. Not a 4-class inner-speech decoder.",
    )
    parser.add_argument("--source", choices=["synthetic", "nieto"], default="synthetic")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--condition", default="inner")
    parser.add_argument("--root", default="", help="Nieto dataset root (folder with derivatives/)")
    parser.add_argument("--out", default="eegvis_report.html")
    parser.add_argument("--window", default="", help="action window tmin,tmax in seconds")
    parser.add_argument("--baseline-end", type=float, default=None)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    parser.add_argument("--streamlit", action="store_true", help="interactive Streamlit UI")
    args = parser.parse_args(argv)

    if args.streamlit:
        app = Path(__file__).with_name("app.py")
        sys.argv = ["streamlit", "run", str(app), "--browser.gatherUsageStats=false"]
        from streamlit.web import cli as stcli

        raise SystemExit(stcli.main())

    from eegvis.adapters.synthetic import make_synthetic
    from eegvis.explorer.report import render_report, write_report

    if args.source == "synthetic":
        batch = make_synthetic()
        window = (0.6, 1.4)
        baseline = (batch.tmin, 0.5)
    else:
        from eegvis.adapters.nieto import load_nieto

        kwargs = {"subject": args.subject, "condition": args.condition}
        if args.root.strip():
            kwargs["root"] = args.root.strip()
        batch = load_nieto(**kwargs)
        window = (1.0, 3.5)
        baseline = (batch.tmin, min(batch.tmin + 0.5, 0.5))

    if args.window:
        parts = [float(x) for x in args.window.split(",")]
        if len(parts) != 2:
            raise SystemExit("--window must be tmin,tmax")
        window = (parts[0], parts[1])
    if args.baseline_end is not None:
        baseline = (batch.tmin, args.baseline_end)

    html = render_report(batch, window=window, baseline=baseline)
    out = write_report(html, Path(args.out))
    print(f"Wrote {out.resolve()}")
    print("Honest note: 4-class inner speech on Nieto 2022 is typically near chance (~25–37%).")
    if args.open_browser:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
