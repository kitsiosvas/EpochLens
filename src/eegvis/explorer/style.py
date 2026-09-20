"""Shared visual constants for the HTML report (and Streamlit parity)."""

from __future__ import annotations

CLASS_PALETTE: tuple[str, ...] = (
    "#1F4E5F",
    "#B85C38",
    "#2F6D4F",
    "#6B3F69",
    "#A67C2D",
    "#3F4C5B",
)

INK = "#1C1917"
MUTED = "#57534E"
RULE = "#E4D9C8"
PAPER = "#F3EEE4"
PANEL = "#FFFCF7"
PICK = "#8F2D21"
WINDOW_FILL = "#D8D0C2"
DPI = 128

REPORT_CSS = """
    :root { --ink: #1c1917; --muted: #57534e; --paper: #f3eee4; --panel: #fffcf7;
            --rule: #e4d9c8; --banner: #efe6d4; --banner-edge: #8a734a; }
    body { font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
           max-width: 1120px; margin: 2.75rem auto 4rem; padding: 0 1.75rem 2rem;
           color: var(--ink); background: var(--paper); line-height: 1.55; }
    h1 { font-family: Palatino, "Palatino Linotype", "Iowan Old Style", Georgia, serif;
         font-size: 2.2rem; font-weight: 600; letter-spacing: -0.02em;
         margin: 0 0 0.2rem; }
    .lede { color: var(--muted); margin: 0 0 1.15rem; font-size: 1.02rem; }
    .sub { color: var(--muted); margin: 0.35rem 0 1.1rem; }
    .banner { background: var(--banner); border: 1px solid #d9ccb4;
              border-left: 3px solid var(--banner-edge);
              padding: 0.8rem 1.05rem; margin: 0 0 1.35rem; color: #4a4338;
              font-size: 0.92rem; }
    .meta { color: var(--ink); margin: 0 0 1.5rem; }
    figure { margin: 2.1rem 0 0; background: var(--panel); padding: 1.2rem 1.25rem 1.05rem;
             border: 1px solid var(--rule); }
    figure h2 { font-size: 1.08rem; font-weight: 650; margin: 0 0 0.85rem; letter-spacing: -0.01em; }
    figure img { width: 100%; background: #fff; display: block; }
    figcaption { color: var(--muted); font-size: 0.88rem; margin: 0.75rem 0 0; line-height: 1.45; }
    h2 { font-size: 1.08rem; font-weight: 650; margin: 1.4rem 0 0.5rem; }
    table.rank { border-collapse: collapse; margin: 0.5rem 0 0.4rem; font-size: 0.9rem; }
    table.rank th, table.rank td { border: 1px solid var(--rule); padding: 0.32rem 0.75rem; text-align: left; }
    table.rank th { background: #efe8db; font-weight: 600; }
    table.rank td.num { font-variant-numeric: tabular-nums; }
    footer { color: var(--muted); font-size: 0.85rem; margin-top: 2.4rem; }
"""


def apply_matplotlib_rc() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#44403C",
            "axes.labelcolor": MUTED,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        }
    )
