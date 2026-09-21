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

# Diverging relative-power (blue = below baseline, red = above). Midpoint is 0.
REL_POWER_SCALE: list[list] = [
    [0.0, "rgb(5,48,97)"],
    [0.25, "rgb(67,147,195)"],
    [0.5, "rgb(247,247,247)"],
    [0.75, "rgb(215,96,67)"],
    [1.0, "rgb(103,0,31)"],
]

# Sequential |z| / magnitude: paper cream toward PICK, not neon Hot.
ZABS_SCALE: list[list] = [
    [0.0, "#FFFCF7"],
    [0.15, "#F6E8C3"],
    [0.35, "#E0C07A"],
    [0.58, "#D08A45"],
    [0.8, "#B85C38"],
    [1.0, PICK],
]

# Sequential band power: paper toward ink teal.
POWER_SCALE: list[list] = [
    [0.0, "#FFFCF7"],
    [0.2, "#E4EDE6"],
    [0.45, "#A7C4B2"],
    [0.7, "#4E8A6A"],
    [1.0, CLASS_PALETTE[0]],
]

BAND_GLYPH: dict[str, str] = {
    "theta": "θ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
}

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
    figure img { width: 100%; background: var(--panel); display: block; }
    figcaption { color: var(--muted); font-size: 0.88rem; margin: 0.75rem 0 0; line-height: 1.45; }
    h2 { font-size: 1.08rem; font-weight: 650; margin: 1.4rem 0 0.5rem; }
    table.rank { border-collapse: collapse; margin: 0.5rem 0 0.4rem; font-size: 0.9rem; }
    table.rank th, table.rank td { border: 1px solid var(--rule); padding: 0.32rem 0.75rem; text-align: left; }
    table.rank th { background: #efe8db; font-weight: 600; }
    table.rank td.num { font-variant-numeric: tabular-nums; }
    footer { color: var(--muted); font-size: 0.85rem; margin-top: 2.4rem; }
    .math { margin: 0.7rem 0 0; padding: 0.7rem 0.85rem; background: #f7f1e6;
            border: 1px dashed var(--rule); color: #4a4338; font-size: 0.9rem; }
    .math p { margin: 0 0 0.45rem; }
    .math .eq { margin: 0.35rem 0 0; overflow-x: auto; }
    .math .eq img { width: auto; max-width: 100%; height: auto; background: transparent; display: block; margin: 0 auto; }
"""


def hex_to_rgba(color: str, alpha: float) -> str:
    """``#RRGGBB`` (or ``#RGB``) to an ``rgba()`` CSS color."""
    h = color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def clipped_bands(fmin: float, fmax: float) -> list[tuple[str, float, float]]:
    """θ/α/β/γ intervals clipped to a plotted frequency range."""
    from epochlens.bands import BANDS

    out: list[tuple[str, float, float]] = []
    for name, lo, hi in BANDS:
        a = max(float(lo), float(fmin))
        b = min(float(hi), float(fmax))
        if b > a:
            out.append((str(name), a, b))
    return out


def head_disk(xy) -> tuple[tuple[float, float], float] | None:
    """Circular head ``((cx, cy), radius)`` around finite sensor coordinates."""
    import numpy as np

    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        return None
    located = xy[np.isfinite(xy).all(axis=1)]
    if located.shape[0] < 1:
        return None
    center = located.mean(axis=0)
    radius = float(np.max(np.linalg.norm(located - center, axis=1)) * 1.15)
    if not np.isfinite(radius) or radius <= 0:
        return None
    return (float(center[0]), float(center[1])), radius


def head_axis_ranges(xy) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """``(xlim, ylim)`` that fit the head cartoon (nose up)."""
    disk = head_disk(xy)
    if disk is None:
        return None
    (cx, cy), r = disk
    return (cx - 1.35 * r, cx + 1.35 * r), (cy - 1.25 * r, cy + 1.55 * r)


def head_plotly_shapes(xy, xref: str = "x", yref: str = "y") -> list[dict]:
    """Circle, nose (y+), and ear ticks in data coordinates."""
    disk = head_disk(xy)
    if disk is None:
        return []
    (cx, cy), r = disk
    line = dict(color=INK, width=1.35)
    nose_w = 0.14 * r
    nose_h = 0.16 * r
    ear_w = 0.16 * r
    ear_h = 0.22 * r
    return [
        dict(
            type="circle",
            xref=xref,
            yref=yref,
            x0=cx - r,
            y0=cy - r,
            x1=cx + r,
            y1=cy + r,
            line=line,
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        ),
        dict(
            type="path",
            xref=xref,
            yref=yref,
            path=(
                f"M {cx - nose_w} {cy + r - 0.02 * r} "
                f"L {cx} {cy + r + nose_h} "
                f"L {cx + nose_w} {cy + r - 0.02 * r}"
            ),
            line=line,
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        ),
        dict(
            type="path",
            xref=xref,
            yref=yref,
            path=(
                f"M {cx - r} {cy - ear_h} "
                f"Q {cx - r - ear_w} {cy} {cx - r} {cy + ear_h}"
            ),
            line=line,
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        ),
        dict(
            type="path",
            xref=xref,
            yref=yref,
            path=(
                f"M {cx + r} {cy - ear_h} "
                f"Q {cx + r + ear_w} {cy} {cx + r} {cy + ear_h}"
            ),
            line=line,
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        ),
    ]


def _segmented_cmap(name: str, scale: list[list]):
    from matplotlib.colors import LinearSegmentedColormap

    colors = [stop[1] for stop in scale]
    return LinearSegmentedColormap.from_list(name, colors)


def zabs_cmap():
    """Matplotlib colormap matching ``ZABS_SCALE``."""
    return _segmented_cmap("epochlens_zabs", ZABS_SCALE)


def power_cmap():
    """Matplotlib colormap matching ``POWER_SCALE``."""
    return _segmented_cmap("epochlens_power", POWER_SCALE)


def epochlens_template():
    """Plotly template: paper/panel, ink text, muted ticks, no extra grid."""
    import plotly.graph_objects as go

    axis = dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor="#44403C",
        linewidth=0.8,
        ticks="outside",
        tickcolor=MUTED,
        tickfont=dict(color=MUTED, size=11),
        title=dict(font=dict(color=MUTED, size=12)),
        automargin=True,
        mirror=False,
    )
    xaxis = dict(
        **axis,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor=MUTED,
        spikethickness=1,
        spikedash="dot",
    )
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=PAPER,
            plot_bgcolor=PANEL,
            font=dict(family="Segoe UI, system-ui, sans-serif", color=INK, size=12),
            title=dict(font=dict(size=14, color=INK), x=0.0, xanchor="left"),
            colorway=list(CLASS_PALETTE),
            hovermode="closest",
            hoverlabel=dict(
                bgcolor=PANEL,
                font=dict(color=INK, size=12, family="Segoe UI, system-ui, sans-serif"),
                bordercolor=RULE,
            ),
            legend=dict(
                bgcolor="rgba(255,252,247,0.94)",
                bordercolor=RULE,
                borderwidth=1,
                font=dict(size=11, color=INK),
                itemsizing="constant",
                tracegroupgap=6,
            ),
            margin=dict(l=56, r=36, t=64, b=52),
            coloraxis=dict(
                colorbar=dict(
                    outlinewidth=0,
                    tickfont=dict(color=MUTED, size=10),
                    title=dict(font=dict(color=MUTED, size=11)),
                    thickness=12,
                    len=0.78,
                )
            ),
            xaxis=xaxis,
            yaxis=axis,
        )
    )


def apply_plotly_style(fig, *, hovermode: str = "closest", **layout):
    """Apply the EpochLens template, then any figure-specific layout kwargs."""
    fig.update_layout(template=epochlens_template(), hovermode=hovermode, **layout)
    return fig


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
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": PAPER,
            "axes.facecolor": PANEL,
            "savefig.facecolor": PAPER,
            "savefig.edgecolor": PAPER,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )
