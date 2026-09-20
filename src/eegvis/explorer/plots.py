"""Plotly figures for the explorer. No dataset constants."""

from __future__ import annotations

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install eegvis[explorer] for plotly") from exc


def scalogram_grid(
    scalogram: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    ch_names: list[str],
    *,
    title: str,
    window: tuple[float, float] | None = None,
    max_channels: int = 16,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """``scalogram`` is (n_channels, n_freqs, n_times)."""
    n_ch = min(scalogram.shape[0], max_channels)
    cols = 4
    rows = int(np.ceil(n_ch / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=ch_names[:n_ch],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )
    for i in range(n_ch):
        r, c = divmod(i, cols)
        fig.add_trace(
            go.Heatmap(
                z=scalogram[i],
                x=times,
                y=freqs,
                coloraxis="coloraxis",
                showscale=False,
            ),
            row=r + 1,
            col=c + 1,
        )
        if window is not None:
            for x in window:
                fig.add_vline(x=x, line_width=1, line_color="white", row=r + 1, col=c + 1)
    fig.update_layout(
        title=title,
        coloraxis=dict(colorscale="Hot", cmin=zmin, cmax=zmax),
        height=max(280, 160 * rows),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Time (s)", row=rows, col=1)
    fig.update_yaxes(title_text="Hz", row=1, col=1)
    return fig


def channel_stem(scores: np.ndarray, ch_names: list[str], title: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(x=list(range(1, len(scores) + 1)), y=scores, hovertext=ch_names, name="score")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Channel index",
        yaxis_title="Score",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def scalp_scatter(
    xy: np.ndarray,
    highlight: np.ndarray,
    title: str,
) -> go.Figure:
    mask = np.zeros(xy.shape[0], dtype=bool)
    mask[np.asarray(highlight, dtype=int)] = True
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xy[~mask, 0],
            y=xy[~mask, 1],
            mode="markers",
            name="other",
            marker=dict(size=8, color="#888"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xy[mask, 0],
            y=xy[mask, 1],
            mode="markers",
            name="selected",
            marker=dict(size=12, color="#c44"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def mds_scatter(
    xy: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray | None,
    class_names: dict[int, str],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    markers = ["circle", "square", "diamond", "cross", "x"]
    sessions = np.ones(len(labels), dtype=int) if sessions is None else sessions
    for cls in np.unique(labels):
        for ses in np.unique(sessions):
            m = (labels == cls) & (sessions == ses)
            if not np.any(m):
                continue
            fig.add_trace(
                go.Scatter(
                    x=xy[m, 0],
                    y=xy[m, 1],
                    mode="markers",
                    name=f"{class_names.get(int(cls), cls)} / ses {ses}",
                    marker=dict(size=9, symbol=markers[int(ses) % len(markers)]),
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="MDS 1",
        yaxis_title="MDS 2",
        yaxis_scaleanchor="x",
        height=480,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def heatmap(
    z: np.ndarray,
    x,
    y,
    title: str,
    xaxis: str,
    yaxis: str,
) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale="Hot"))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
