"""Plotly figures for the explorer. No dataset constants."""

from __future__ import annotations

import numpy as np

from epochlens.explorer.style import CLASS_PALETTE

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install epochlens[explorer] for plotly") from exc


def _class_color(key, index: int) -> str:
    if key is None:
        return CLASS_PALETTE[index % len(CLASS_PALETTE)]
    return CLASS_PALETTE[int(key) % len(CLASS_PALETTE)]


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
                fig.add_vline(x=x, line_width=1, line_color="#1c1917", row=r + 1, col=c + 1)
    if zmin is None or zmax is None:
        vmax = float(np.nanpercentile(np.abs(scalogram[:n_ch]), 98))
        vmax = max(vmax, 1e-9)
        zmin = -vmax if zmin is None else zmin
        zmax = vmax if zmax is None else zmax
    fig.update_layout(
        title=title,
        coloraxis=dict(colorscale="RdBu", cmin=zmin, cmax=zmax, colorbar=dict(title="relative power")),
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
    markers = ["circle", "square", "diamond", "triangle-up", "x"]
    sessions = np.ones(len(labels), dtype=int) if sessions is None else sessions
    for cls in np.unique(labels):
        color = _class_color(cls, int(cls))
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
                    marker=dict(size=9, symbol=markers[int(ses) % len(markers)], color=color),
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


def mean_traces(
    times: np.ndarray,
    means: dict,
    sems: dict,
    class_names: dict[int, str],
    ch_names: list[str],
    window: tuple[float, float] | None = None,
) -> go.Figure:
    n_ch = len(ch_names)
    cols = 2 if n_ch > 1 else 1
    rows = int(np.ceil(n_ch / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=ch_names,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )
    keys = list(means.keys())
    ymax = 0.0
    for key in means:
        ymax = max(ymax, float(np.percentile(np.abs(means[key]), 99)))
    ymax = max(ymax * 1.25, 1.0)
    for i, name in enumerate(ch_names):
        r, c = divmod(i, cols)
        for k, key in enumerate(keys):
            y = means[key][i]
            e = sems[key][i]
            color = _class_color(key, k)
            label = "all" if key is None else class_names.get(int(key), str(key))
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=y,
                    mode="lines",
                    name=label,
                    legendgroup=str(key),
                    showlegend=i == 0,
                    line=dict(color=color, width=1.6),
                    hovertemplate=f"{name} · {label}<extra></extra>",
                ),
                row=r + 1,
                col=c + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate([y + e, (y - e)[::-1]]),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.12,
                    line=dict(width=0),
                    name=label,
                    legendgroup=str(key),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=r + 1,
                col=c + 1,
            )
        if window is not None:
            fig.add_vrect(
                x0=window[0],
                x1=window[1],
                fillcolor="#D8D0C2",
                opacity=0.35,
                line_width=0,
                row=r + 1,
                col=c + 1,
            )
            for x in window:
                fig.add_vline(x=x, line_width=1, line_dash="dash", line_color="#57534E", row=r + 1, col=c + 1)
    fig.update_layout(
        title="Class-mean waveforms ± SEM (ranked channels, baseline z)",
        height=max(280, 180 * rows),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Time (s)", row=rows, col=1)
    fig.update_yaxes(title_text="baseline z", row=1, col=1)
    fig.update_yaxes(range=[-ymax, ymax])
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


def pairwise_heatmaps(
    maps: np.ndarray,
    times,
    ch_names: list[str],
    pair_labels: list[str],
    title: str,
) -> go.Figure:
    n = len(pair_labels)
    cols = min(3, max(n, 1))
    rows = int(np.ceil(n / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=pair_labels,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    vmax = float(np.nanpercentile(maps, 98))
    vmax = max(vmax, 1e-9)
    for i in range(n):
        r, c = divmod(i, cols)
        fig.add_trace(
            go.Heatmap(
                z=maps[i],
                x=times,
                y=ch_names,
                coloraxis="coloraxis",
                showscale=False,
            ),
            row=r + 1,
            col=c + 1,
        )
    fig.update_layout(
        title=title,
        coloraxis=dict(colorscale="Hot", cmin=0, cmax=vmax, colorbar=dict(title="|z|")),
        height=max(320, 220 * rows),
        margin=dict(l=60, r=20, t=70, b=40),
    )
    fig.update_xaxes(title_text="Time (s)", row=rows, col=1)
    fig.update_yaxes(title_text="Channel", row=1, col=1)
    return fig


def _spectrum_ylim(freqs: np.ndarray, spec_means: dict) -> tuple[float, float, float]:
    fmax_show = min(float(np.asarray(freqs)[-1]), 45.0)
    if not spec_means:
        return 1e-6, 1.0, fmax_show
    show = np.asarray(freqs) <= fmax_show
    stacked = np.concatenate([np.asarray(spec_means[key])[:, show] for key in spec_means], axis=0)
    positives = stacked[stacked > 0]
    ymin = float(np.percentile(positives, 2)) if positives.size else 1e-6
    ymax = float(np.max(stacked)) if stacked.size else 1.0
    ymin = max(ymin * 0.5, 1e-12)
    ymax = max(ymax * 1.25, ymin * 10.0)
    return ymin, ymax, fmax_show


def class_mean_spectra(
    freqs: np.ndarray,
    spec_means: dict,
    class_names: dict[int, str],
    ch_names: list[str],
) -> go.Figure:
    n_ch = len(ch_names)
    cols = 2 if n_ch > 1 else 1
    rows = int(np.ceil(n_ch / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=ch_names,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )
    keys = list(spec_means.keys())
    ymin, ymax, fmax_show = _spectrum_ylim(freqs, spec_means)
    for i, name in enumerate(ch_names):
        r, c = divmod(i, cols)
        for k, key in enumerate(keys):
            y = spec_means[key][i]
            color = _class_color(key, k)
            label = "all" if key is None else class_names.get(int(key), str(key))
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=y,
                    mode="lines",
                    name=label,
                    legendgroup=str(key),
                    showlegend=i == 0,
                    line=dict(color=color, width=1.6),
                    hovertemplate=f"{name} · {label}<extra></extra>",
                ),
                row=r + 1,
                col=c + 1,
            )
    fig.update_layout(
        title="Class-mean spectra in the analysis window (ranked channels)",
        height=max(280, 180 * rows),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(range=[float(freqs[0]), fmax_show])
    fig.update_xaxes(title_text="Hz", row=rows, col=1)
    fig.update_yaxes(type="log", range=[float(np.log10(ymin)), float(np.log10(ymax))])
    fig.update_yaxes(title_text="Power", row=1, col=1)
    return fig


def band_topomaps(
    xy: np.ndarray,
    band_means: np.ndarray,
    band_names: list[str],
) -> go.Figure:
    from epochlens.topo import interpolate_topo

    xy = np.asarray(xy, dtype=np.float64)
    band_means = np.asarray(band_means, dtype=np.float64)
    n = len(band_names)
    if band_means.ndim != 2 or band_means.shape[1] != n:
        raise ValueError("band_means must be (n_channels, n_bands)")
    cols = min(4, max(n, 1))
    rows = int(np.ceil(n / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(band_names),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    coloraxes: dict[str, dict] = {}
    for i, name in enumerate(band_names):
        r, c = divmod(i, cols)
        Xi, Yi, Zi = interpolate_topo(xy, band_means[:, i])
        vmax = max(float(np.nanpercentile(np.abs(Zi), 98)), 1e-9)
        axis_i = r * cols + c + 1
        coloraxis = "coloraxis" if axis_i == 1 else f"coloraxis{axis_i}"
        fig.add_trace(
            go.Heatmap(
                z=Zi,
                x=Xi[0],
                y=Yi[:, 0],
                coloraxis=coloraxis,
                showscale=True,
                hoverongaps=False,
            ),
            row=r + 1,
            col=c + 1,
        )
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                name="sensors",
                marker=dict(size=6, color="#f3eee4", line=dict(width=0.6, color="#1C1917")),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r + 1,
            col=c + 1,
        )
        xname = "x" if axis_i == 1 else f"x{axis_i}"
        fig.update_xaxes(
            constrain="domain",
            showticklabels=False,
            title_text="",
            showgrid=False,
            zeroline=False,
            row=r + 1,
            col=c + 1,
        )
        fig.update_yaxes(
            scaleanchor=xname,
            scaleratio=1,
            constrain="domain",
            showticklabels=False,
            title_text="",
            showgrid=False,
            zeroline=False,
            row=r + 1,
            col=c + 1,
        )
        coloraxes[coloraxis] = dict(
            colorscale="Inferno",
            cmin=0,
            cmax=vmax,
            colorbar=dict(
                title=name,
                len=0.72 / rows,
                thickness=10,
                x=min(1.02, (c + 1) / cols - 0.01),
                y=1.0 - (r + 0.5) / rows,
                yanchor="middle",
                outlinewidth=0,
            ),
        )
    fig.update_layout(
        title="Band-power topography",
        height=max(340, 300 * rows),
        margin=dict(l=20, r=50, t=60, b=20),
        **coloraxes,
    )
    return fig
