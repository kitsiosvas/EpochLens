"""Plotly figures for the explorer. No dataset constants."""

from __future__ import annotations

import numpy as np

from epochlens.explorer.style import (
    BAND_GLYPH,
    CLASS_PALETTE,
    INK,
    MUTED,
    PANEL,
    PICK,
    POWER_SCALE,
    REL_POWER_SCALE,
    RULE,
    WINDOW_FILL,
    ZABS_SCALE,
    apply_plotly_style,
    clipped_bands,
    head_axis_ranges,
    head_plotly_shapes,
    hex_to_rgba,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install epochlens[explorer] for plotly") from exc


def _class_color(key, index: int) -> str:
    if key is None:
        return CLASS_PALETTE[index % len(CLASS_PALETTE)]
    return CLASS_PALETTE[int(key) % len(CLASS_PALETTE)]


def _class_label(key, class_names: dict) -> str:
    if key is None:
        return "all"
    return class_names.get(int(key), str(key))


def _xy_ref(row: int, col: int, ncols: int) -> tuple[str, str]:
    axis_i = (row - 1) * ncols + col
    if axis_i == 1:
        return "x", "y"
    return f"x{axis_i}", f"y{axis_i}"


def _legend_below(rows: int) -> dict:
    return dict(
        orientation="h",
        yanchor="top",
        y=-0.10 if rows <= 1 else -0.06,
        x=0.5,
        xanchor="center",
        bgcolor="rgba(255,252,247,0.0)",
        borderwidth=0,
        font=dict(size=11, color=INK),
        itemsizing="constant",
        tracegroupgap=8,
    )


def _add_window(fig, window: tuple[float, float] | None, row: int, col: int) -> None:
    if window is None:
        return
    fig.add_vrect(
        x0=window[0],
        x1=window[1],
        fillcolor=WINDOW_FILL,
        opacity=0.45,
        line_width=0,
        layer="below",
        row=row,
        col=col,
    )
    for x in window:
        fig.add_vline(
            x=x,
            line_width=1,
            line_dash="dash",
            line_color=MUTED,
            row=row,
            col=col,
        )


def _add_band_guides(fig, fmin: float, fmax: float, row: int, col: int, *, label: bool) -> None:
    bands = clipped_bands(fmin, fmax)
    for i, (name, a, b) in enumerate(bands):
        fig.add_vrect(
            x0=a,
            x1=b,
            fillcolor=WINDOW_FILL if i % 2 == 0 else RULE,
            opacity=0.38 if i % 2 == 0 else 0.28,
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )
        if label:
            fig.add_annotation(
                x=(a + b) / 2.0,
                y=0.97,
                yref="y domain",
                text=BAND_GLYPH.get(name, name),
                showarrow=False,
                font=dict(size=10, color=MUTED),
                row=row,
                col=col,
            )
    edges = sorted({a for _, a, _ in bands} | {b for _, _, b in bands})
    for x in edges:
        if fmin < x < fmax:
            fig.add_vline(
                x=x,
                line_width=0.8,
                line_dash="dot",
                line_color=MUTED,
                row=row,
                col=col,
            )


def _blank_unused(fig, n_used: int, rows: int, cols: int) -> None:
    for i in range(n_used, rows * cols):
        r, c = divmod(i, cols)
        fig.update_xaxes(visible=False, showspikes=False, row=r + 1, col=c + 1)
        fig.update_yaxes(visible=False, showspikes=False, row=r + 1, col=c + 1)


def _hide_xy(fig, row: int, col: int, xname: str) -> None:
    fig.update_xaxes(
        constrain="domain",
        showticklabels=False,
        title_text="",
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showspikes=False,
        row=row,
        col=col,
    )
    fig.update_yaxes(
        scaleanchor=xname,
        scaleratio=1,
        constrain="domain",
        showticklabels=False,
        title_text="",
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showspikes=False,
        row=row,
        col=col,
    )


def _channel_hover_names(n: int, ch_names: list[str] | None) -> list[str]:
    if ch_names is not None and len(ch_names) == n:
        return list(ch_names)
    return [f"ch {i}" for i in range(n)]


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
        subplot_titles=list(ch_names[:n_ch]),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )
    if zmin is None or zmax is None:
        vmax = float(np.nanpercentile(np.abs(scalogram[:n_ch]), 98))
        vmax = max(vmax, 1e-9)
        zmin = -vmax if zmin is None else zmin
        zmax = vmax if zmax is None else zmax
    for i in range(n_ch):
        r, c = divmod(i, cols)
        fig.add_trace(
            go.Heatmap(
                z=scalogram[i],
                x=times,
                y=freqs,
                coloraxis="coloraxis",
                showscale=False,
                zmid=0,
                hovertemplate=(
                    "Time = %{x:.3f} s<br>Frequency = %{y:.1f} Hz<br>"
                    "rel. power = %{z:.3f}<extra>" + ch_names[i] + "</extra>"
                ),
            ),
            row=r + 1,
            col=c + 1,
        )
        if window is not None:
            for x in window:
                fig.add_vline(
                    x=x,
                    line_width=1,
                    line_dash="dash",
                    line_color=INK,
                    row=r + 1,
                    col=c + 1,
                )
    apply_plotly_style(
        fig,
        hovermode="closest",
        title=title,
        coloraxis=dict(
            colorscale=REL_POWER_SCALE,
            cmin=zmin,
            cmax=zmax,
            cmid=0,
            colorbar=dict(title=dict(text="rel. power", side="right"), outlinewidth=0),
        ),
        height=max(300, 170 * rows),
        margin=dict(l=56, r=72, t=64, b=52),
    )
    fig.update_xaxes(title_text="Time (s)", row=rows, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    _blank_unused(fig, n_ch, rows, cols)
    return fig


def channel_stem(
    scores: np.ndarray,
    ch_names: list[str],
    title: str,
    highlight: np.ndarray | None = None,
    *,
    bad: np.ndarray | None = None,
    selected: int | None = None,
) -> go.Figure:
    scores = np.asarray(scores, dtype=np.float64)
    names = list(ch_names)
    n = len(scores)
    pick = set()
    if highlight is not None:
        pick = {int(i) for i in np.asarray(highlight, dtype=int)}
    bad_mask = np.zeros(n, dtype=bool)
    if bad is not None:
        bad_mask = np.asarray(bad, dtype=bool)
        if bad_mask.shape != (n,):
            raise ValueError(f"bad must have length {n}")
    bad_fill = "#C4B8A5"
    colors: list[str] = []
    patterns: list[str] = []
    line_widths: list[float] = []
    line_colors: list[str] = []
    sel = None if selected is None else int(selected)
    for i in range(n):
        is_bad = bool(bad_mask[i])
        is_sel = sel is not None and i == sel
        if is_bad:
            fill = bad_fill
            patterns.append("/")
        elif i in pick:
            fill = PICK
            patterns.append("")
        else:
            fill = MUTED
            patterns.append("")
        if is_sel:
            if is_bad:
                fill = bad_fill
            elif i in pick:
                fill = PICK
            else:
                fill = MUTED
            line_widths.append(2.5)
            line_colors.append(INK)
        else:
            line_widths.append(0)
            line_colors.append("rgba(0,0,0,0)")
        colors.append(fill)
    fig = go.Figure(
        go.Bar(
            x=names,
            y=scores,
            customdata=np.arange(n),
            marker=dict(
                color=colors,
                line=dict(width=line_widths, color=line_colors),
                pattern=dict(shape=patterns),
            ),
            width=0.62,
            hovertemplate="%{x}<br>score = %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    n_ticks = n
    tickangle = -40 if n_ticks <= 18 else -90
    ticksize = 11 if n_ticks <= 12 else (9 if n_ticks <= 24 else 8)
    apply_plotly_style(
        fig,
        hovermode="closest",
        title=title,
        height=max(320, 280 + min(n, 8) * 4),
        margin=dict(l=56, r=24, t=56, b=88 if tickangle != 0 else 52),
        bargap=0.28,
    )
    fig.update_xaxes(
        title_text="",
        tickangle=tickangle,
        tickfont=dict(size=ticksize, color=MUTED),
        showspikes=False,
    )
    fig.update_yaxes(title_text="Score", zeroline=True, zerolinecolor=RULE, zerolinewidth=1)
    return fig


def trial_strip(
    times: np.ndarray,
    trials: np.ndarray,
    labels: np.ndarray | None,
    class_names: dict,
    *,
    channel_name: str,
    window: tuple[float, float] | None = None,
    max_per_class: int = 2,
) -> go.Figure:
    """Overlay a few raw trials and class means for one channel.

    ``trials`` is ``(n_trials, n_times)``, already baseline z-scored by the caller.
    """
    times = np.asarray(times, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim != 2 or trials.shape[1] != times.shape[0]:
        raise ValueError("trials must be (n_trials, n_times) matching times")
    fig = make_subplots(rows=1, cols=1)
    _add_window(fig, window, 1, 1)

    if labels is None:
        n_show = min(8, trials.shape[0])
        for i in range(n_show):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=trials[i],
                    mode="lines",
                    line=dict(color=_class_color(None, 0), width=1),
                    opacity=0.35,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
        mean = trials.mean(axis=0)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=mean,
                mode="lines",
                name="all",
                line=dict(color=_class_color(None, 0), width=2.4),
                hovertemplate="all<br>z = %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    else:
        labels = np.asarray(labels)
        keys = [int(k) for k in np.unique(labels)]
        for k, key in enumerate(keys):
            color = _class_color(key, k)
            label = _class_label(key, class_names)
            cls_idx = np.flatnonzero(labels == key)
            for ti in cls_idx[:max_per_class]:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=trials[ti],
                        mode="lines",
                        line=dict(color=color, width=1),
                        opacity=0.35,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
            mean = trials[cls_idx].mean(axis=0)
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=mean,
                    mode="lines",
                    name=label,
                    legendgroup=str(key),
                    line=dict(color=color, width=2.4),
                    hovertemplate=f"{label}<br>z = %{{y:.3f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    apply_plotly_style(
        fig,
        hovermode="closest",
        title=str(channel_name),
        height=380,
        margin=dict(l=56, r=28, t=56, b=56),
        legend=_legend_below(1),
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="z", zeroline=True, zerolinecolor=RULE, zerolinewidth=1)
    return fig


def class_scalogram_grid(
    rel_by_class: dict,
    times: np.ndarray,
    freqs: np.ndarray,
    ch_names: list[str],
    class_names: dict[int, str],
    window: tuple[float, float] | None = None,
) -> go.Figure:
    """Rows are channels, columns are classes. Shared diverging relative-power scale."""
    keys = [int(k) for k in rel_by_class.keys()]
    n_ch = len(ch_names)
    n_cls = len(keys)
    if n_ch < 1 or n_cls < 1:
        raise ValueError("need at least one channel and one class")
    titles: list[str] = []
    for r in range(n_ch):
        for key in keys:
            if r == 0:
                titles.append(str(class_names.get(key, key)))
            else:
                titles.append("")
    fig = make_subplots(
        rows=n_ch,
        cols=n_cls,
        subplot_titles=titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.07,
    )
    stacked = np.concatenate([np.asarray(rel_by_class[key]) for key in keys], axis=0)
    vmax = float(np.nanpercentile(np.abs(stacked), 98))
    vmax = max(vmax, 1e-9)
    for r, name in enumerate(ch_names):
        for c, key in enumerate(keys):
            fig.add_trace(
                go.Heatmap(
                    z=rel_by_class[key][r],
                    x=times,
                    y=freqs,
                    coloraxis="coloraxis",
                    showscale=False,
                    zmid=0,
                    hovertemplate=(
                        "Time = %{x:.3f} s<br>Frequency = %{y:.1f} Hz<br>"
                        "rel. power = %{z:.3f}<extra>"
                        + f"{name} · {_class_label(key, class_names)}"
                        + "</extra>"
                    ),
                ),
                row=r + 1,
                col=c + 1,
            )
            if window is not None:
                for x in window:
                    fig.add_vline(
                        x=x,
                        line_width=1,
                        line_dash="dash",
                        line_color=INK,
                        row=r + 1,
                        col=c + 1,
                    )
        fig.update_yaxes(title_text=f"{name} (Hz)", row=r + 1, col=1)
    apply_plotly_style(
        fig,
        hovermode="closest",
        coloraxis=dict(
            colorscale=REL_POWER_SCALE,
            cmin=-vmax,
            cmax=vmax,
            cmid=0,
            colorbar=dict(title=dict(text="rel. power", side="right"), outlinewidth=0),
        ),
        height=max(300, 170 * n_ch),
        margin=dict(l=80, r=72, t=52, b=52),
    )
    fig.update_xaxes(title_text="Time (s)", row=n_ch, col=1)
    return fig


def scalp_scatter(
    xy: np.ndarray,
    highlight: np.ndarray,
    title: str,
    ch_names: list[str] | None = None,
    *,
    bad: np.ndarray | None = None,
    selected: int | None = None,
) -> go.Figure:
    from epochlens.topo import located_mask

    xy = np.asarray(xy, dtype=np.float64)
    ok = located_mask(xy)
    mask = np.zeros(xy.shape[0], dtype=bool)
    mask[np.asarray(highlight, dtype=int)] = True
    names = _channel_hover_names(xy.shape[0], ch_names)
    idx = np.arange(xy.shape[0])
    n = xy.shape[0]
    bad_mask = np.zeros(n, dtype=bool)
    if bad is not None:
        bad_mask = np.asarray(bad, dtype=bool)
        if bad_mask.shape != (n,):
            raise ValueError(f"bad must have length {n}")
    fig = go.Figure()
    # Bad located sensors: open MUTED circles, excluded from highlight set.
    bad_loc = ok & bad_mask
    other = ok & ~mask & ~bad_mask
    hi = ok & mask & ~bad_mask
    fig.add_trace(
        go.Scatter(
            x=xy[other, 0],
            y=xy[other, 1],
            mode="markers",
            name="other",
            marker=dict(size=9, color=CLASS_PALETTE[0], line=dict(width=0.6, color=INK)),
            customdata=idx[other],
            hovertext=[names[i] for i in idx[other]],
            hovertemplate="%{hovertext}<extra>other</extra>",
        )
    )
    if np.any(bad_loc):
        fig.add_trace(
            go.Scatter(
                x=xy[bad_loc, 0],
                y=xy[bad_loc, 1],
                mode="markers",
                name="bad",
                marker=dict(
                    size=9,
                    symbol="circle-open",
                    color=MUTED,
                    line=dict(width=1.2, color=MUTED),
                ),
                customdata=idx[bad_loc],
                hovertext=[names[i] for i in idx[bad_loc]],
                hovertemplate="%{hovertext}<extra>bad</extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=xy[hi, 0],
            y=xy[hi, 1],
            mode="markers",
            name="selected",
            marker=dict(size=13, color=PICK, line=dict(width=0.8, color=INK)),
            customdata=idx[hi],
            hovertext=[names[i] for i in idx[hi]],
            hovertemplate="%{hovertext}<extra>selected</extra>",
        )
    )
    if selected is not None:
        sel = int(selected)
        if 0 <= sel < n and ok[sel]:
            fig.add_trace(
                go.Scatter(
                    x=[xy[sel, 0]],
                    y=[xy[sel, 1]],
                    mode="markers",
                    name="focus",
                    marker=dict(
                        size=18,
                        symbol="circle-open",
                        color=INK,
                        line=dict(width=2, color=INK),
                    ),
                    customdata=[sel],
                    hovertext=[names[sel]],
                    hovertemplate="%{hovertext}<extra>focus</extra>",
                    showlegend=False,
                )
            )
    shapes = head_plotly_shapes(xy)
    ranges = head_axis_ranges(xy)
    apply_plotly_style(
        fig,
        hovermode="closest",
        title=title,
        height=460,
        margin=dict(l=24, r=24, t=56, b=56),
        shapes=shapes,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,252,247,0.0)",
            borderwidth=0,
        ),
    )
    fig.update_xaxes(
        title_text="",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showspikes=False,
        constrain="domain",
    )
    fig.update_yaxes(
        title_text="",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        showspikes=False,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    if ranges is not None:
        fig.update_xaxes(range=list(ranges[0]))
        fig.update_yaxes(range=list(ranges[1]))
    return fig


def _class_hulls(xy: np.ndarray, labels: np.ndarray) -> list[go.Scatter]:
    traces: list[go.Scatter] = []
    try:
        from scipy.spatial import ConvexHull
    except ImportError:  # pragma: no cover
        return traces
    global_span = float(np.max(np.ptp(xy, axis=0)))
    if global_span <= 0:
        return traces
    for cls in np.unique(labels):
        pts = xy[labels == cls]
        if pts.shape[0] < 6:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        area = float(getattr(hull, "volume", 0.0))
        if area > 0.40 * global_span * global_span:
            continue
        verts = np.append(hull.vertices, hull.vertices[0])
        poly = pts[verts]
        color = _class_color(cls, int(cls))
        traces.append(
            go.Scatter(
                x=poly[:, 0],
                y=poly[:, 1],
                mode="lines",
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.08),
                line=dict(color=hex_to_rgba(color, 0.45), width=1),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"hull-{int(cls)}",
            )
        )
    return traces


def mds_scatter(
    xy: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray | None,
    class_names: dict[int, str],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    for hull in _class_hulls(np.asarray(xy, dtype=np.float64), np.asarray(labels)):
        fig.add_trace(hull)
    markers = ["circle", "square", "diamond", "triangle-up", "x"]
    sessions = np.ones(len(labels), dtype=int) if sessions is None else sessions
    for cls in np.unique(labels):
        color = _class_color(cls, int(cls))
        for ses in np.unique(sessions):
            m = (labels == cls) & (sessions == ses)
            if not np.any(m):
                continue
            label = f"{class_names.get(int(cls), cls)} / ses {ses}"
            fig.add_trace(
                go.Scatter(
                    x=xy[m, 0],
                    y=xy[m, 1],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    marker=dict(
                        size=10,
                        symbol=markers[int(ses) % len(markers)],
                        color=color,
                        line=dict(width=0.4, color=INK),
                    ),
                    hovertemplate=(
                        f"{class_names.get(int(cls), cls)} · session {ses}"
                        "<br>MDS 1 = %{x:.2f}<br>MDS 2 = %{y:.2f}<extra></extra>"
                    ),
                )
            )
    apply_plotly_style(
        fig,
        hovermode="closest",
        title=title,
        height=520,
        margin=dict(l=56, r=170, t=56, b=48),
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top",
            bgcolor=PANEL,
            bordercolor=RULE,
            borderwidth=1,
            font=dict(size=12, color=INK),
            itemsizing="constant",
            itemwidth=40,
        ),
    )
    fig.update_xaxes(
        title_text="MDS 1",
        zeroline=True,
        zerolinecolor=RULE,
        zerolinewidth=1,
        showspikes=False,
    )
    fig.update_yaxes(
        title_text="MDS 2",
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        zerolinecolor=RULE,
        zerolinewidth=1,
        showspikes=False,
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
        subplot_titles=list(ch_names),
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.07,
    )
    keys = list(means.keys())
    ymax = 0.0
    for key in means:
        ymax = max(ymax, float(np.percentile(np.abs(means[key]), 99)))
    ymax = max(ymax * 1.25, 1.0)
    for i, name in enumerate(ch_names):
        r, c = divmod(i, cols)
        _add_window(fig, window, r + 1, c + 1)
        for k, key in enumerate(keys):
            y = means[key][i]
            e = sems[key][i]
            color = _class_color(key, k)
            label = _class_label(key, class_names)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate([y + e, (y - e)[::-1]]),
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.22),
                    line=dict(width=0, color=hex_to_rgba(color, 0.0)),
                    name=label,
                    legendgroup=str(key),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=r + 1,
                col=c + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=y,
                    mode="lines",
                    name=label,
                    legendgroup=str(key),
                    showlegend=i == 0,
                    line=dict(color=color, width=1.7),
                    hovertemplate=(
                        "Time = %{x:.3f} s<br>z = %{y:.3f}<br>"
                        f"{label}<br>{name}<extra></extra>"
                    ),
                ),
                row=r + 1,
                col=c + 1,
            )
    apply_plotly_style(
        fig,
        hovermode="x unified",
        title="Class-mean waveforms ± SEM (ranked channels, baseline z)",
        height=max(300, 190 * rows),
        margin=dict(l=56, r=28, t=64, b=72),
        legend=_legend_below(rows),
    )
    fig.update_xaxes(title_text="Time (s)", row=rows, col=1)
    fig.update_yaxes(title_text="baseline z", row=1, col=1)
    fig.update_yaxes(range=[-ymax, ymax])
    _blank_unused(fig, n_ch, rows, cols)
    return fig


def heatmap(
    z: np.ndarray,
    x,
    y,
    title: str,
    xaxis: str,
    yaxis: str,
) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=ZABS_SCALE,
            colorbar=dict(title=dict(text="|z|"), outlinewidth=0),
            hovertemplate="%{x}<br>%{y}<br>%{z:.3f}<extra></extra>",
        )
    )
    apply_plotly_style(
        fig,
        hovermode="closest",
        title=title,
        height=400,
        margin=dict(l=72, r=64, t=56, b=52),
    )
    fig.update_xaxes(title_text=xaxis)
    fig.update_yaxes(title_text=yaxis)
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
    n_ch = len(ch_names)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(pair_labels),
        shared_xaxes=True,
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    vmax = float(np.nanpercentile(maps, 98))
    vmax = max(vmax, 1e-9)
    ticksize = 11 if n_ch <= 10 else (9 if n_ch <= 18 else 8)
    for i in range(n):
        r, c = divmod(i, cols)
        fig.add_trace(
            go.Heatmap(
                z=maps[i],
                x=times,
                y=list(ch_names),
                coloraxis="coloraxis",
                showscale=False,
                hovertemplate=(
                    "%{y}<br>Time = %{x:.3f} s<br>|z| = %{z:.2f}<extra>"
                    + pair_labels[i]
                    + "</extra>"
                ),
            ),
            row=r + 1,
            col=c + 1,
        )
        fig.update_yaxes(
            autorange="reversed",
            tickfont=dict(size=ticksize, color=MUTED),
            title_text="Channel" if c == 0 else "",
            row=r + 1,
            col=c + 1,
        )
    apply_plotly_style(
        fig,
        hovermode="closest",
        title=title,
        coloraxis=dict(
            colorscale=ZABS_SCALE,
            cmin=0,
            cmax=vmax,
            colorbar=dict(title=dict(text="|z|", side="right"), outlinewidth=0),
        ),
        height=max(340, (16 * n_ch + 90) * rows),
        margin=dict(l=88, r=72, t=70, b=52),
    )
    fig.update_xaxes(title_text="Time (s)", row=rows, col=1)
    _blank_unused(fig, n, rows, cols)
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
        subplot_titles=list(ch_names),
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.07,
    )
    keys = list(spec_means.keys())
    ymin, ymax, fmax_show = _spectrum_ylim(freqs, spec_means)
    fmin_show = float(freqs[0])
    for i, name in enumerate(ch_names):
        r, c = divmod(i, cols)
        _add_band_guides(fig, fmin_show, fmax_show, r + 1, c + 1, label=True)
        for k, key in enumerate(keys):
            y = spec_means[key][i]
            color = _class_color(key, k)
            label = _class_label(key, class_names)
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=y,
                    mode="lines",
                    name=label,
                    legendgroup=str(key),
                    showlegend=i == 0,
                    line=dict(color=color, width=1.7),
                    hovertemplate=(
                        "Frequency = %{x:.1f} Hz<br>Power = %{y:.3g}<br>"
                        f"{label}<br>{name}<extra></extra>"
                    ),
                ),
                row=r + 1,
                col=c + 1,
            )
    apply_plotly_style(
        fig,
        hovermode="x unified",
        title="Class-mean spectra in the analysis window (ranked channels)",
        height=max(300, 190 * rows),
        margin=dict(l=56, r=28, t=64, b=72),
        legend=_legend_below(rows),
    )
    fig.update_xaxes(range=[fmin_show, fmax_show])
    fig.update_xaxes(title_text="Frequency (Hz)", row=rows, col=1)
    fig.update_yaxes(type="log", range=[float(np.log10(ymin)), float(np.log10(ymax))])
    fig.update_yaxes(title_text="Power", row=1, col=1)
    _blank_unused(fig, n_ch, rows, cols)
    return fig


def band_topomaps(
    xy: np.ndarray,
    band_means: np.ndarray,
    band_names: list[str],
    ch_names: list[str] | None = None,
    *,
    selected: int | None = None,
) -> go.Figure:
    from epochlens.topo import interpolate_topo, located_mask

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
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )
    coloraxes: dict[str, dict] = {}
    ok = located_mask(xy)
    names = _channel_hover_names(xy.shape[0], ch_names)
    idx = np.arange(xy.shape[0])
    shapes: list[dict] = []
    ranges = head_axis_ranges(xy)
    sel = None if selected is None else int(selected)
    for i, name in enumerate(band_names):
        r, c = divmod(i, cols)
        Xi, Yi, Zi = interpolate_topo(xy, band_means[:, i])
        vmax = max(float(np.nanpercentile(np.abs(Zi), 98)), 1e-9)
        axis_i = r * cols + c + 1
        coloraxis = "coloraxis" if axis_i == 1 else f"coloraxis{axis_i}"
        xref, yref = _xy_ref(r + 1, c + 1, cols)
        fig.add_trace(
            go.Heatmap(
                z=Zi,
                x=Xi[0],
                y=Yi[:, 0],
                coloraxis=coloraxis,
                showscale=True,
                hoverongaps=False,
                hovertemplate="x = %{x:.2f}<br>y = %{y:.2f}<br>power = %{z:.3g}<extra>"
                + name
                + "</extra>",
            ),
            row=r + 1,
            col=c + 1,
        )
        ok_idx = idx[ok]
        sensor_custom = np.column_stack([ok_idx, band_means[ok, i]])
        fig.add_trace(
            go.Scatter(
                x=xy[ok, 0],
                y=xy[ok, 1],
                mode="markers",
                name="sensors",
                marker=dict(size=8, color=PANEL, line=dict(width=0.7, color=INK)),
                showlegend=False,
                hovertext=[names[j] for j in ok_idx],
                customdata=sensor_custom,
                hovertemplate="%{hovertext}<br>power = %{customdata[1]:.3g}<extra>"
                + name
                + "</extra>",
            ),
            row=r + 1,
            col=c + 1,
        )
        if sel is not None and 0 <= sel < xy.shape[0] and ok[sel]:
            power_sel = float(band_means[sel, i])
            fig.add_trace(
                go.Scatter(
                    x=[xy[sel, 0]],
                    y=[xy[sel, 1]],
                    mode="markers",
                    name="focus",
                    marker=dict(
                        size=18,
                        symbol="circle-open",
                        color=INK,
                        line=dict(width=2, color=INK),
                    ),
                    showlegend=False,
                    hovertext=[names[sel]],
                    customdata=[[sel, power_sel]],
                    hovertemplate="%{hovertext}<br>power = %{customdata[1]:.3g}<extra>"
                    + name
                    + "</extra>",
                ),
                row=r + 1,
                col=c + 1,
            )
        _hide_xy(fig, r + 1, c + 1, xref)
        if ranges is not None:
            fig.update_xaxes(range=list(ranges[0]), row=r + 1, col=c + 1)
            fig.update_yaxes(range=list(ranges[1]), row=r + 1, col=c + 1)
        shapes.extend(head_plotly_shapes(xy, xref=xref, yref=yref))
        coloraxes[coloraxis] = dict(
            colorscale=POWER_SCALE,
            cmin=0,
            cmax=vmax,
            colorbar=dict(
                title=dict(text="", font=dict(size=10)),
                thickness=10,
                outlinewidth=0,
                tickfont=dict(size=9, color=MUTED),
                len=0.72 / rows,
                y=1.0 - (r + 0.5) / rows,
                yanchor="middle",
                x=min(0.98, (c + 1) / cols - 0.02),
                xanchor="left",
            ),
        )
    # Place colorbars against each subplot domain so they do not sit on titles.
    for i in range(n):
        r, c = divmod(i, cols)
        axis_i = r * cols + c + 1
        coloraxis = "coloraxis" if axis_i == 1 else f"coloraxis{axis_i}"
        xkey = "xaxis" if axis_i == 1 else f"xaxis{axis_i}"
        ykey = "yaxis" if axis_i == 1 else f"yaxis{axis_i}"
        xdom = getattr(fig.layout[xkey], "domain", None)
        ydom = getattr(fig.layout[ykey], "domain", None)
        if xdom is not None and ydom is not None:
            coloraxes[coloraxis]["colorbar"].update(
                x=float(xdom[1]) + 0.008,
                y=0.5 * (float(ydom[0]) + float(ydom[1])),
                len=max(0.18, float(ydom[1]) - float(ydom[0]) - 0.02),
                yanchor="middle",
                xanchor="left",
            )
    apply_plotly_style(
        fig,
        hovermode="closest",
        title="Band-power topography",
        height=max(360, 320 * rows),
        margin=dict(l=20, r=64, t=56, b=24),
        shapes=shapes,
        **coloraxes,
    )
    _blank_unused(fig, n, rows, cols)
    return fig
