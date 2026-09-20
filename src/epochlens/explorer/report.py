"""Single-file first-look HTML report. Matplotlib only; no Streamlit required."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from epochlens.types import EpochBatch
from epochlens.bands import band_power, window_spectrum
from epochlens.cwt import class_relative_scalograms, energy_channel_score, mean_cwt_power, relative_scalogram
from epochlens.decoding import logeuclid_lda_cv
from epochlens.discriminability import pairwise_maps
from epochlens.explorer.mathnotes import html_block
from epochlens.explorer.summary import facts_line
from epochlens.explorer.style import (
    CLASS_PALETTE,
    DPI,
    INK,
    MUTED,
    PICK,
    REPORT_CSS,
    WINDOW_FILL,
    apply_matplotlib_rc,
)
from epochlens.ranking import prepare_ranking, top_channels
from epochlens.riemann import embed_mds, pairwise_distances, session_whiten, trial_covariances
from epochlens.topo import can_draw_scalp, interpolate_topo, located_mask
from epochlens.waveforms import class_mean_sem

HONESTY = (
    "This report is a first look at epoched EEG: waveforms, time–frequency, "
    "scalp, and covariance geometry. It is not a classifier, and it is not a "
    "preprocessing pipeline."
)
DECODE_NOTE = (
    "Cross-validated LDA on log-Euclidean covariances is a sanity check against "
    "chance, not a brain–computer interface."
)


def _png(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.clf()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _class_color(key, index: int) -> str:
    if key is None:
        return CLASS_PALETTE[index % len(CLASS_PALETTE)]
    return CLASS_PALETTE[int(key) % len(CLASS_PALETTE)]


def _class_label(key, class_names: dict) -> str:
    if key is None:
        return "all"
    return class_names.get(int(key), str(key))


def _head_outline(ax, xy, *, fill=None) -> None:
    from matplotlib.patches import Circle

    xy = np.asarray(xy, dtype=np.float64)
    located = xy[located_mask(xy)]
    center = located.mean(axis=0)
    radius = float(np.max(np.linalg.norm(located - center, axis=1)) * 1.15)
    ax.add_patch(
        Circle(
            center,
            radius,
            facecolor="none" if fill is None else fill,
            edgecolor=INK,
            lw=1.15,
            zorder=5,
        )
    )
    nose_y = center[1] + radius
    w = 0.14 * radius
    h = 0.16 * radius
    ax.plot(
        [center[0] - w, center[0], center[0] + w],
        [nose_y - 0.02 * radius, nose_y + h, nose_y - 0.02 * radius],
        color=INK,
        lw=1.15,
        zorder=5,
        solid_capstyle="round",
    )
    ax.set_xlim(center[0] - 1.25 * radius, center[0] + 1.25 * radius)
    ax.set_ylim(center[1] - 1.2 * radius, center[1] + 1.45 * radius)


def _clip_topo(im, ax, xy) -> None:
    from matplotlib.patches import Circle

    xy = np.asarray(xy, dtype=np.float64)
    located = xy[located_mask(xy)]
    center = located.mean(axis=0)
    radius = float(np.max(np.linalg.norm(located - center, axis=1)) * 1.15)
    im.set_clip_path(Circle(center, radius, transform=ax.transData))


def _window_guides(ax, window: tuple[float, float]) -> None:
    ax.axvspan(window[0], window[1], color=WINDOW_FILL, alpha=0.7, lw=0, zorder=0)
    ax.axvline(window[0], color=MUTED, lw=0.7, ls="--", zorder=1)
    ax.axvline(window[1], color=MUTED, lw=0.7, ls="--", zorder=1)


def _index_for_times(times: np.ndarray, selected: np.ndarray) -> np.ndarray:
    times = np.asarray(times)
    selected = np.asarray(selected)
    idx = np.searchsorted(times, selected)
    return np.clip(idx, 0, times.size - 1)


def _scalogram_figure(rel, times, freqs, ch_names, window, max_channels: int = 8):
    import matplotlib.pyplot as plt

    n_ch = min(rel.shape[0], max_channels)
    cols = 4
    rows = int(np.ceil(n_ch / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(11.2, 2.45 * rows), squeeze=False, layout="constrained"
    )
    vmax = np.percentile(np.abs(rel[:n_ch]), 98)
    vmax = max(float(vmax), 1e-9)
    im = None
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n_ch:
            ax.axis("off")
            continue
        im = ax.pcolormesh(
            times, freqs, rel[i], shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        ax.axvline(window[0], color=INK, lw=0.8, ls="--")
        ax.axvline(window[1], color=INK, lw=0.8, ls="--")
        ax.set_title(ch_names[i], fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="relative power")
    return fig


def _stem_figure(scores, ch_names, picks=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.2, 3.1), layout="constrained")
    x = np.arange(1, len(scores) + 1)
    ax.bar(x, scores, color="#8A8175", width=0.8)
    if picks is not None:
        ax.bar(x[np.asarray(picks)], scores[np.asarray(picks)], color=PICK, width=0.8)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Score")
    return fig


def _mds_figure(xy, labels, sessions, class_names):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5.7, 5.3), layout="constrained")
    markers = ["o", "s", "D", "^"]
    sessions = np.ones(len(labels), dtype=int) if sessions is None else sessions
    uniq_cls = np.unique(labels)
    uniq_ses = np.unique(sessions)
    for cls in uniq_cls:
        color = _class_color(cls, int(cls))
        for ses in uniq_ses:
            m = (labels == cls) & (sessions == ses)
            if not np.any(m):
                continue
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                marker=markers[int(ses) % len(markers)],
                c=color,
                s=38,
                alpha=0.88,
                edgecolors="none",
                zorder=3,
            )
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_aspect("equal", adjustable="datalim")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_class_color(cls, int(cls)),
            markersize=7,
            label=_class_label(cls, class_names),
        )
        for cls in uniq_cls
    ]
    if uniq_ses.size >= 2:
        handles.extend(
            Line2D(
                [0],
                [0],
                marker=markers[i % len(markers)],
                color="none",
                markerfacecolor=INK,
                markersize=7,
                label=f"session {ses}",
            )
            for i, ses in enumerate(uniq_ses)
        )
    ax.legend(handles=handles, fontsize=7, loc="best", frameon=False)
    return fig


def _traces_figure(times, means, sems, class_names, ch_names, window):
    import matplotlib.pyplot as plt

    n_ch = len(ch_names)
    cols = 2 if n_ch > 1 else 1
    rows = int(np.ceil(n_ch / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(11.2, 2.25 * rows), squeeze=False, layout="constrained"
    )
    keys = list(means.keys())
    ymax = 0.0
    for key in means:
        ymax = max(ymax, float(np.percentile(np.abs(means[key]), 99)))
    ymax = max(ymax * 1.25, 1.0)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n_ch:
            ax.axis("off")
            continue
        _window_guides(ax, window)
        for k, key in enumerate(keys):
            y = means[key][i]
            e = sems[key][i]
            color = _class_color(key, k)
            label = _class_label(key, class_names)
            ax.plot(times, y, color=color, lw=1.35, label=label if i == 0 else None, zorder=2)
            ax.fill_between(times, y - e, y + e, color=color, alpha=0.28, linewidth=0, zorder=2)
        ax.set_title(ch_names[i], fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("baseline z")
        ax.set_ylim(-ymax, ymax)
        if i == 0:
            ax.legend(fontsize=7, loc="best", frameon=False)
    return fig


def _psd_figure(freqs, spec_means, class_names, ch_names):
    import matplotlib.pyplot as plt

    n_ch = len(ch_names)
    cols = 2 if n_ch > 1 else 1
    rows = int(np.ceil(n_ch / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(11.2, 2.15 * rows), squeeze=False, layout="constrained"
    )
    keys = list(spec_means.keys())
    fmax_show = min(float(freqs[-1]), 45.0)
    show = freqs <= fmax_show
    stacked = np.concatenate([spec_means[key][:, show] for key in keys], axis=0)
    positives = stacked[stacked > 0]
    ymin = float(np.percentile(positives, 2)) if positives.size else 1e-6
    ymax = float(np.max(stacked)) if stacked.size else 1.0
    ymin = max(ymin * 0.5, 1e-12)
    ymax = max(ymax * 1.25, ymin * 10.0)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n_ch:
            ax.axis("off")
            continue
        for k, key in enumerate(keys):
            color = _class_color(key, k)
            label = _class_label(key, class_names)
            ax.plot(freqs, spec_means[key][i], color=color, lw=1.35, label=label if i == 0 else None)
        ax.set_title(ch_names[i], fontsize=9)
        ax.set_xlabel("Hz")
        ax.set_ylabel("Power")
        ax.set_xlim(freqs[0], fmax_show)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        if i == 0:
            ax.legend(fontsize=7, loc="best", frameon=False)
    return fig


def _ranking_table(ch_names, scores, picks, bad, *, labeled: bool = False) -> str:
    pickset = {int(i) for i in np.asarray(picks, dtype=int)}
    vis = np.nan_to_num(np.asarray(scores, dtype=np.float64), neginf=np.nan)
    order = np.argsort(np.nan_to_num(vis, nan=-np.inf))[::-1]
    rows = [
        "<table class=\"rank\"><thead><tr>"
        "<th>Rank</th><th>Channel</th><th>Score</th><th>visualization subset</th>"
        "</tr></thead><tbody>"
    ]
    rank = 0
    shown = 0
    n_show = 24
    for idx in order:
        if bool(bad[int(idx)]) or not np.isfinite(vis[int(idx)]):
            continue
        rank += 1
        if shown >= n_show:
            continue
        shown += 1
        flag = "yes" if int(idx) in pickset else ""
        rows.append(
            f"<tr><td class=\"num\">{rank}</td><td>{ch_names[int(idx)]}</td>"
            f"<td class=\"num\">{vis[int(idx)]:.3f}</td><td>{flag}</td></tr>"
        )
    rows.append("</tbody></table>")
    n_bad = int(np.sum(bad))
    if n_bad:
        names = ", ".join(ch_names[i] for i in np.flatnonzero(bad))
        rows.append(f"<p class=\"sub\">Excluded bad channels: {names}</p>")
    n_finite = int(np.sum(np.isfinite(vis) & ~np.asarray(bad, dtype=bool)))
    note = "Waveforms, CWT, and spectra use the top-k visualization subset."
    if labeled:
        note += " MDS and the chance check use the full montage (all channels)."
    if n_finite > n_show:
        note = f"Showing top {n_show} of {n_finite} channels. " + note
    rows.append(f"<p class=\"sub\">{note}</p>")
    return "\n".join(rows)


def _folds_figure(report):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 2.7), layout="constrained")
    x = np.arange(1, report.fold_scores.size + 1)
    ax.bar(x, 100.0 * report.fold_scores, color="#6B7C8A", width=0.68)
    ax.axhline(
        100.0 * report.chance,
        color=PICK,
        ls="--",
        lw=1.15,
        label=f"chance {100 * report.chance:.0f}%",
    )
    ax.set_ylim(0, max(100.0, 100.0 * report.fold_scores.max() + 8))
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    return fig


def _bin_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.array([0.0, 1.0])
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    mid = 0.5 * (x[:-1] + x[1:])
    first = x[0] - 0.5 * (x[1] - x[0])
    last = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.concatenate([[first], mid, [last]])


def _pairwise_maps_figure(maps, times, ch_names, pair_labels):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    n = len(pair_labels)
    cols = min(3, max(n, 1))
    rows = int(np.ceil(n / cols))
    n_ch = len(ch_names)
    height = max(3.4, 0.34 * n_ch + 1.8)
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.9 * cols, height * rows), squeeze=False, layout="constrained"
    )
    vmax = float(np.nanpercentile(maps, 98))
    vmax = max(vmax, 1e-9)
    xedges = _bin_edges(np.asarray(times))
    yedges = np.arange(n_ch + 1) - 0.5
    yticks = np.arange(n_ch)
    im = None
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis("off")
            continue
        im = ax.pcolormesh(
            xedges,
            yedges,
            maps[i],
            cmap="magma",
            vmin=0,
            vmax=vmax,
            shading="auto",
        )
        ax.set_ylim(n_ch - 0.5, -0.5)
        ax.set_title(pair_labels[i], fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_yticks(yticks)
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.yaxis.set_minor_locator(NullLocator())
        if i % cols == 0:
            ax.set_ylabel("Channel")
            ax.set_yticklabels(list(ch_names), fontsize=7)
        else:
            ax.set_yticklabels([])
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="|z|")
    return fig


def _topo_figure(xy, values, highlight=None, vmin=None, vmax=None):
    import matplotlib.pyplot as plt

    Xi, Yi, Zi = interpolate_topo(xy, values)
    fig, ax = plt.subplots(figsize=(4.9, 4.9), layout="constrained")
    if vmax is None:
        vmax = np.nanpercentile(np.abs(Zi), 98)
    vmax = max(float(vmax), 1e-9)
    if vmin is None:
        vmin = -vmax
    im = ax.pcolormesh(Xi, Yi, Zi, shading="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax, zorder=2)
    _clip_topo(im, ax, xy)
    ok = located_mask(xy)
    ax.scatter(xy[ok, 0], xy[ok, 1], c=INK, s=10, zorder=3)
    if highlight is not None:
        h = np.asarray(highlight, dtype=int)
        h = h[ok[h]]
        ax.scatter(
            xy[h, 0],
            xy[h, 1],
            facecolors="none",
            edgecolors=INK,
            s=64,
            zorder=4,
        )
    _head_outline(ax, xy)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _band_topo_figure(xy, band_means, band_names):
    import matplotlib.pyplot as plt

    n = len(band_names)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.25 * cols, 3.15 * rows), squeeze=False, layout="constrained"
    )
    ok = located_mask(xy)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis("off")
            continue
        Xi, Yi, Zi = interpolate_topo(xy, band_means[:, i])
        vmax = max(float(np.nanpercentile(np.abs(Zi), 98)), 1e-9)
        im = ax.pcolormesh(Xi, Yi, Zi, shading="auto", cmap="inferno", vmin=0, vmax=vmax, zorder=2)
        _clip_topo(im, ax, xy)
        ax.scatter(xy[ok, 0], xy[ok, 1], c="#f3eee4", s=7, zorder=6)
        _head_outline(ax, xy)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(band_names[i], fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _class_scalogram_figure(rel_by_class, times, freqs, ch_names, class_names, window):
    import matplotlib.pyplot as plt

    keys = list(rel_by_class.keys())
    n_ch = len(ch_names)
    n_cls = len(keys)
    fig, axes = plt.subplots(
        n_ch, n_cls, figsize=(2.7 * n_cls, 1.85 * n_ch), squeeze=False, layout="constrained"
    )
    stacked = np.concatenate([rel_by_class[k] for k in keys], axis=0)
    vmax = np.percentile(np.abs(stacked), 98)
    vmax = max(float(vmax), 1e-9)
    im = None
    for r in range(n_ch):
        for c, key in enumerate(keys):
            ax = axes[r][c]
            im = ax.pcolormesh(
                times,
                freqs,
                rel_by_class[key][r],
                shading="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.axvline(window[0], color=INK, lw=0.6, ls="--")
            ax.axvline(window[1], color=INK, lw=0.6, ls="--")
            if r == 0:
                ax.set_title(_class_label(key, class_names), fontsize=9)
            if c == 0:
                ax.set_ylabel(ch_names[r], fontsize=8)
            if r == n_ch - 1:
                ax.set_xlabel("s", fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="relative power")
    return fig


def _figure_block(title: str, b64: str, caption: str, math_key: str | None = None) -> str:
    math = html_block(math_key) if math_key else ""
    return (
        f"<figure><h2>{title}</h2>"
        f"<img alt=\"{title}\" src=\"data:image/png;base64,{b64}\"/>"
        f"<figcaption>{caption}</figcaption>{math}</figure>"
    )


def render_report(
    batch: EpochBatch,
    *,
    window: tuple[float, float],
    baseline: tuple[float, float],
    fmin: float = 4.0,
    fmax: float = 40.0,
    voices_per_octave: int = 12,
    decim: int = 2,
    top_k: int = 8,
    sidecar: Path | None = None,
    full: bool = False,
) -> str:
    """Return a self-contained first-look HTML document.

    Default path is the fast explorer. ``full=True`` adds per-class CWT,
    a CWT energy stem, and discriminability topography.
    """
    import matplotlib

    matplotlib.use("Agg")
    apply_matplotlib_rc()

    zbatch, scores, picks, ave, disc_times, pairs, bad = prepare_ranking(
        batch, window, baseline, top_k
    )
    k = max(int(picks.size), 1)
    subset = batch.pick(picks) if picks.size else batch
    zsub = zbatch.pick(picks) if picks.size else zbatch
    means, sems = class_mean_sem(zsub)
    mean_power, freqs, cwt_times = mean_cwt_power(
        subset,
        fmin=fmin,
        fmax=fmax,
        voices_per_octave=voices_per_octave,
        decim=decim,
        use_cache=True,
    )
    rel = relative_scalogram(mean_power, cwt_times, baseline)
    energy = energy_channel_score(rel, cwt_times, window) if full else None
    grand_bands = None
    band_names: list[str] = []
    draw_scalp = can_draw_scalp(batch.montage_xy)
    if draw_scalp:
        power, band_names = band_power(batch, window)
        grand_bands = power.mean(axis=0)

    sections: list[tuple[str, str, str, str | None]] = []
    sections.append(
        (
            "Class-mean waveforms (ranked channels, baseline z-scored)",
            _png(_traces_figure(zsub.times, means, sems, batch.class_names, zsub.ch_names, window)),
            "Look for class separation in the shaded analysis window. Traces are baseline z-scored means ± SEM.",
            "waveforms",
        )
    )
    spec, spec_freqs = window_spectrum(subset, window)
    spec_means: dict[int | None, np.ndarray] = {}
    if batch.labels is None:
        spec_means[None] = spec.mean(axis=0)
    else:
        for cls in np.unique(batch.labels):
            spec_means[int(cls)] = spec[batch.labels == cls].mean(axis=0)
    sections.append(
        (
            "Class-mean spectra in the analysis window",
            _png(_psd_figure(spec_freqs, spec_means, batch.class_names, subset.ch_names)),
            "Log power in the analysis window. Shared log scale keeps quiet channels from looking structured.",
            "spectra",
        )
    )
    sections.append(
        (
            "Relative CWT scalograms (ranked channels)",
            _png(_scalogram_figure(rel, cwt_times, freqs, subset.ch_names, window, max_channels=k)),
            "Relative power versus the baseline window (diverging scale). Dashed lines mark the analysis window.",
            "cwt",
        )
    )
    if full and batch.labels is not None and picks.size:
        show_n = min(4, subset.n_channels)
        rel_by_class, ct, cf, class_ch_names = class_relative_scalograms(
            subset,
            baseline,
            show_n=show_n,
            fmin=fmin,
            fmax=fmax,
            voices_per_octave=voices_per_octave,
            decim=decim,
            use_cache=True,
        )
        sections.append(
            (
                "Per-class relative CWT",
                _png(
                    _class_scalogram_figure(
                        rel_by_class,
                        ct,
                        cf,
                        class_ch_names,
                        batch.class_names,
                        window,
                    )
                ),
                "Per-class relative power versus the baseline window. "
                "Each column is a class. Look for time–frequency structure that is not shared across columns.",
                "cwt",
            )
        )
    if full and energy is not None:
        sections.append(
            (
                "CWT energy score on ranked channels",
                _png(_stem_figure(energy, subset.ch_names)),
                "Per-channel energy of the relative scalogram in the analysis window.",
                "cwt_energy",
            )
        )
    if grand_bands is not None and draw_scalp:
        sections.append(
            (
                "Band-power topography",
                _png(_band_topo_figure(batch.montage_xy, grand_bands, band_names)),
                "Mean band power on the scalp. Each map is scaled independently so spatial structure stays visible; nose at the top.",
                "scalp",
            )
        )

    chance = None
    if batch.labels is not None and ave is not None and disc_times is not None:
        pair_txt = ", ".join(
            f"{_class_label(a, batch.class_names)} vs {_class_label(b, batch.class_names)}"
            for a, b in pairs
        )
        show = top_channels(scores, min(24, scores.size))
        t_idx = _index_for_times(zbatch.times, disc_times)
        pair_maps, _ = pairwise_maps(zbatch.data[:, :, t_idx], batch.labels, method="wilcoxon")
        pair_labels = [
            f"{_class_label(a, batch.class_names)} vs {_class_label(b, batch.class_names)}"
            for a, b in pairs
        ]
        sections.append(
            (
                f"Pairwise discriminability maps ({pair_txt})",
                _png(_pairwise_maps_figure(pair_maps[:, show, :], disc_times, [batch.ch_names[i] for i in show], pair_labels)),
                "One panel per class pair: Mann–Whitney U converted to |z| (README shorthand Wilcoxon). "
                "Time is on the x-axis. Ranking uses the mean across pairs.",
                "disc",
            )
        )
        sections.append(
            (
                "Discriminability channel score",
                _png(
                    _stem_figure(
                        np.nan_to_num(scores, neginf=0.0),
                        batch.ch_names,
                        picks=picks,
                    )
                ),
                "Mean pairwise |z| used to pick the visualization subset for waveforms / CWT / spectra (highlighted). MDS uses the full montage.",
                "ranking",
            )
        )
        if full and draw_scalp:
            vis_scores = np.nan_to_num(scores, neginf=0.0)
            sections.append(
                (
                    "Discriminability topography",
                    _png(_topo_figure(batch.montage_xy, vis_scores, highlight=picks)),
                    "Same discriminability score on the scalp. Circled sensors are in the visualization subset.",
                    "disc",
                )
            )

        covs = trial_covariances(batch, window)
        xy0 = embed_mds(pairwise_distances(covs, metric="logeuclid"))
        sections.append(
            (
                "Log-Euclidean MDS of trial covariances",
                _png(_mds_figure(xy0, batch.labels, batch.sessions, batch.class_names)),
                "Full-montage trial covariance embeddings (all channels). Color is class; marker is session. Look for class clusters versus session grouping.",
                "mds",
            )
        )
        n_ses = 0 if batch.sessions is None else int(np.unique(batch.sessions).size)
        if n_ses >= 2:
            xy1 = embed_mds(
                pairwise_distances(session_whiten(covs, batch.sessions, metric="logeuclid"), metric="logeuclid")
            )
            sections.append(
                (
                    "Session-whitened MDS",
                    _png(_mds_figure(xy1, batch.labels, batch.sessions, batch.class_names)),
                    "Same full-montage embedding after per-session whitening. Session structure should recede if class geometry remains.",
                    "mds",
                )
            )
        try:
            chance = logeuclid_lda_cv(batch, window)
            acc = f"{100 * chance.accuracy:.1f}%"
            ch = f"{100 * chance.chance:.0f}%"
            maj = f"{100 * chance.majority:.0f}%"
            sections.append(
                (
                    "Sanity check versus chance",
                    _png(_folds_figure(chance)),
                    f"{chance.n_classes}-class {chance.method} on the full montage (all channels) is {acc} "
                    f"(chance {ch}; majority {maj}). Ranked-channel selection is not used. "
                    "A sanity check versus chance, not a classifier and not a BCI.",
                    "chance",
                )
            )
        except ValueError:
            chance = None

    n_bad = int(np.sum(bad))
    bits = [facts_line(batch)]
    bits.append(f"analysis window {window[0]:g}–{window[1]:g} s")
    if n_bad:
        bits.append(f"excluded {n_bad} bad channel{'s' if n_bad != 1 else ''}")
    meta = " · ".join(bits)
    banner = HONESTY
    footer = "EpochLens — first-look figures for labeled EEG (MIT)."
    if chance is not None:
        banner = f"{banner} {DECODE_NOTE}"
    rank_html = _ranking_table(
        batch.ch_names, scores, picks, bad, labeled=batch.labels is not None
    )
    rank_html += html_block("ranking")
    if sidecar is not None:
        payload = {
            "dataset": batch.dataset,
            "subject_id": batch.subject_id,
            "condition": batch.condition,
            "class_names": {str(k): v for k, v in batch.class_names.items()},
            "n_trials": batch.n_trials,
            "n_channels": batch.n_channels,
            "sfreq": batch.sfreq,
            "window": [window[0], window[1]],
            "baseline": [baseline[0], baseline[1]],
            "bad_channels": [batch.ch_names[i] for i in np.flatnonzero(bad)],
            "ranking": [
                {
                    "rank": i + 1,
                    "index": int(idx),
                    "name": batch.ch_names[int(idx)],
                    "score": float(np.nan_to_num(scores[int(idx)], neginf=0.0)),
                }
                for i, idx in enumerate(np.asarray(picks, dtype=int))
            ],
        }
        if chance is not None:
            payload["chance"] = {
                "accuracy": chance.accuracy,
                "chance": chance.chance,
                "majority": chance.majority,
                "n_classes": chance.n_classes,
                "n_splits": chance.n_splits,
                "method": chance.method,
            }
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    body = "\n".join(_figure_block(title, b64, caption, key) for title, b64, caption, key in sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>EpochLens — {batch.dataset or "report"}</title>
  <style>
{REPORT_CSS}
  </style>
</head>
<body>
  <h1>EpochLens</h1>
  <p class="lede">First-look figures for labeled EEG epochs.</p>
  <div class="banner">{banner}</div>
  <p class="meta">{meta}</p>
  <h2>Ranked channels</h2>
  {rank_html}
  {body}
  <footer>{footer}</footer>
</body>
</html>
"""


def write_report(html: str, path: Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
