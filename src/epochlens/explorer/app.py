"""Streamlit explorer. Any labeled epochs (FIF / NPZ / dummy data)."""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from epochlens.adapters.synthetic import make_synthetic
from epochlens.bands import band_power, band_range_caption, class_mean_band_power, window_spectrum
from epochlens.cwt import class_relative_scalograms, mean_cwt_power, relative_scalogram
from epochlens.decoding import ChanceReport, logeuclid_lda_cv, logeuclid_mdm_cv
from epochlens.discriminability import peak_map
from epochlens.explorer.mathnotes import show_math
from epochlens.explorer.plots import (
    band_topomaps,
    channel_stem,
    class_mean_spectra,
    class_scalogram_grid,
    mds_scatter,
    mean_traces,
    pairwise_heatmaps,
    scalogram_grid,
    scalp_scatter,
    trial_strip,
)
from epochlens.explorer.report import DECODE_NOTE, HONESTY, render_report, report_pngs
from epochlens.explorer.style import CLASS_PALETTE, MUTED, PLOTLY_CONFIG, apply_plotly_style
from epochlens.ranking import prepare_ranking, ranking_table, session_channel_votes, top_channels
from epochlens.riemann import embed_mds, mds_stress, pairwise_distances, session_whiten, trial_covariances
from epochlens.explorer.summary import dataset_facts
from epochlens.topo import can_draw_scalp
from epochlens.waveforms import class_mean_sem
from epochlens.windows import default_windows

_COV_ESTIMATORS = ("lwf", "oas", "scm")
_COV_EST_LABEL = {"lwf": "Ledoit–Wolf", "oas": "OAS", "scm": "sample covariance"}
_PLOTLY_ON_SELECT = "rerun"


st.set_page_config(page_title="EpochLens", layout="wide")

_SOURCE_SYNTH = "Dummy data"
_SOURCE_FIF = "MNE epochs FIF"
_SOURCE_NPZ = "NPZ epochs"


@st.cache_data(show_spinner=True)
def _load_batch(source: str, fif: str, npz: str):
    if source == _SOURCE_SYNTH:
        return make_synthetic()
    if source == _SOURCE_FIF:
        from epochlens.adapters.fif import load_epochs_fif

        return load_epochs_fif(fif)
    from epochlens.adapters.npz import load_epochs

    return load_epochs(npz)


def _class_spectra(subset, window: tuple[float, float]) -> tuple[np.ndarray, dict]:
    spec, freqs = window_spectrum(subset, window)
    means: dict[int | None, np.ndarray] = {}
    if subset.labels is None:
        means[None] = spec.mean(axis=0)
    else:
        for cls in np.unique(subset.labels):
            means[int(cls)] = spec[subset.labels == cls].mean(axis=0)
    return freqs, means


def _class_band_rows(subset, window: tuple[float, float], class_names: dict) -> list[dict]:
    power, names = band_power(subset, window)
    rows: list[dict] = []
    if subset.labels is None:
        grand = power.mean(axis=0)
        for ci, ch in enumerate(subset.ch_names):
            row = {"channel": ch}
            for bi, bname in enumerate(names):
                row[bname] = float(grand[ci, bi])
            rows.append(row)
        return rows
    means, classes = class_mean_band_power(power, subset.labels)
    for ci, ch in enumerate(subset.ch_names):
        for ki, cls in enumerate(classes):
            row = {
                "channel": ch,
                "class": class_names.get(int(cls), str(int(cls))),
            }
            for bi, bname in enumerate(names):
                row[bname] = float(means[ki, ci, bi])
            rows.append(row)
    return rows


def _disc_curve_figure(
    times: np.ndarray,
    curve: np.ndarray,
    peak_t: float,
    peak_y: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=curve,
            mode="lines",
            name="mean |z|",
            line=dict(color=CLASS_PALETTE[0], width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[peak_t],
            y=[peak_y],
            mode="markers",
            name="peak",
            marker=dict(color=CLASS_PALETTE[1], size=11),
        )
    )
    apply_plotly_style(
        fig,
        hovermode="x unified",
        title="Mean |z| over time (shown channels)",
        height=280,
        margin=dict(l=56, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="mean |z|")
    return fig


def _cv_fold_figure(lda: ChanceReport, mdm: ChanceReport) -> go.Figure:
    folds = [f"fold {i + 1}" for i in range(lda.n_splits)]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=folds,
            y=lda.fold_scores,
            name="LDA",
            marker_color=CLASS_PALETTE[0],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=folds,
            y=mdm.fold_scores,
            name="MDM",
            mode="markers+lines",
            marker=dict(color=CLASS_PALETTE[1], size=9),
            line=dict(color=CLASS_PALETTE[1], width=1.6),
        )
    )
    fig.add_hline(y=lda.chance, line_dash="dash", line_color=MUTED, annotation_text="chance")
    fig.add_hline(y=lda.majority, line_dash="dot", line_color=CLASS_PALETTE[4], annotation_text="majority")
    apply_plotly_style(
        fig,
        hovermode="x unified",
        title="Stratified CV fold accuracy vs chance",
        height=320,
        margin=dict(l=56, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.28,
    )
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05])
    return fig


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _default_source() -> str:
    if os.environ.get("EPOCHLENS_NPZ", "").strip():
        return _SOURCE_NPZ
    if os.environ.get("EPOCHLENS_FIF", "").strip():
        return _SOURCE_FIF
    return _SOURCE_SYNTH


def _save_upload(upload, suffix: str) -> str:
    folder = Path(st.session_state.setdefault("_epochlens_uploads", tempfile.mkdtemp(prefix="epochlens_")))
    dest = folder / f"epochs{suffix}"
    dest.write_bytes(upload.getbuffer())
    return str(dest)


def _channel_from_selection(event) -> int | None:
    """Read a channel index from a Streamlit Plotly selection ``customdata``."""
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return None
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None
    pt = points[0]
    if isinstance(pt, dict):
        cd = pt.get("customdata")
    else:
        cd = getattr(pt, "customdata", None)
    if cd is None:
        return None
    if isinstance(cd, (list, tuple, np.ndarray)):
        cd = cd[0]
    try:
        return int(cd)
    except (TypeError, ValueError):
        return None


def _apply_focus_from_chart(event, current_focus: int) -> None:
    ch = _channel_from_selection(event)
    if ch is None:
        return
    if int(ch) != int(current_focus):
        st.session_state["focus_override"] = int(ch)
        st.rerun()


def _plotly_chart(fig, *, key: str | None = None, on_select: str | None = None):
    kwargs: dict = {"width": "stretch", "config": PLOTLY_CONFIG}
    if key is not None:
        kwargs["key"] = key
    if on_select is not None:
        kwargs["on_select"] = on_select
    try:
        return st.plotly_chart(fig, **kwargs)
    except TypeError:
        kwargs.pop("on_select", None)
        kwargs.pop("config", None)
        try:
            return st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)
        except TypeError:
            return st.plotly_chart(fig, width="stretch", key=key)


def render() -> None:
    st.title("EpochLens")
    st.caption("Look at labeled EEG epochs. Classes come from the recording. HTML export is a snapshot, not the app.")

    with st.sidebar:
        st.header("Data")
        choices = [_SOURCE_SYNTH, _SOURCE_FIF, _SOURCE_NPZ]
        source = st.selectbox("Source", choices, index=choices.index(_default_source()))
        fif = os.environ.get("EPOCHLENS_FIF", "").strip()
        npz = os.environ.get("EPOCHLENS_NPZ", "").strip()
        if source == _SOURCE_FIF:
            up = st.file_uploader("Upload *-epo.fif", type=["fif"])
            fif = st.text_input(
                "or path on disk",
                value=fif,
                help="Any MNE *-epo.fif. Class names come from event_id.",
            )
            if up is not None:
                fif = _save_upload(up, "-epo.fif")
        elif source == _SOURCE_NPZ:
            up = st.file_uploader("Upload epochs NPZ", type=["npz"])
            npz = st.text_input("or path on disk", value=npz, help="EpochLens NPZ from any experiment.")
            if up is not None:
                npz = _save_upload(up, ".npz")

    if source == _SOURCE_FIF and not fif.strip():
        st.info("Upload a `*-epo.fif` or enter a path.")
        st.stop()
    if source == _SOURCE_NPZ and not npz.strip():
        st.info("Upload an epochs NPZ or enter a path.")
        st.stop()

    try:
        batch = _load_batch(source, fif, npz)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    tmin = float(batch.tmin)
    tmax = float(batch.times[-1])
    if tmax <= tmin:
        tmax = tmin + 1.0
    win0, base0 = default_windows(batch.dataset, tmin, tmax)
    span = max(tmax - tmin, 1e-6)
    step = min(0.05, span / 40.0)
    key = f"{batch.dataset}-{batch.n_trials}-{batch.n_times}-{tmin:.4f}-{tmax:.4f}"

    with st.sidebar:
        st.header("Windows (seconds)")
        t_pre = st.slider(
            "Baseline end",
            tmin,
            tmax,
            _clamp(float(base0[1]), tmin, tmax),
            step,
            key=f"baseline-{key}",
        )
        t_win0 = st.slider(
            "Analysis start",
            tmin,
            tmax,
            _clamp(float(win0[0]), tmin, tmax),
            step,
            key=f"win0-{key}",
        )
        t_win1 = st.slider(
            "Analysis end",
            tmin,
            tmax,
            _clamp(float(win0[1]), tmin, tmax),
            step,
            key=f"win1-{key}",
        )
        st.header("CWT")
        fmin = st.number_input("fmin (Hz)", 1.0, 40.0, 4.0)
        fmax = st.number_input("fmax (Hz)", 8.0, 80.0, 40.0)
        voices = st.number_input("Voices / octave", 4, 40, 12)
        decim = st.number_input("Time decim", 1, 8, 2)
        top_k = st.slider("Top channels", 3, 32, 8)

    st.caption(HONESTY)

    facts = dataset_facts(batch)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Epoch duration", facts["duration"])
    m2.metric("Channels", facts["channels"])
    m3.metric("Trials", facts["trials"])
    m4.metric("Sampling", facts["sfreq"])
    m5.metric("Classes", facts["classes"])
    detail = [facts["span"], f"{facts['samples']} samples"]
    if facts["class_detail"]:
        detail.append(facts["class_detail"])
    if facts["sessions"]:
        detail.append(f"{facts['sessions']} sessions")
    if batch.dataset:
        detail.append(str(batch.dataset))
    if batch.subject_id:
        detail.append(f"subject {batch.subject_id}")
    if facts["montage"] == "yes":
        detail.append("montage")
    st.caption(" · ".join(detail))

    window = (float(t_win0), float(t_win1))
    baseline = (float(batch.tmin), float(t_pre))
    k = min(int(top_k), batch.n_channels)
    zbatch, scores, picks, ave, disc_times, pairs, bad, pair_maps = prepare_ranking(
        batch, window, baseline, k
    )
    subset = batch.pick(picks)
    zsub = zbatch.pick(picks)

    focus_state = f"focus-ch-{key}"
    widget_key = f"focus-box-{key}"
    if "focus_override" in st.session_state:
        override = int(st.session_state.pop("focus_override"))
        st.session_state[widget_key] = override
        st.session_state[focus_state] = override
    elif widget_key in st.session_state:
        st.session_state[focus_state] = int(st.session_state[widget_key])
    elif focus_state not in st.session_state:
        st.session_state[focus_state] = int(picks[0]) if picks.size else 0
    focus = int(st.session_state[focus_state])
    focus = max(0, min(focus, batch.n_channels - 1))
    st.session_state[widget_key] = focus

    with st.sidebar:
        st.header("Focus")
        focus = st.selectbox(
            "Channel",
            options=list(range(batch.n_channels)),
            format_func=lambda i: batch.ch_names[int(i)],
            key=widget_key,
        )
        focus = int(focus)
        st.session_state[focus_state] = focus
        if bool(bad[focus]):
            st.caption(f"{batch.ch_names[focus]} is flagged as a bad channel.")

    view = st.radio(
        "View",
        [
            "Overview",
            "Waveforms",
            "Time–frequency",
            "Scalp",
            "Discriminability",
            "Ranking",
            "MDS",
            "vs chance",
        ],
        horizontal=True,
    )

    if view == "Overview":
        bits: list[str] = []
        if facts["class_detail"]:
            bits.append(facts["class_detail"])
        else:
            bits.append("No class labels on this batch")
        if int(np.sum(bad)):
            names = ", ".join(batch.ch_names[i] for i in np.flatnonzero(bad))
            bits.append(f"Bad channels: {names}")
        else:
            bits.append("No bad channels")
        if ave is not None and disc_times is not None:
            ave_peak = np.array(ave, copy=True)
            ave_peak[bad] = np.nan
            peak_t, peak_ch, _ = peak_map(ave_peak, disc_times)
            bits.append(f"Peak mean |z| at {peak_t:.3f} s on {batch.ch_names[peak_ch]}")
        st.write(". ".join(bits) + ".")
        st.write(
            "Visualization subset: "
            + (", ".join(batch.ch_names[i] for i in picks) if picks.size else "(none)")
            + "."
        )

        if batch.labels is None:
            st.write("No class labels, so MDS and the chance check are skipped.")
        else:
            ov_key = (
                f"overview-{batch.dataset}-{batch.n_trials}-{batch.n_times}"
                f"-{window[0]:.4f}-{window[1]:.4f}"
            )
            cached = st.session_state.get(ov_key)
            if cached is None:
                with st.spinner("Computing overview MDS and chance check…"):
                    covs = trial_covariances(batch, window)
                    dist = pairwise_distances(covs, metric="logeuclid")
                    xy = embed_mds(dist)
                    stress = float(mds_stress(dist, xy))
                    lda = None
                    mdm = None
                    cv_error = None
                    try:
                        lda = logeuclid_lda_cv(batch, window, covs=covs)
                        mdm = logeuclid_mdm_cv(batch, window, covs=covs)
                    except ValueError as exc:
                        cv_error = str(exc)
                    cached = {
                        "xy": xy,
                        "stress": stress,
                        "lda": lda,
                        "mdm": mdm,
                        "cv_error": cv_error,
                    }
                    st.session_state[ov_key] = cached
            _plotly_chart(
                mds_scatter(
                    cached["xy"],
                    batch.labels,
                    batch.sessions,
                    batch.class_names,
                    "Log-Euclidean MDS (full montage)",
                ),
                key=f"overview-mds-{ov_key}",
            )
            st.caption(f"Kruskal stress-1: {cached['stress']:.3f} (0 is exact).")
            if cached.get("cv_error"):
                st.error(cached["cv_error"])
            elif cached["lda"] is not None and cached["mdm"] is not None:
                lda: ChanceReport = cached["lda"]
                mdm: ChanceReport = cached["mdm"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("LDA CV accuracy", f"{100 * lda.accuracy:.1f}%")
                c2.metric("MDM CV accuracy", f"{100 * mdm.accuracy:.1f}%")
                c3.metric("Chance", f"{100 * lda.chance:.0f}%")
                c4.metric("Majority", f"{100 * lda.majority:.0f}%")
                _plotly_chart(_cv_fold_figure(lda, mdm), key=f"overview-cv-{ov_key}")
                st.caption(
                    f"LDA {100 * lda.accuracy:.1f}% · MDM {100 * mdm.accuracy:.1f}% · "
                    f"chance {100 * lda.chance:.0f}%. "
                    "Sanity check on the full montage, not a BCI."
                )

    elif view == "Waveforms":
        means, sems = class_mean_sem(zsub)
        _plotly_chart(
            mean_traces(zsub.times, means, sems, batch.class_names, zsub.ch_names, window),
            key=f"wave-means-{key}",
        )
        st.caption("Ranked channels. Baseline z-scored class means ± SEM; shaded region is the analysis window.")
        show_math(st, "waveforms")
        _plotly_chart(
            trial_strip(
                zbatch.times,
                zbatch.data[:, focus, :],
                batch.labels,
                batch.class_names,
                channel_name=batch.ch_names[focus],
                window=window,
            ),
            key=f"wave-strip-{key}-{focus}",
        )
        st.caption(
            f"Focused channel {batch.ch_names[focus]}: bold lines are class means; "
            "thin lines are individual trials (at most two per class)."
        )
        spec_freqs, spec_means = _class_spectra(subset, window)
        _plotly_chart(
            class_mean_spectra(spec_freqs, spec_means, batch.class_names, subset.ch_names),
            key=f"wave-spec-{key}",
        )
        st.caption(
            "Class-mean spectra in the analysis window (ranked channels). "
            "Log power; shared y-range includes the actual maximum. "
            f"Canonical bands: {band_range_caption()}."
        )
        focus_sub = batch.pick([focus])
        f_freqs, f_means = _class_spectra(focus_sub, window)
        _plotly_chart(
            class_mean_spectra(f_freqs, f_means, batch.class_names, focus_sub.ch_names),
            key=f"wave-spec-focus-{key}-{focus}",
        )
        st.caption(
            f"Class-mean spectrum for focused channel {batch.ch_names[focus]}. "
            f"Canonical bands: {band_range_caption()}."
        )
        st.dataframe(_class_band_rows(subset, window, batch.class_names), hide_index=True, width="stretch")
        st.caption("Class-mean band power in the analysis window (ranked channels).")
        show_math(st, "spectra")
        if int(np.sum(bad)):
            st.caption(f"Excluded {int(np.sum(bad))} bad channel(s) from ranking.")

    elif view == "Time–frequency":
        mean_power, freqs, times = mean_cwt_power(
            subset,
            fmin=float(fmin),
            fmax=float(fmax),
            voices_per_octave=int(voices),
            decim=int(decim),
        )
        rel = relative_scalogram(mean_power, times, baseline)
        _plotly_chart(
            scalogram_grid(
                rel,
                times,
                freqs,
                subset.ch_names,
                title="Relative CWT scalogram (baseline-normalized, ranked channels)",
                window=window,
                max_channels=k,
            ),
            key=f"cwt-ranked-{key}",
        )
        st.caption(
            "Relative power versus the baseline window (diverging RdBu), across trials / ranked channels. "
            "Dashed lines mark the analysis window."
        )
        if batch.labels is not None:
            show_n = min(4, subset.n_channels)
            rel_by_class, cls_times, cls_freqs, cls_names = class_relative_scalograms(
                subset,
                baseline,
                show_n=show_n,
                fmin=float(fmin),
                fmax=float(fmax),
                voices_per_octave=int(voices),
                decim=int(decim),
                use_cache=True,
            )
            _plotly_chart(
                class_scalogram_grid(
                    rel_by_class,
                    cls_times,
                    cls_freqs,
                    cls_names,
                    batch.class_names,
                    window,
                ),
                key=f"cwt-class-{key}",
            )
            st.caption(
                "Per-class relative power versus the baseline window. "
                "Each column is a class. Look for time–frequency structure that is not shared across columns."
            )
        show_math(st, "cwt")

    elif view == "Scalp":
        if can_draw_scalp(batch.montage_xy):
            power, names = band_power(batch, window)
            grand = power.mean(axis=0)
            event = _plotly_chart(
                band_topomaps(
                    batch.montage_xy,
                    grand,
                    names,
                    ch_names=batch.ch_names,
                    selected=focus,
                ),
                key=f"scalp-grand-{key}",
                on_select=_PLOTLY_ON_SELECT,
            )
            _apply_focus_from_chart(event, focus)
            st.caption(
                "Grand-mean band power on the scalp. Each map is scaled independently so spatial "
                f"structure stays visible; sensors overlaid. Focus: {batch.ch_names[focus]}."
            )
            if batch.labels is not None:
                class_means, classes = class_mean_band_power(power, batch.labels)
                tabs = st.tabs([str(batch.class_names.get(int(c), c)) for c in classes])
                for ti, (tab, mean) in enumerate(zip(tabs, class_means)):
                    with tab:
                        event = _plotly_chart(
                            band_topomaps(
                                batch.montage_xy,
                                mean,
                                names,
                                ch_names=batch.ch_names,
                                selected=focus,
                            ),
                            key=f"scalp-class-{key}-{ti}",
                            on_select=_PLOTLY_ON_SELECT,
                        )
                        _apply_focus_from_chart(event, focus)
                        st.caption(
                            "Class-mean band power on the scalp. Each map is scaled independently "
                            f"so spatial structure stays visible; sensors overlaid. Focus: {batch.ch_names[focus]}."
                        )
            show_math(st, "scalp")
        else:
            _plotly_chart(
                channel_stem(
                    np.nan_to_num(scores, neginf=0.0),
                    batch.ch_names,
                    "Channel score used for ranking",
                    highlight=picks,
                    bad=bad,
                    selected=focus,
                ),
                key=f"scalp-stem-{key}",
            )
            if batch.montage_xy is None:
                st.caption("No montage coordinates on this batch, so band-power topography is unavailable.")
            else:
                st.caption(
                    "Fewer than three sensors have coordinates, so band-power topography is unavailable."
                )

    elif view == "Discriminability":
        if ave is None or disc_times is None or pair_maps is None:
            st.warning("No class labels on this batch.")
        else:
            show = top_channels(scores, min(24, scores.size))
            pair_labels = [
                f"{batch.class_names.get(int(a), a)} vs {batch.class_names.get(int(b), b)}"
                for a, b in pairs
            ]
            _plotly_chart(
                pairwise_heatmaps(
                    pair_maps[:, show, :],
                    disc_times,
                    [batch.ch_names[i] for i in show],
                    pair_labels,
                    "Pairwise discriminability (Mann–Whitney |z|)",
                ),
                key=f"disc-maps-{key}",
            )
            st.caption(
                "One panel per class pair: Mann–Whitney U → |z| (README shorthand Wilcoxon). "
                "Channel ranking uses the mean across pairs."
            )
            ave_peak = np.array(ave, copy=True)
            ave_peak[bad] = np.nan
            peak_t, peak_ch, peak_ti = peak_map(ave_peak, disc_times)
            shown = show if show.size else np.arange(ave.shape[0])
            curve = np.asarray(ave)[shown].mean(axis=0)
            _plotly_chart(
                _disc_curve_figure(disc_times, curve, peak_t, float(curve[peak_ti])),
                key=f"disc-curve-{key}",
            )
            st.caption(
                f"Peak mean |z| at {peak_t:.3f} s on {batch.ch_names[peak_ch]}. "
                f"Focused channel: {batch.ch_names[focus]}. "
                "The curve averages |z| across the channels shown in the heatmaps; "
                "the marker is the time of the map maximum (all channels except excluded)."
            )
            show_math(st, "disc")

    elif view == "Ranking":
        event = _plotly_chart(
            channel_stem(
                np.nan_to_num(scores, neginf=0.0),
                batch.ch_names,
                "Channel score used for ranking",
                highlight=picks,
                bad=bad,
                selected=focus,
            ),
            key=f"rank-stem-{key}",
            on_select=_PLOTLY_ON_SELECT,
        )
        _apply_focus_from_chart(event, focus)
        if can_draw_scalp(batch.montage_xy):
            event = _plotly_chart(
                scalp_scatter(
                    batch.montage_xy,
                    picks,
                    f"Top {len(picks)} channels",
                    ch_names=batch.ch_names,
                    bad=bad,
                    selected=focus,
                ),
                key=f"rank-scalp-{key}",
                on_select=_PLOTLY_ON_SELECT,
            )
            _apply_focus_from_chart(event, focus)
        rank_cap = (
            f"Highlighted sensors are the visualization subset (waveforms / CWT / spectra). "
            f"Focus: {batch.ch_names[focus]}."
        )
        if int(np.sum(bad)):
            bad_names = ", ".join(batch.ch_names[i] for i in np.flatnonzero(bad))
            rank_cap += f" Bad channels: {bad_names}."
        st.caption(rank_cap)
        st.write("Top channels:", ", ".join(batch.ch_names[i] for i in picks))
        votes = session_channel_votes(batch, window, baseline, k)
        order = np.argsort(np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf))[::-1]
        table = ranking_table(scores, batch.ch_names, picks, votes)
        try:
            df_state = st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
                key=f"rank-table-{key}",
            )
            selection = getattr(df_state, "selection", None)
            rows = getattr(selection, "rows", None) if selection is not None else None
            if rows is None and isinstance(selection, dict):
                rows = selection.get("rows")
            if rows:
                row_i = int(rows[0])
                if 0 <= row_i < order.size:
                    ch = int(order[row_i])
                    if ch != focus:
                        st.session_state["focus_override"] = ch
                        st.rerun()
        except TypeError:
            st.dataframe(table, hide_index=True, width="stretch")
        if votes is not None:
            _plotly_chart(
                channel_stem(
                    votes.astype(float),
                    batch.ch_names,
                    "Session votes for the visualization subset",
                    highlight=picks,
                    bad=bad,
                    selected=focus,
                ),
                key=f"rank-votes-{key}",
            )
            st.caption(
                "How often each channel is in the top-k when ranking is computed per session. "
                "Sessions are not subjects."
            )
        show_math(st, "ranking")

    elif view == "MDS":
        if batch.labels is None:
            st.warning("No class labels on this batch.")
        else:
            metric_label = st.radio(
                "Distance",
                ["log-Euclidean", "affine-invariant Riemann"],
                horizontal=True,
                help="Both metrics are pyRiemann, full montage. Affine-invariant is the SPD geodesic.",
            )
            estimator = st.radio(
                "Covariance estimator",
                list(_COV_ESTIMATORS),
                horizontal=True,
                index=0,
                help="pyRiemann Covariances: Ledoit–Wolf (lwf), OAS, or sample (scm).",
            )
            metric = "logeuclid" if metric_label.startswith("log") else "riemann"
            covs = trial_covariances(batch, window, estimator=estimator)
            with st.spinner("Embedding trial covariances…"):
                dist0 = pairwise_distances(covs, metric=metric)
                xy0 = embed_mds(dist0)
            _plotly_chart(
                mds_scatter(
                    xy0,
                    batch.labels,
                    batch.sessions,
                    batch.class_names,
                    f"{metric_label} MDS (all channels)",
                ),
                key=f"mds-main-{key}",
            )
            st.caption(f"Kruskal stress-1: {mds_stress(dist0, xy0):.3f} (0 is exact).")
            n_ses = 0 if batch.sessions is None else int(np.unique(batch.sessions).size)
            if n_ses >= 2:
                with st.spinner("Session whitening…"):
                    whitened = session_whiten(covs, batch.sessions, metric=metric)
                    dist1 = pairwise_distances(whitened, metric=metric)
                    xy1 = embed_mds(dist1)
                _plotly_chart(
                    mds_scatter(
                        xy1,
                        batch.labels,
                        batch.sessions,
                        batch.class_names,
                        "After per-session whitening (all channels)",
                    ),
                    key=f"mds-white-{key}",
                )
                st.caption(f"Kruskal stress-1 after whitening: {mds_stress(dist1, xy1):.3f}.")
            est_name = _COV_EST_LABEL.get(estimator, estimator)
            st.caption(
                "Covariance geometry uses the full montage (all channels). "
                "Ranked-channel selection is not used. "
                f"Covariances, distances, and session whitening are pyRiemann "
                f"({est_name}; log-Euclidean or affine-invariant Riemann). "
                "Points are classical MDS."
            )
            show_math(st, "mds")

    else:
        if batch.labels is None:
            st.warning("No class labels on this batch.")
        else:
            st.caption(
                DECODE_NOTE
                + " Ranked-channel selection is not used. "
                "Log-Euclidean MDM is a second chance check. Not a BCI."
            )
            try:
                covs = trial_covariances(batch, window)
                report = logeuclid_lda_cv(batch, window, covs=covs)
                mdm = logeuclid_mdm_cv(batch, window, covs=covs)
            except ValueError as exc:
                st.error(str(exc))
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("LDA CV accuracy", f"{100 * report.accuracy:.1f}%")
                c2.metric("MDM CV accuracy", f"{100 * mdm.accuracy:.1f}%")
                c3.metric("Chance", f"{100 * report.chance:.0f}%")
                c4.metric("Majority", f"{100 * report.majority:.0f}%")
                _plotly_chart(_cv_fold_figure(report, mdm), key=f"chance-folds-{key}")
                st.write(
                    f"{report.method} · {report.n_splits}-fold · {report.n_features} features · "
                    f"{report.n_trials} trials · full montage (all channels)"
                )
                st.write(
                    f"{mdm.method} · {mdm.n_splits}-fold · {mdm.n_trials} trials · "
                    "log-Euclidean MDM (another chance check, not a BCI)"
                )
                show_math(st, "chance")

    with st.sidebar:
        st.header("Export")
        full = st.checkbox("Full HTML (per-class CWT)", value=False)
        if st.button("Build HTML snapshot"):
            with st.spinner("Rendering matplotlib report…"):
                st.session_state["html_export"] = render_report(
                    batch,
                    window=window,
                    baseline=baseline,
                    fmin=float(fmin),
                    fmax=float(fmax),
                    voices_per_octave=int(voices),
                    decim=int(decim),
                    top_k=k,
                    full=full,
                    sidecar=None,
                )
        snapshot = st.session_state.get("html_export")
        if snapshot:
            st.download_button(
                "Download HTML",
                data=snapshot,
                file_name="epochlens_report.html",
                mime="text/html",
            )
        if st.button("Build figure zip"):
            with st.spinner("Rendering figure zip…"):
                items = report_pngs(
                    batch,
                    window=window,
                    baseline=baseline,
                    fmin=float(fmin),
                    fmax=float(fmax),
                    voices_per_octave=int(voices),
                    decim=int(decim),
                    top_k=k,
                    full=full,
                )
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name, data in items:
                        zf.writestr(name, data)
                st.session_state["zip_export"] = buf.getvalue()
        zip_bytes = st.session_state.get("zip_export")
        if zip_bytes:
            st.download_button(
                "Download figure zip",
                data=zip_bytes,
                file_name="epochlens_figures.zip",
                mime="application/zip",
            )

    st.caption("EpochLens — first-look figures for labeled EEG (MIT).")


render()
