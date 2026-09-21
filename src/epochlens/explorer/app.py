"""Streamlit explorer. Any labeled epochs (FIF / NPZ / dummy data)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from epochlens.adapters.synthetic import make_synthetic
from epochlens.bands import band_power, window_spectrum
from epochlens.cwt import class_relative_scalograms, mean_cwt_power, relative_scalogram
from epochlens.decoding import logeuclid_lda_cv
from epochlens.discriminability import pairwise_maps
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
)
from epochlens.explorer.report import DECODE_NOTE, HONESTY, render_report
from epochlens.ranking import prepare_ranking, session_channel_votes, top_channels
from epochlens.riemann import embed_mds, pairwise_distances, session_whiten, trial_covariances
from epochlens.explorer.summary import dataset_facts
from epochlens.topo import can_draw_scalp
from epochlens.waveforms import class_mean_sem
from epochlens.windows import default_windows


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
    zbatch, scores, picks, ave, disc_times, pairs, bad = prepare_ranking(
        batch, window, baseline, k
    )
    subset = batch.pick(picks)
    zsub = zbatch.pick(picks)

    view = st.radio(
        "View",
        ["Waveforms", "Time–frequency", "Scalp", "Discriminability", "Ranking", "MDS", "vs chance"],
        horizontal=True,
    )

    if view == "Waveforms":
        means, sems = class_mean_sem(zsub)
        st.plotly_chart(
            mean_traces(zsub.times, means, sems, batch.class_names, zsub.ch_names, window),
            width="stretch",
        )
        st.caption("Ranked channels. Baseline z-scored class means ± SEM; shaded region is the analysis window.")
        show_math(st, "waveforms")
        spec_freqs, spec_means = _class_spectra(subset, window)
        st.plotly_chart(
            class_mean_spectra(spec_freqs, spec_means, batch.class_names, subset.ch_names),
            width="stretch",
        )
        st.caption(
            "Class-mean spectra in the analysis window (ranked channels). "
            "Log power; shared y-range includes the actual maximum."
        )
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
        st.plotly_chart(
            scalogram_grid(
                rel,
                times,
                freqs,
                subset.ch_names,
                title="Relative CWT scalogram (baseline-normalized, ranked channels)",
                window=window,
                max_channels=k,
            ),
            width="stretch",
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
            st.plotly_chart(
                class_scalogram_grid(
                    rel_by_class,
                    cls_times,
                    cls_freqs,
                    cls_names,
                    batch.class_names,
                    window,
                ),
                width="stretch",
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
            st.plotly_chart(
                band_topomaps(batch.montage_xy, grand, names),
                width="stretch",
            )
            st.caption(
                "Mean band power on the scalp. Each map is scaled independently so spatial "
                "structure stays visible; sensors overlaid."
            )
            show_math(st, "scalp")
        else:
            st.plotly_chart(
                channel_stem(
                    np.nan_to_num(scores, neginf=0.0),
                    batch.ch_names,
                    "Channel score used for ranking",
                ),
                width="stretch",
            )
            if batch.montage_xy is None:
                st.caption("No montage coordinates on this batch, so band-power topography is unavailable.")
            else:
                st.caption(
                    "Fewer than three sensors have coordinates, so band-power topography is unavailable."
                )

    elif view == "Discriminability":
        if ave is None or disc_times is None:
            st.warning("No class labels on this batch.")
        else:
            show = top_channels(scores, min(24, scores.size))
            t_idx = np.clip(np.searchsorted(zbatch.times, disc_times), 0, zbatch.times.size - 1)
            maps, _ = pairwise_maps(zbatch.data[:, :, t_idx], batch.labels, method="wilcoxon")
            pair_labels = [
                f"{batch.class_names.get(int(a), a)} vs {batch.class_names.get(int(b), b)}"
                for a, b in pairs
            ]
            st.plotly_chart(
                pairwise_heatmaps(
                    maps[:, show, :],
                    disc_times,
                    [batch.ch_names[i] for i in show],
                    pair_labels,
                    "Pairwise discriminability (Mann–Whitney |z|)",
                ),
                width="stretch",
            )
            st.caption(
                "One panel per class pair: Mann–Whitney U → |z| (README shorthand Wilcoxon). "
                "Channel ranking uses the mean across pairs."
            )
            show_math(st, "disc")

    elif view == "Ranking":
        st.plotly_chart(
            channel_stem(
                np.nan_to_num(scores, neginf=0.0),
                batch.ch_names,
                "Channel score used for ranking",
            ),
            width="stretch",
        )
        if can_draw_scalp(batch.montage_xy):
            st.plotly_chart(
                scalp_scatter(batch.montage_xy, picks, f"Top {len(picks)} channels"),
                width="stretch",
            )
        st.caption("Highlighted sensors are the visualization subset (waveforms / CWT / spectra).")
        st.write("Top channels:", ", ".join(batch.ch_names[i] for i in picks))
        votes = session_channel_votes(batch, window, baseline, k)
        if votes is not None:
            st.plotly_chart(
                channel_stem(votes.astype(float), batch.ch_names, "Session votes for the visualization subset"),
                width="stretch",
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
            metric = "logeuclid" if metric_label.startswith("log") else "riemann"
            covs = trial_covariances(batch, window)
            with st.spinner("Embedding trial covariances…"):
                xy0 = embed_mds(pairwise_distances(covs, metric=metric))
            st.plotly_chart(
                mds_scatter(
                    xy0,
                    batch.labels,
                    batch.sessions,
                    batch.class_names,
                    f"{metric_label} MDS (all channels)",
                ),
                width="stretch",
            )
            n_ses = 0 if batch.sessions is None else int(np.unique(batch.sessions).size)
            if n_ses >= 2:
                with st.spinner("Session whitening…"):
                    whitened = session_whiten(covs, batch.sessions, metric=metric)
                    xy1 = embed_mds(pairwise_distances(whitened, metric=metric))
                st.plotly_chart(
                    mds_scatter(
                        xy1,
                        batch.labels,
                        batch.sessions,
                        batch.class_names,
                        "After per-session whitening (all channels)",
                    ),
                    width="stretch",
                )
            st.caption(
                "Covariance geometry uses the full montage (all channels). "
                "Ranked-channel selection is not used. "
                "Covariances, distances, and session whitening are pyRiemann "
                "(Ledoit–Wolf; log-Euclidean or affine-invariant Riemann). "
                "Points are classical MDS."
            )
            show_math(st, "mds")

    else:
        if batch.labels is None:
            st.warning("No class labels on this batch.")
        else:
            st.caption(DECODE_NOTE + " Ranked-channel selection is not used. Not a BCI.")
            try:
                report = logeuclid_lda_cv(batch, window)
            except ValueError as exc:
                st.error(str(exc))
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("CV accuracy", f"{100 * report.accuracy:.1f}%")
                c2.metric("Chance", f"{100 * report.chance:.0f}%")
                c3.metric("Majority", f"{100 * report.majority:.0f}%")
                st.write(
                    f"{report.method} · {report.n_splits}-fold · {report.n_features} features · "
                    f"{report.n_trials} trials · full montage (all channels)"
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

    st.caption("EpochLens — first-look figures for labeled EEG (MIT).")


render()
