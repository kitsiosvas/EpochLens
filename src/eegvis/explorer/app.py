"""Streamlit explorer. Synthetic demo by default; Nieto is optional."""

from __future__ import annotations

import streamlit as st

from eegvis.adapters.synthetic import make_synthetic
from eegvis.cwt import cwt_power, energy_channel_score, mean_scalogram, relative_scalogram
from eegvis.discriminability import aggregate_pairs, pairwise_maps
from eegvis.explorer.plots import channel_stem, heatmap, mds_scatter, scalogram_grid, scalp_scatter
from eegvis.ranking import score_channels, top_channels
from eegvis.riemann import embed_mds, pairwise_distances, session_whiten, trial_covariances
from eegvis.windows import time_mask


st.set_page_config(page_title="eegvis", layout="wide")


@st.cache_data(show_spinner=True)
def _load_batch(source: str, subject: int, condition: str, root: str):
    if source == "Synthetic demo":
        return make_synthetic()
    from eegvis.adapters.nieto import load_nieto

    kwargs = {"subject": subject, "condition": condition}
    if root.strip():
        kwargs["root"] = root.strip()
    return load_nieto(**kwargs)


def render() -> None:
    st.title("eegvis")
    st.caption("Dataset-agnostic EEG visualization. Not a decoder.")

    st.info(
        "Four-class inner speech on Nieto 2022 is typically near chance "
        "(about 25–37%; chance is 25%). This tool shows structure in time–frequency "
        "and covariance geometry. It does not classify inner speech."
    )

    with st.sidebar:
        st.header("Data")
        source = st.selectbox("Source", ["Synthetic demo", "Nieto 2022"])
        subject = st.number_input("Subject", min_value=1, max_value=10, value=1, step=1)
        condition = st.selectbox("Condition", ["inner", "pronounced", "visualized"])
        root = st.text_input("Nieto root (derivatives parent)", value="")
        st.header("Windows (seconds)")
        st.caption("On Nieto 2022 the paper's action window is 1.0–3.5 s (epoch tmin is usually −0.5 s).")
        t_pre = st.slider("Baseline end", 0.0, 1.0, 0.5, 0.05)
        t_act0 = st.slider("Action start", 0.0, 3.0, 0.6, 0.05)
        t_act1 = st.slider("Action end", 0.5, 5.0, 1.4, 0.05)
        st.header("CWT")
        fmin = st.number_input("fmin (Hz)", 1.0, 40.0, 4.0)
        fmax = st.number_input("fmax (Hz)", 8.0, 80.0, 40.0)
        voices = st.number_input("Voices / octave", 4, 40, 12)
        decim = st.number_input("Time decim", 1, 8, 2)
        top_k = st.slider("Top channels", 3, 32, 8)

    try:
        batch = _load_batch(source, int(subject), condition, root)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.write(
        f"**{batch.dataset}** · subject `{batch.subject_id}` · condition `{batch.condition}` · "
        f"{batch.n_trials} trials · {batch.n_channels} channels · {batch.sfreq:g} Hz"
    )

    window = (float(t_act0), float(t_act1))
    baseline = (float(batch.tmin), float(t_pre))

    tab_sc, tab_disc, tab_rank, tab_spd = st.tabs(
        ["Scalograms", "Discriminability", "Sensor ranking", "Riemannian MDS"]
    )

    with tab_sc:
        power, freqs, times = cwt_power(
            batch,
            fmin=float(fmin),
            fmax=float(fmax),
            voices_per_octave=int(voices),
            decim=int(decim),
        )
        means = mean_scalogram(power, batch.labels)
        all_sc = means[None]
        rel = relative_scalogram(all_sc, times, baseline)
        st.plotly_chart(
            scalogram_grid(
                rel,
                times,
                freqs,
                batch.ch_names,
                title="Relative CWT scalogram (baseline-normalized, all classes)",
                window=window,
                max_channels=min(16, batch.n_channels),
            ),
            use_container_width=True,
        )
        energy = energy_channel_score(rel, times, window)
        st.plotly_chart(
            channel_stem(energy, batch.ch_names, "CWT energy score in action window"),
            use_container_width=True,
        )

    with tab_disc:
        if batch.labels is None:
            st.warning("No class labels on this batch.")
        else:
            mask = time_mask(batch.times, window[0], window[1])
            feats = batch.data[:, :, mask]
            maps, pairs = pairwise_maps(feats, batch.labels, method="wilcoxon")
            ave = aggregate_pairs(maps, "mean")
            st.plotly_chart(
                heatmap(
                    ave,
                    batch.times[mask],
                    list(range(1, batch.n_channels + 1)),
                    "Mean pairwise |Wilcoxon z| (channels × time in window)",
                    "Time (s)",
                    "Channel",
                ),
                use_container_width=True,
            )
            st.caption("Pairs: " + ", ".join(f"{a} vs {b}" for a, b in pairs))

    with tab_rank:
        if batch.labels is None:
            st.warning("No class labels on this batch.")
        else:
            mask = time_mask(batch.times, window[0], window[1])
            maps, _ = pairwise_maps(batch.data[:, :, mask], batch.labels, method="wilcoxon")
            scores = score_channels(aggregate_pairs(maps, "mean"))
            picks = top_channels(scores, int(top_k))
            st.plotly_chart(
                channel_stem(scores, batch.ch_names, "Discriminability channel score"),
                use_container_width=True,
            )
            if batch.montage_xy is not None:
                st.plotly_chart(
                    scalp_scatter(batch.montage_xy, picks, f"Top {len(picks)} channels"),
                    use_container_width=True,
                )
            st.write("Top channels:", ", ".join(batch.ch_names[i] for i in picks))

    with tab_spd:
        covs = trial_covariances(batch, window)
        dist0 = pairwise_distances(covs, metric="logeuclid")
        xy0 = embed_mds(dist0)
        st.plotly_chart(
            mds_scatter(xy0, batch.labels, batch.sessions, batch.class_names, "Log-Euclidean MDS"),
            use_container_width=True,
        )
        if batch.sessions is not None:
            whitened = session_whiten(covs, batch.sessions, metric="logeuclid")
            xy1 = embed_mds(pairwise_distances(whitened, metric="logeuclid"))
            st.plotly_chart(
                mds_scatter(
                    xy1,
                    batch.labels,
                    batch.sessions,
                    batch.class_names,
                    "After per-session whitening",
                ),
                use_container_width=True,
            )

    st.caption(
        "Cite the dataset: Nieto et al., Scientific Data 2022. "
        "This explorer is GPL-3 (hard fork in purpose of N-Nieto/Inner_Speech_Dataset)."
    )


render()
