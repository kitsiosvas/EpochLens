"""Single-file HTML report. Matplotlib only; no Streamlit required."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np

from eegvis.types import EpochBatch
from eegvis.cwt import energy_channel_score, mean_cwt_power, relative_scalogram
from eegvis.discriminability import aggregate_pairs, pairwise_maps
from eegvis.ranking import score_channels, top_channels
from eegvis.riemann import embed_mds, pairwise_distances, session_whiten, trial_covariances
from eegvis.windows import time_mask

HONESTY = (
    "Four-class inner speech on Nieto 2022 is typically near chance "
    "(about 25–37%; chance is 25%). This report visualizes time–frequency "
    "structure and covariance geometry. It is not a decoder."
)


def _png(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    fig.clf()
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _scalogram_figure(rel, times, freqs, ch_names, window, max_channels: int = 8):
    import matplotlib.pyplot as plt

    n_ch = min(rel.shape[0], max_channels)
    cols = 4
    rows = int(np.ceil(n_ch / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.4 * rows), squeeze=False, layout="constrained")
    vmax = np.percentile(np.abs(rel[:n_ch]), 98)
    im = None
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n_ch:
            ax.axis("off")
            continue
        im = ax.pcolormesh(times, freqs, rel[i], shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axvline(window[0], color="k", lw=0.8, ls="--")
        ax.axvline(window[1], color="k", lw=0.8, ls="--")
        ax.set_title(ch_names[i], fontsize=9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="relative power")
    fig.suptitle("Relative CWT scalograms (baseline-normalized)", fontsize=12)
    return fig


def _stem_figure(scores, ch_names, title: str, picks=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.2))
    x = np.arange(1, len(scores) + 1)
    ax.bar(x, scores, color="#444", width=0.8)
    if picks is not None:
        ax.bar(x[np.asarray(picks)], scores[np.asarray(picks)], color="#b33", width=0.8)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Score")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _scalp_figure(xy, picks, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(xy[:, 0], xy[:, 1], c="#888", s=28)
    ax.scatter(xy[picks, 0], xy[picks, 1], c="#b33", s=48, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _heatmap_figure(z, x, y_label: str, title: str, xlabel: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(z, aspect="auto", origin="lower", cmap="hot", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    if x is not None and len(x) > 1:
        ticks = np.linspace(0, z.shape[1] - 1, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{x[int(t)]:.2f}" for t in ticks])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def _mds_figure(xy, labels, sessions, class_names, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    markers = ["o", "s", "D", "^"]
    sessions = np.ones(len(labels), dtype=int) if sessions is None else sessions
    for cls in np.unique(labels):
        for ses in np.unique(sessions):
            m = (labels == cls) & (sessions == ses)
            if not np.any(m):
                continue
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                marker=markers[int(ses) % len(markers)],
                s=36,
                alpha=0.85,
                label=f"{class_names.get(int(cls), cls)} / ses {ses}",
            )
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    return fig


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
) -> str:
    """Return a self-contained HTML document."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    mean_power, freqs, times = mean_cwt_power(
        batch,
        fmin=fmin,
        fmax=fmax,
        voices_per_octave=voices_per_octave,
        decim=decim,
        use_cache=True,
    )
    rel = relative_scalogram(mean_power, times, baseline)
    energy = energy_channel_score(rel, times, window)
    energy_picks = top_channels(energy, min(8, batch.n_channels))
    scal_names = [batch.ch_names[i] for i in energy_picks]

    sections: list[tuple[str, str]] = []
    sections.append(
        (
            "Relative CWT scalograms (highest-energy channels)",
            _png(_scalogram_figure(rel[energy_picks], times, freqs, scal_names, window, max_channels=8)),
        )
    )
    sections.append(("CWT energy score in the action window", _png(_stem_figure(energy, batch.ch_names, "CWT energy score"))))

    if batch.labels is not None:
        mask = time_mask(batch.times, window[0], window[1])
        maps, pairs = pairwise_maps(batch.data[:, :, mask], batch.labels, method="wilcoxon")
        ave = aggregate_pairs(maps, "mean")
        scores = score_channels(ave)
        picks = top_channels(scores, top_k)
        pair_txt = ", ".join(f"{a} vs {b}" for a, b in pairs)
        sections.append(
            (
                f"Mean pairwise |Wilcoxon z| ({pair_txt})",
                _png(
                    _heatmap_figure(
                        ave,
                        batch.times[mask],
                        "Channel",
                        "Discriminability (channels × time)",
                        "Time (s)",
                    )
                ),
            )
        )
        sections.append(
            (
                "Discriminability channel score",
                _png(_stem_figure(scores, batch.ch_names, "Discriminability score", picks=picks)),
            )
        )
        if batch.montage_xy is not None:
            sections.append(
                (
                    f"Top {len(picks)} channels",
                    _png(_scalp_figure(batch.montage_xy, picks, "Top channels")),
                )
            )

        covs = trial_covariances(batch.pick(picks), window)
        xy0 = embed_mds(pairwise_distances(covs, metric="logeuclid"))
        sections.append(
            (
                "Log-Euclidean MDS of trial covariances",
                _png(_mds_figure(xy0, batch.labels, batch.sessions, batch.class_names, "Before session whitening")),
            )
        )
        if batch.sessions is not None:
            xy1 = embed_mds(
                pairwise_distances(session_whiten(covs, batch.sessions, metric="logeuclid"), metric="logeuclid")
            )
            sections.append(
                (
                    "After per-session whitening",
                    _png(_mds_figure(xy1, batch.labels, batch.sessions, batch.class_names, "After session whitening")),
                )
            )

    meta = (
        f"{batch.dataset} · subject {batch.subject_id} · condition {batch.condition} · "
        f"{batch.n_trials} trials · {batch.n_channels} channels · {batch.sfreq:g} Hz · "
        f"action window {window[0]:g}–{window[1]:g} s"
    )
    blocks = []
    for title, b64 in sections:
        blocks.append(f"<h2>{title}</h2><img alt=\"{title}\" src=\"data:image/png;base64,{b64}\"/>")
    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>eegvis — {batch.dataset or "report"}</title>
  <style>
    body {{ font-family: "Segoe UI", Helvetica, sans-serif; max-width: 1080px;
           margin: 2rem auto; padding: 0 1.25rem 3rem; color: #1a1a1a; background: #fafafa; }}
    h1 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: 0.35rem; }}
    .sub {{ color: #555; margin-bottom: 1.25rem; }}
    .banner {{ background: #fff6e5; border: 1px solid #e6d5a8; padding: 0.9rem 1rem; margin: 1rem 0 1.5rem; }}
    img {{ width: 100%; background: #fff; border: 1px solid #e6e6e6; }}
    h2 {{ font-size: 1.05rem; font-weight: 600; margin: 1.6rem 0 0.5rem; }}
    footer {{ color: #777; font-size: 0.85rem; margin-top: 2rem; }}
  </style>
</head>
<body>
  <h1>eegvis</h1>
  <p class="sub">Dataset-agnostic EEG visualization. Not a decoder.</p>
  <div class="banner">{HONESTY}</div>
  <p>{meta}</p>
  {body}
  <footer>
    Cite the dataset: Nieto et al., Scientific Data 2022.
    This explorer is GPL-3 (hard fork in purpose of N-Nieto/Inner_Speech_Dataset).
  </footer>
</body>
</html>
"""


def write_report(html: str, path: Path) -> Path:
    path = Path(path)
    path.write_text(html, encoding="utf-8")
    return path
