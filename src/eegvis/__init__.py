"""Dataset-agnostic first-look figures for labeled EEG epochs."""

from __future__ import annotations

from pathlib import Path

from eegvis.bands import band_power
from eegvis.cwt import cwt_power, mean_cwt_power
from eegvis.decoding import logeuclid_lda_cv
from eegvis.discriminability import pairwise_maps
from eegvis.ranking import score_channels, top_channels
from eegvis.riemann import embed_mds, session_whiten, trial_covariances
from eegvis.types import EpochBatch
from eegvis.waveforms import class_mean_sem

__all__ = [
    "EpochBatch",
    "band_power",
    "class_mean_sem",
    "cwt_power",
    "embed_mds",
    "logeuclid_lda_cv",
    "mean_cwt_power",
    "pairwise_maps",
    "score_channels",
    "session_whiten",
    "top_channels",
    "trial_covariances",
    "write_html",
]
__version__ = "0.1.0"


def write_html(batch: EpochBatch, path: str | Path, *, window: tuple[float, float], baseline: tuple[float, float], **kwargs) -> Path:
    """Write a first-look HTML report (and a JSON sidecar) for ``batch``."""
    from eegvis.explorer.report import render_report, write_report

    out = Path(path)
    html = render_report(batch, window=window, baseline=baseline, sidecar=out.with_suffix(".json"), **kwargs)
    return write_report(html, out)
