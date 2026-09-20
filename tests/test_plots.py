import numpy as np
import pytest

go = pytest.importorskip("plotly.graph_objects")

from eegvis.adapters.synthetic import make_synthetic
from eegvis.bands import band_power, window_spectrum
from eegvis.explorer.plots import band_topomaps, class_mean_spectra
from eegvis.ranking import prepare_ranking
from eegvis.topo import interpolate_topo


def _ranked_spectra(batch, window, baseline, top_k: int = 4):
    _, _, picks, *_ = prepare_ranking(batch, window, baseline, top_k)
    subset = batch.pick(picks)
    spec, freqs = window_spectrum(subset, window)
    means = {int(cls): spec[batch.labels == cls].mean(axis=0) for cls in np.unique(batch.labels)}
    return subset, freqs, means


def _yaxis_type(axis) -> str:
    return "" if axis.type is None else str(axis.type)


def test_mean_traces_share_ylim_so_noise_is_not_zoomed():
    from eegvis.explorer.plots import mean_traces
    from eegvis.waveforms import class_mean_sem

    batch = make_synthetic(n_channels=8, trials_per_class=8, seed=7)
    window, baseline = (0.6, 1.4), (0.0, 0.4)
    zbatch, _scores, picks, *_rest = prepare_ranking(batch, window, baseline, 4)
    zsub = zbatch.pick(picks)
    means, sems = class_mean_sem(zsub)
    fig = mean_traces(zsub.times, means, sems, batch.class_names, zsub.ch_names, window)
    ranges = []
    for key in fig.layout:
        if str(key).startswith("yaxis"):
            ax = fig.layout[key]
            if ax.range is not None:
                ranges.append((float(ax.range[0]), float(ax.range[1])))
    assert ranges
    lo, hi = ranges[0]
    assert hi >= 1.0
    assert abs(lo + hi) < 1e-9
    assert len(set(ranges)) == 1


def test_class_mean_spectra_log_y_includes_actual_max():
    batch = make_synthetic(n_channels=8, trials_per_class=6, seed=7)
    window, baseline = (0.6, 1.4), (0.0, 0.4)
    subset, freqs, spec_means = _ranked_spectra(batch, window, baseline)
    fig = class_mean_spectra(freqs, spec_means, batch.class_names, subset.ch_names)
    assert isinstance(fig, go.Figure)
    n_cls = len(spec_means)
    assert len(fig.data) == n_cls * len(subset.ch_names)

    fmax_show = min(float(freqs[-1]), 45.0)
    stacked = np.concatenate([spec_means[k][:, freqs <= fmax_show] for k in spec_means], axis=0)
    actual_max = float(np.max(stacked))
    y0 = fig.layout.yaxis
    assert _yaxis_type(y0) == "log"
    assert y0.range is not None
    hi = float(y0.range[1])
    assert hi >= np.log10(actual_max) - 1e-9
    assert 10 ** hi <= actual_max * 2.0
    shared = []
    for key in fig.layout:
        if str(key).startswith("yaxis") and key != "yaxis":
            ax = fig.layout[key]
            if ax.range is not None:
                shared.append((float(ax.range[0]), float(ax.range[1])))
                assert _yaxis_type(ax) in ("log", "")
            elif ax.matches:
                assert ax.matches in ("y", "y1")
    if shared:
        assert set(shared) == {(float(y0.range[0]), hi)}


def test_band_topomaps_one_subplot_per_band():
    batch = make_synthetic(n_channels=8, trials_per_class=6, seed=7)
    power, names = band_power(batch, (0.6, 1.4))
    grand = power.mean(axis=0)
    Xi, Yi, Zi = interpolate_topo(batch.montage_xy, grand[:, 0])
    assert Zi.shape == Xi.shape == Yi.shape
    assert np.any(np.isfinite(Zi))

    fig = band_topomaps(batch.montage_xy, grand, names)
    assert isinstance(fig, go.Figure)
    heatmaps = [t for t in fig.data if t.type == "heatmap"]
    scatters = [t for t in fig.data if t.type == "scatter"]
    assert len(heatmaps) == len(names)
    assert len(names) >= 2
    assert len(scatters) == len(names)
    titles = [ann.text for ann in fig.layout.annotations]
    for name in names:
        assert name in titles
    coloraxes = {t.coloraxis for t in heatmaps}
    assert len(coloraxes) == len(names)
    yaxes = [fig.layout.yaxis]
    yaxes.extend(fig.layout[k] for k in fig.layout if str(k).startswith("yaxis") and k != "yaxis")
    assert any(getattr(ax, "scaleanchor", None) for ax in yaxes)


def test_streamlit_helpers_cover_readme_plots():
    from eegvis.cwt import mean_cwt_power, relative_scalogram
    from eegvis.discriminability import pairwise_maps
    from eegvis.explorer.plots import (
        channel_stem,
        mean_traces,
        pairwise_heatmaps,
        scalogram_grid,
        scalp_scatter,
        mds_scatter,
    )
    from eegvis.riemann import embed_mds, pairwise_distances, session_whiten, trial_covariances
    from eegvis.waveforms import class_mean_sem

    batch = make_synthetic(n_channels=8, trials_per_class=6, seed=7)
    window, baseline = (0.6, 1.4), (0.0, 0.4)
    zbatch, scores, picks, ave, disc_times, pairs, _bad = prepare_ranking(
        batch, window, baseline, 4
    )
    subset = batch.pick(picks)
    zsub = zbatch.pick(picks)
    means, sems = class_mean_sem(zsub)
    assert isinstance(mean_traces(zsub.times, means, sems, batch.class_names, zsub.ch_names, window), go.Figure)

    spec, freqs = window_spectrum(subset, window)
    spec_means = {int(cls): spec[batch.labels == cls].mean(axis=0) for cls in np.unique(batch.labels)}
    assert isinstance(class_mean_spectra(freqs, spec_means, batch.class_names, subset.ch_names), go.Figure)

    power, cf, ct = mean_cwt_power(subset, fmin=6.0, fmax=20.0, voices_per_octave=4, decim=4, use_cache=False)
    rel = relative_scalogram(power, ct, baseline)
    assert isinstance(scalogram_grid(rel, ct, cf, subset.ch_names, title="cwt", window=window), go.Figure)

    bp, names = band_power(batch, window)
    assert isinstance(band_topomaps(batch.montage_xy, bp.mean(axis=0), names), go.Figure)

    maps, _ = pairwise_maps(zbatch.data[:, :, :8], batch.labels)
    labels = [f"{a} vs {b}" for a, b in pairs]
    assert isinstance(pairwise_heatmaps(maps[:, :4, :8], np.arange(8), subset.ch_names[:4], labels, "pairs"), go.Figure)
    assert isinstance(channel_stem(np.nan_to_num(scores, neginf=0.0), batch.ch_names, "rank"), go.Figure)
    assert isinstance(scalp_scatter(batch.montage_xy, picks, "scalp"), go.Figure)

    covs = trial_covariances(batch, window)
    xy = embed_mds(pairwise_distances(covs, metric="logeuclid"))
    assert isinstance(mds_scatter(xy, batch.labels, batch.sessions, batch.class_names, "mds"), go.Figure)
    whitened = session_whiten(covs, batch.sessions, metric="logeuclid")
    xy1 = embed_mds(pairwise_distances(whitened, metric="logeuclid"))
    assert isinstance(mds_scatter(xy1, batch.labels, batch.sessions, batch.class_names, "white"), go.Figure)
