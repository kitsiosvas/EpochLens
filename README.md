# eegvis

Python EEG visualization — **not** a copy of the Inner Speech Dataset processing pipeline.

This package is the product side of a hard fork of [N-Nieto/Inner_Speech_Dataset](https://github.com/N-Nieto/Inner_Speech_Dataset). Scope is different: a **dataset-agnostic algorithm core** and a fast explorer. [Nieto et al. 2022](https://www.nature.com/articles/s41597-022-01147-2) (OpenNeuro [ds003626](https://openneuro.org/datasets/ds003626)) is the first adapter and demo, not the math.

Licensed **GPL-3** (inherited from the upstream fork). See `NOTICE`.

## What it is

A library + viewer for:

- batched CWT scalograms
- pairwise discriminability maps (Wilcoxon / t-test)
- sensor ranking and cross-subject voting
- Riemannian / log-Euclidean covariance embeddings (including session whitening)

Inputs are generic epochs `(n_trials, n_channels, n_times)` plus sampling rate, channel names, labels, and sessions. No BioSemi-128, event codes, or Nieto time windows are baked into the core.

## What it is not

- A think-to-type BCI
- A claim that four-class inner speech is decoded (careful papers on this set sit near **25–37%**; chance is **25%**)
- A replacement for MNE ERP / TFR / PSD plots
- A MATLAB wrapper

## Install

Python 3.10+ with a current pip:

```text
python -m pip install -U pip
pip install -e ".[dev]"
```

On older pip (21.x), the editable install can fail. Run tests against the tree instead:

```text
cd eegvis
python -m pytest tests
```

(`pyproject.toml` sets `pythonpath = ["src"]` for pytest.)

## Explorer

Default path writes a single HTML report (synthetic EEG, planted class effect). No Streamlit required:

```text
cd eegvis
python -m eegvis.explorer --out eegvis_report.html --open
```

The page states that 4-class inner speech on Nieto 2022 is typically near chance. Optional Streamlit UI: `python -m eegvis.explorer --streamlit` after `pip install streamlit plotly`.

Nieto 2022 (local derivatives folder that contains `derivatives/`):

```text
python -m eegvis.explorer --source nieto --subject 1 --root /path/to/ds003626 --out nieto.html
```

## Core contract

```python
from eegvis import EpochBatch
from eegvis.cwt import cwt_power, mean_scalogram
from eegvis.discriminability import pairwise_maps
from eegvis.ranking import score_channels, top_channels
from eegvis.riemann import trial_covariances, session_whiten, embed_mds
```

`EpochBatch.data` is `float64` with shape `(n_trials, n_channels, n_times)`. Time windows are arguments, not constants.

## Layout

| Path | Role |
| --- | --- |
| `src/eegvis/` | Dataset-agnostic algorithms |
| `src/eegvis/adapters/nieto.py` | Nieto 2022 only |
| `src/eegvis/adapters/synthetic.py` | Demo / tests without a dataset |
| `src/eegvis/explorer/` | HTML report + optional Streamlit UI |

Parent-tree `Matlab/` and `Python_Processing/` are **reference only**. Do not import them.
