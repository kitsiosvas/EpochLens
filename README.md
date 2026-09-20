# eegvis

Independent Python toolkit for **EEG visualization**: scalograms, class-separability maps, sensor ranking, and Riemannian covariance embeddings.

Algorithms take generic epochs `(trials × channels × time)`. Optional loaders can pull public datasets; the core does not.

## What it does

- Batched CWT scalograms
- Pairwise discriminability maps (Wilcoxon / t-test)
- Channel ranking and voting
- Log-Euclidean / Riemannian trial embeddings (including session whitening)
- HTML explorer (matplotlib) and optional Streamlit UI

## What it is not

- A brain–computer interface or inner-speech decoder
- A replacement for MNE’s standard ERP / TFR / PSD plots
- A MATLAB wrapper

## Install

```text
python -m pip install -U pip
pip install -e ".[dev]"
```

On older pip, run tests from the tree (`pythonpath = src` is set in `pyproject.toml`):

```text
python -m pytest tests
```

## Explorer

Synthetic demo (no dataset required):

```text
python -m eegvis.explorer --out eegvis_report.html --open
```

Optional public dataset loader (OpenNeuro [ds003626](https://openneuro.org/datasets/ds003626), Nieto et al. 2022) — one adapter, not the product:

```text
python -m eegvis.explorer --source nieto --subject 1 --root /path/to/ds003626 --out nieto.html
```

When that adapter is used, the report notes that 4-class inner speech on that set is typically near chance (~25–37%).

## Core

```python
from eegvis import EpochBatch
from eegvis.cwt import cwt_power, mean_cwt_power
from eegvis.discriminability import pairwise_maps
from eegvis.ranking import score_channels, top_channels
from eegvis.riemann import trial_covariances, session_whiten, embed_mds
```

Time windows, sampling rate, and montage are arguments. Nothing BioSemi- or experiment-specific belongs in the math.

## Layout

| Path | Role |
| --- | --- |
| `src/eegvis/` | Visualization algorithms |
| `src/eegvis/adapters/synthetic.py` | Built-in demo EEG |
| `src/eegvis/adapters/nieto.py` | Optional OpenNeuro ds003626 loader |
| `src/eegvis/explorer/` | HTML report + optional Streamlit UI |

License: MIT.
