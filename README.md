# eegvis

First-look figures for **labeled EEG epochs**.

The goal is stunning, honest graphs of the signals — class-mean waveforms, time–frequency, scalp, and covariance geometry — so you can see whether a contrast is in the recording. Algorithms take generic epochs `(trials × channels × time)`. Class names come from the file: song A vs B, left vs right, any labels you epoch. The core does not know about named public datasets.

## Goal

A Streamlit explorer of epoched EEG that is meant to be looked at. Optional HTML export for sharing. Not a preprocessor, not a decoder, not a replacement for MNE.

## What it plots

- Class-mean waveforms (baseline z-scored) ± SEM
- Class-mean spectra in the analysis window
- Batched CWT scalograms
- Band-power topography
- Pairwise discriminability maps (Wilcoxon / t-test)
- Channel ranking (table + JSON sidecar)
- Log-Euclidean / Riemannian trial embeddings (including session whitening)
- Streamlit app (default) and optional HTML snapshot
- Optional MNE epochs FIF loader

A cross-validated LDA check against chance can appear in the report. That number is a sanity check on covariance geometry, not a product feature and not a BCI.

## What it is not

- A preprocessing pipeline (filter, rereference, ICA, rejection)
- A brain–computer interface or decoder
- A replacement for MNE’s full ERP / TFR / PSD / cleaning stack
- A MATLAB wrapper
- A dataset collection: epoch in MNE (or save NPZ), then load the epochs

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

Streamlit is the app. Synthetic demo loads with no file:

```text
python -m eegvis.explorer
```

Any already-epoched experiment (music listening, motor imagery, ERP, speech, …) as MNE FIF or eegvis NPZ. Upload in the sidebar, or pass a path:

```text
python -m eegvis.explorer --fif path/to/epochs-epo.fif
python -m eegvis.explorer --npz path/to/epochs.npz
```

Class labels come from MNE `event_id` or the NPZ sidecar. Epoch in MNE first if you have continuous recordings. eegvis does not cut raw data.

Optional static HTML snapshot (also available as Export in the sidebar):

```text
python -m eegvis.explorer --html --out report.html --open
python -m eegvis.explorer --fif path/to/epochs-epo.fif --html --window 0.0,2.0 --out report.html
```

Add `--full` on the HTML path for per-class CWT and extra topomaps.

## Core

```python
from eegvis import EpochBatch, write_html
from eegvis.cwt import cwt_power, mean_cwt_power
from eegvis.discriminability import pairwise_maps
from eegvis.ranking import score_channels, top_channels
from eegvis.riemann import trial_covariances, session_whiten, embed_mds

write_html(batch, "report.html", window=(0.5, 2.0), baseline=(0.0, 0.4))
```

Time windows, sampling rate, and montage are arguments. Nothing BioSemi- or experiment-specific belongs in the math.

## Layout

| Path | Role |
| --- | --- |
| `src/eegvis/` | Figure-facing algorithms |
| `src/eegvis/adapters/synthetic.py` | Built-in demo EEG |
| `src/eegvis/adapters/npz.py` | Save/load generic epochs |
| `src/eegvis/adapters/fif.py` | Any MNE `*-epo.fif` |
| `src/eegvis/explorer/` | Streamlit app + HTML export |

License: MIT.
