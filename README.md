# EpochLens

First-look figures for **labeled EEG epochs**.

Algorithms take generic epochs `(trials × channels × time)`. Class names come from the file (song A vs B, left vs right, anything you epoch). The core does not know about named public datasets.

## Goal

A Streamlit explorer of epoched EEG that is meant to be looked at. Optional HTML export for sharing. Not a preprocessor, not a decoder, not a replacement for MNE.

## What it plots

- Class-mean waveforms (baseline z-scored) ± SEM
- Class-mean spectra in the analysis window (log power)
- Batched Morlet CWT scalograms (power = `|coeff|²`)
- Band-power topography (θ / α / β / γ)
- Pairwise discriminability maps (Mann–Whitney |z|; README shorthand Wilcoxon)
- Channel ranking (table + JSON sidecar on HTML export)
- Log-Euclidean / Riemannian trial embeddings, including session whitening
- Optional cross-validated log-Euclid LDA vs chance (sanity check, not a BCI)

Waveforms, spectra, and CWT use a ranked-channel subset. MDS and the chance check use the **full montage**.

Streamlit views: Waveforms, Time–frequency, Scalp, Discriminability, Ranking, MDS, vs chance. Each view includes the LaTeX for the equations EpochLens actually computes.

## What it is not

- A preprocessing pipeline (filter, rereference, ICA, rejection)
- A brain–computer interface or decoder
- A replacement for MNE’s full ERP / TFR / PSD / cleaning stack
- A MATLAB wrapper
- A dataset collection: epoch in MNE (or save NPZ), then load the epochs

## Install

Python 3.10+. From the repo root:

```text
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
```

`[dev]` is enough for Streamlit, Plotly, matplotlib, and tests. Add `[fif]` (or use `[all]`) for MNE `*-epo.fif`. `.venv/` is gitignored.

```text
.\.venv\Scripts\python.exe -m pytest tests
```

## Explorer

Streamlit is the app. Dummy data loads with no file:

```text
.\.venv\Scripts\python.exe -m epochlens.explorer
```

(`epochlens` is the same entry point.)

Any already-epoched experiment (music listening, motor imagery, ERP, speech, …) as MNE FIF or EpochLens NPZ. Upload in the sidebar, or pass a path:

```text
.\.venv\Scripts\python.exe -m epochlens.explorer --fif path/to/epochs-epo.fif
.\.venv\Scripts\python.exe -m epochlens.explorer --npz path/to/epochs.npz
```

Class labels come from MNE `event_id` or the NPZ sidecar. Epoch in MNE first if you have continuous recordings. EpochLens does not cut raw data.

Optional static HTML snapshot (also **Export** in the Streamlit sidebar):

```text
.\.venv\Scripts\python.exe -m epochlens.explorer --html --out report.html --open
.\.venv\Scripts\python.exe -m epochlens.explorer --fif path/to/epochs-epo.fif --html --window 0.0,2.0 --out report.html
```

`--full` on the HTML path adds per-class CWT and extra topomaps. `--window` / `--baseline-end` apply to HTML only; Streamlit has sliders. HTML math is typeset with MathJax (needs a network connection when you open the file).

Two short PhysioNet motor-imagery runs (subject 1, a few MB) can be epoched to FIF with `examples/epoch_eegbci.py`, then loaded like any other epochs file.

## Core

```python
from epochlens import EpochBatch, write_html
from epochlens.adapters.npz import save_epochs, load_epochs
from epochlens.cwt import cwt_power, mean_cwt_power
from epochlens.discriminability import pairwise_maps
from epochlens.ranking import score_channels, top_channels
from epochlens.riemann import trial_covariances, session_whiten, embed_mds

write_html(batch, "report.html", window=(0.5, 2.0), baseline=(0.0, 0.4))
```

FIF (needs MNE):

```python
from epochlens.adapters.fif import load_epochs_fif

batch = load_epochs_fif("path/to/epochs-epo.fif")
```

The FIF loader keeps EEG channels only. Session ids come from MNE metadata (`session`, `run`, …) when present.

Time windows, sampling rate, and montage are arguments. Nothing BioSemi- or experiment-specific belongs in the math.

## Layout

| Path | Role |
| --- | --- |
| `src/epochlens/` | Figure-facing algorithms |
| `src/epochlens/adapters/synthetic.py` | Built-in demo EEG |
| `src/epochlens/adapters/npz.py` | Save/load generic epochs (no MNE) |
| `src/epochlens/adapters/fif.py` | Any MNE `*-epo.fif` |
| `src/epochlens/explorer/` | Streamlit app + HTML export |

License: MIT.
