"""Epoch two short PhysioNet EEG Motor Imagery runs (subject 1). Not an EpochLens adapter.

Downloads only runs 4 and 8 (~a few MB), epochs left vs right imagined fist, writes FIF.
Each run is epoched separately with a ``run`` metadata column, then concatenated.
"""

from __future__ import annotations

from pathlib import Path


def main() -> Path:
    import mne
    import pandas as pd
    from mne.datasets import eegbci

    dest = Path("data")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "eegbci_s01-epo.fif"

    runs = [4, 8]
    files = eegbci.load_data(subjects=1, runs=runs, update_path=True)
    pieces = []
    for run, path in zip(runs, files):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore")
        raw.filter(1.0, 40.0, verbose="ERROR")
        events, _ = mne.events_from_annotations(raw, event_id={"T1": 2, "T2": 3}, verbose="ERROR")
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"left": 2, "right": 3},
            tmin=0.0,
            tmax=2.0,
            baseline=None,
            preload=True,
            verbose="ERROR",
        )
        epochs.metadata = pd.DataFrame({"run": [run] * len(epochs)})
        pieces.append(epochs)
    epochs = mne.concatenate_epochs(pieces, verbose="ERROR")
    epochs.save(out, overwrite=True)
    print(f"Wrote {out.resolve()}  ({len(epochs)} epochs, {len(epochs.ch_names)} ch, {epochs.info['sfreq']:g} Hz)")
    return out


if __name__ == "__main__":
    main()
