"""Epoch two short PhysioNet EEG Motor Imagery runs (subject 1). Not an eegvis adapter.

Downloads only runs 4 and 8 (~a few MB), epochs left vs right imagined fist, writes FIF.
"""

from __future__ import annotations

from pathlib import Path


def main() -> Path:
    import mne
    from mne.datasets import eegbci

    dest = Path("data")
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "eegbci_s01-epo.fif"

    files = eegbci.load_data(subjects=1, runs=[4, 8], update_path=True)
    raws = [mne.io.read_raw_edf(path, preload=True, verbose="ERROR") for path in files]
    for raw in raws:
        eegbci.standardize(raw)
        raw.set_montage("standard_1005", on_missing="ignore")
    raw = mne.concatenate_raws(raws, verbose="ERROR")
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
    epochs.save(out, overwrite=True)
    print(f"Wrote {out.resolve()}  ({len(epochs)} epochs, {len(epochs.ch_names)} ch, {epochs.info['sfreq']:g} Hz)")
    return out


if __name__ == "__main__":
    main()
