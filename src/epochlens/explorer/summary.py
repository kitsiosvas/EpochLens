"""Glanceable facts about a loaded ``EpochBatch``."""

from __future__ import annotations

import numpy as np

from epochlens.types import EpochBatch


def _fmt_seconds(span: float) -> str:
    if span < 1.0:
        return f"{1000.0 * span:.0f} ms"
    if abs(span - round(span)) < 1e-6:
        return f"{int(round(span))} s"
    return f"{span:.2f} s".rstrip("0").rstrip(".")


def dataset_facts(batch: EpochBatch) -> dict[str, str]:
    tmin = float(batch.tmin)
    tmax = float(batch.times[-1])
    duration = float(batch.n_times) / float(batch.sfreq)
    facts = {
        "duration": _fmt_seconds(duration),
        "span": f"{tmin:g} – {tmax:g} s",
        "channels": str(batch.n_channels),
        "trials": str(batch.n_trials),
        "sfreq": f"{batch.sfreq:g} Hz",
        "samples": str(batch.n_times),
        "classes": "unlabeled",
        "class_detail": "",
        "sessions": "",
        "montage": "yes" if batch.montage_xy is not None else "no",
    }
    if batch.labels is not None:
        keys = [int(c) for c in np.unique(batch.labels)]
        parts = []
        for key in keys:
            n = int(np.sum(batch.labels == key))
            name = batch.class_names.get(key, str(key))
            parts.append(f"{name} × {n}")
        facts["classes"] = str(len(keys))
        facts["class_detail"] = ", ".join(parts)
    if batch.sessions is not None:
        facts["sessions"] = str(int(np.unique(batch.sessions).size))
    return facts


def facts_line(batch: EpochBatch) -> str:
    f = dataset_facts(batch)
    bits = [f"{f['duration']} epochs", f"{f['channels']} channels", f"{f['trials']} trials", f"{f['sfreq']}"]
    if f["class_detail"]:
        bits.append(f["class_detail"])
    bits.append(f"{f['samples']} samples · {f['span']}")
    if f["sessions"]:
        bits.append(f"{f['sessions']} sessions")
    if f["montage"] == "yes":
        bits.append("montage")
    return " · ".join(bits)
