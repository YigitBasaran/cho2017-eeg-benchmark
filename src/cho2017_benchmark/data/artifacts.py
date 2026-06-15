"""Artifact / bad-trial bookkeeping.

The dataset provides ``bad_trial_indices`` (class-specific, one-based) split into
a voltage/amplitude criterion and an EMG-correlation ("mi") criterion. We map
those raw indices to explicit per-trial boolean flags and an exclusion reason,
using the resolved index base and reason mapping -- never a hard-coded guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import paths
from .epoching import EpochedSubject
from .loader import SubjectRecording

EXCLUDED_TRIALS_COLUMNS = [
    "subject_id", "class_label", "original_trial_index", "run_id",
    "exclusion_reason", "source", "notes",
]

_REASONS = ("amplitude", "emg_correlation", "provided_flag")


def excluded_subjects(cfg: Any) -> set[int]:
    """Subject indices removed from the model comparison (e.g. {29, 34})."""
    return {int(s) for s in cfg.excluded_subjects}


def compute_trial_flags(
    recording: SubjectRecording,
    epoched: EpochedSubject,
    resolution: dict[str, Any],
) -> dict[str, Any]:
    """Compute per-trial bad/included flags aligned to *epoched* trial order."""
    offset = 1 if resolution.get("index_base", "one_based") == "one_based" else 0
    reason_mapping = resolution.get("rejection_reason_mapping", {})

    n_by_side = {
        "left": int(np.sum(epoched.class_label == "left_hand")),
        "right": int(np.sum(epoched.class_label == "right_hand")),
    }
    sets: dict[str, dict[str, set[int]]] = {
        side: {reason: set() for reason in _REASONS} for side in ("left", "right")
    }
    notes: list[str] = []

    for field, sides in recording.bad_trial_indices.items():
        reason = reason_mapping.get(field, "provided_flag")
        if reason not in _REASONS:
            reason = "provided_flag"
        for side in ("left", "right"):
            raw = np.asarray(sides.get(side, np.array([], dtype=int)), dtype=int)
            if raw.size == 0:
                continue
            idx0 = raw - offset
            n = n_by_side[side]
            valid = idx0[(idx0 >= 0) & (idx0 < n)]
            dropped = idx0[(idx0 < 0) | (idx0 >= n)]
            if dropped.size:
                notes.append(f"{field}/{side}: dropped out-of-range indices {dropped.tolist()}")
            sets[side][reason].update(int(i) for i in valid)

    bad_amplitude: list[bool] = []
    bad_emg: list[bool] = []
    included: list[bool] = []
    exclusion_reason: list[str] = []
    for class_label, original_index in zip(epoched.class_label, epoched.original_trial_index):
        side = "left" if class_label == "left_hand" else "right"
        oi = int(original_index)
        a = oi in sets[side]["amplitude"]
        e = oi in sets[side]["emg_correlation"]
        o = oi in sets[side]["provided_flag"]
        reasons = [name for name, flag in
                   (("amplitude", a), ("emg_correlation", e), ("provided_flag", o)) if flag]
        bad_amplitude.append(a)
        bad_emg.append(e)
        included.append(len(reasons) == 0)
        exclusion_reason.append(";".join(reasons))

    return {
        "bad_amplitude": np.array(bad_amplitude, dtype=bool),
        "bad_emg": np.array(bad_emg, dtype=bool),
        "included": np.array(included, dtype=bool),
        "exclusion_reason": np.array(exclusion_reason, dtype=object),
        "notes": notes,
    }


def build_excluded_trials(
    epoched: EpochedSubject, flags: dict[str, Any], *, source: str = "provided_bad_trials"
) -> pd.DataFrame:
    """Rows (one per excluded trial) for the ``excluded_trials.csv`` manifest."""
    mask = ~flags["included"]
    if not np.any(mask):
        return pd.DataFrame(columns=EXCLUDED_TRIALS_COLUMNS)
    notes = "; ".join(flags.get("notes", []))
    df = pd.DataFrame({
        "subject_id": epoched.subject_id,
        "class_label": epoched.class_label[mask],
        "original_trial_index": epoched.original_trial_index[mask],
        "run_id": epoched.run_id[mask],
        "exclusion_reason": flags["exclusion_reason"][mask],
        "source": source,
        "notes": notes,
    })
    return df[EXCLUDED_TRIALS_COLUMNS]


def write_excluded_trials(df: pd.DataFrame, dest=None):
    dest = dest if dest is not None else paths.manifests_dir() / "excluded_trials.csv"
    from pathlib import Path

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest
