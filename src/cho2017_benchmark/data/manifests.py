"""The trial-level manifest.

``trial_manifest.csv`` is the complete record of *every* original trial (included
and excluded) and is built BEFORE bad-trial removal and BEFORE split generation.
Split assignment derives from this manifest; model loaders later select
``included == True`` rows only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .. import paths

TRIAL_MANIFEST_COLUMNS = [
    "subject_id", "trial_id", "class_label", "original_trial_index",
    "event_sample", "run_id", "run_id_source", "included",
    "bad_amplitude", "bad_emg", "exclusion_reason",
]


def _column(record: Mapping[str, Any], key: str, n: int) -> np.ndarray:
    """Fetch *key* from *record* as a length-*n* array (broadcasting scalars)."""
    value = record[key]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.repeat(arr, n)
    return arr


def build_trial_manifest(record: Mapping[str, Any]) -> pd.DataFrame:
    """Build a trial manifest from a per-subject record (e.g. a loaded ``.npz``).

    *record* must provide every column in :data:`TRIAL_MANIFEST_COLUMNS`
    (``subject_id`` and ``run_id_source`` may be scalars).
    """
    n = int(np.asarray(record["trial_id"]).shape[0])
    data = {col: _column(record, col, n) for col in TRIAL_MANIFEST_COLUMNS}
    return pd.DataFrame(data)[TRIAL_MANIFEST_COLUMNS]


def write_trial_manifest(df: pd.DataFrame, dest: Path | str | None = None) -> Path:
    dest = Path(dest) if dest is not None else paths.manifests_dir() / "trial_manifest.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest
