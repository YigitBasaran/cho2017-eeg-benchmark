"""Per-trial test predictions with a fixed schema.

The unique key of a prediction row is
``model_name + subject_id + seed + trial_id``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = [
    "model_name", "subject_id", "trial_id", "original_trial_index", "run_id",
    "true_label", "predicted_label", "probability_left", "probability_right",
    "split", "seed", "checkpoint_path",
]


def build_prediction_frame(
    *,
    model_name: str,
    subject_id: str,
    split_meta: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    split: str = "test",
    seed: int = 42,
    checkpoint_path: str = "",
) -> pd.DataFrame:
    """Assemble a prediction frame from ordered split metadata and outputs.

    *split_meta* must be in the same order as *y_pred* / *y_prob* and contain
    ``trial_id, original_trial_index, run_id, true_label``.
    """
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:  # positive-class prob only
        prob_right = y_prob
        prob_left = 1.0 - y_prob
    else:
        prob_left = y_prob[:, 0]
        prob_right = y_prob[:, 1]

    df = pd.DataFrame({
        "model_name": model_name,
        "subject_id": subject_id,
        "trial_id": np.asarray(split_meta["trial_id"]),
        "original_trial_index": np.asarray(split_meta["original_trial_index"]),
        "run_id": np.asarray(split_meta["run_id"]),
        "true_label": np.asarray(split_meta["true_label"]),
        "predicted_label": np.asarray(y_pred),
        "probability_left": prob_left,
        "probability_right": prob_right,
        "split": split,
        "seed": int(seed),
        "checkpoint_path": checkpoint_path,
    })
    return df[PREDICTION_COLUMNS]
