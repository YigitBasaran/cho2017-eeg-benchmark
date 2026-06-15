"""Per-subject classification metrics with a fixed, documented schema.

Undefined metrics (e.g. ROC AUC with a single class present in the test set)
are recorded as NaN with a reason rather than crashing or silently using 0.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Label convention: 0 = left_hand, 1 = right_hand (right = positive class).
LEFT, RIGHT = 0, 1

SUBJECT_METRICS_COLUMNS = [
    "model_name", "subject_id", "split_method", "n_train", "n_validation",
    "n_test", "accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa",
    "roc_auc", "left_precision", "left_recall", "left_f1", "right_precision",
    "right_recall", "right_f1", "tn", "fp", "fn", "tp", "best_epoch",
    "train_time_seconds", "inference_mean_ms", "inference_median_ms",
    "inference_p95_ms", "trainable_parameters", "status", "error_message",
    "seed", "split_manifest_hash", "preprocessing_hash",
]


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, str | None]:
    """ROC AUC using positive-class scores; NaN + reason when undefined."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return math.nan, "single_class_in_truth"
    try:
        return float(roc_auc_score(y_true, np.asarray(y_score))), None
    except Exception as exc:  # noqa: BLE001
        return math.nan, f"roc_auc_error:{exc}"


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)`` with the fixed label order [0, 1]."""
    cm = confusion_matrix(y_true, y_pred, labels=[LEFT, RIGHT])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _nan_metric_fields() -> dict[str, Any]:
    keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa", "roc_auc",
        "left_precision", "left_recall", "left_f1", "right_precision",
        "right_recall", "right_f1",
    ]
    out: dict[str, Any] = {k: math.nan for k in keys}
    out.update({"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    return out


def compute_subject_metrics(
    y_true: np.ndarray | None,
    y_pred: np.ndarray | None,
    y_prob: np.ndarray | None,
    *,
    model_name: str,
    subject_id: str,
    split_method: str,
    n_train: int,
    n_validation: int,
    n_test: int,
    best_epoch: int | float = math.nan,
    train_time_seconds: float = math.nan,
    inference: dict[str, float] | None = None,
    trainable_parameters: int | float = math.nan,
    seed: int = 42,
    split_manifest_hash: str = "",
    preprocessing_hash: str = "",
    status: str = "ok",
    error_message: str = "",
) -> dict[str, Any]:
    """Compute the full subject-metrics row (schema = SUBJECT_METRICS_COLUMNS)."""
    inference = inference or {}
    row: dict[str, Any] = {
        "model_name": model_name, "subject_id": subject_id, "split_method": split_method,
        "n_train": int(n_train), "n_validation": int(n_validation), "n_test": int(n_test),
        "best_epoch": best_epoch, "train_time_seconds": train_time_seconds,
        "inference_mean_ms": inference.get("mean_ms", math.nan),
        "inference_median_ms": inference.get("median_ms", math.nan),
        "inference_p95_ms": inference.get("p95_ms", math.nan),
        "trainable_parameters": trainable_parameters,
        "status": status, "error_message": error_message, "seed": int(seed),
        "split_manifest_hash": split_manifest_hash, "preprocessing_hash": preprocessing_hash,
    }

    if status != "ok" or y_true is None or y_pred is None or len(y_true) == 0:
        row.update(_nan_metric_fields())
        return {k: row.get(k) for k in SUBJECT_METRICS_COLUMNS}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prob_pos = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        prob_pos = y_prob[:, RIGHT] if y_prob.ndim == 2 else y_prob

    row["accuracy"] = float(accuracy_score(y_true, y_pred))
    row["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    row["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=[LEFT, RIGHT], zero_division=0))
    row["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred, labels=[LEFT, RIGHT]))
    row["roc_auc"] = (safe_roc_auc(y_true, prob_pos)[0] if prob_pos is not None else math.nan)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[LEFT, RIGHT], zero_division=0
    )
    row["left_precision"], row["right_precision"] = float(precision[0]), float(precision[1])
    row["left_recall"], row["right_recall"] = float(recall[0]), float(recall[1])
    row["left_f1"], row["right_f1"] = float(f1[0]), float(f1[1])

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    row["tn"], row["fp"], row["fn"], row["tp"] = tn, fp, fn, tp

    return {k: row.get(k) for k in SUBJECT_METRICS_COLUMNS}
