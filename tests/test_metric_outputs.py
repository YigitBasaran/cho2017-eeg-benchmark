import math

import numpy as np

from cho2017_benchmark.evaluation.metrics import (
    SUBJECT_METRICS_COLUMNS,
    compute_subject_metrics,
    confusion_counts,
    safe_roc_auc,
)


def test_metric_schema_and_values():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_prob = np.tile([[0.4, 0.6]], (6, 1))
    row = compute_subject_metrics(
        y_true, y_pred, y_prob, model_name="eegnet", subject_id="s01",
        split_method="run_holdout", n_train=10, n_validation=3, n_test=6,
        seed=42, split_manifest_hash="h", preprocessing_hash="p",
    )
    assert set(row) == set(SUBJECT_METRICS_COLUMNS)
    assert 0.0 <= row["accuracy"] <= 1.0
    assert row["tn"] + row["fp"] + row["fn"] + row["tp"] == 6


def test_safe_roc_auc_single_class():
    val, reason = safe_roc_auc(np.array([1, 1, 1]), np.array([0.5, 0.6, 0.7]))
    assert math.isnan(val) and reason == "single_class_in_truth"


def test_confusion_counts_label_order():
    tn, fp, fn, tp = confusion_counts(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)


def test_failed_subject_is_nan():
    row = compute_subject_metrics(
        None, None, None, model_name="atcnet", subject_id="s10", split_method="",
        n_train=0, n_validation=0, n_test=0, status="failed", error_message="boom",
        seed=42, split_manifest_hash="", preprocessing_hash="",
    )
    assert row["status"] == "failed" and math.isnan(row["accuracy"])
    assert set(row) == set(SUBJECT_METRICS_COLUMNS)
