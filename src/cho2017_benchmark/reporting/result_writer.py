"""Schema-validated writers for experiment results.

All per-model result files go through these writers so the documented schemas are
enforced in one place. Aggregate metrics are computed on participant-level
(seed-averaged) scores, with pooled metrics labelled separately by the caller.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..evaluation.metrics import SUBJECT_METRICS_COLUMNS
from ..evaluation.predictions import PREDICTION_COLUMNS

AGGREGATE_METRIC_KEYS = ["accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa", "roc_auc"]


def assert_schema(df: pd.DataFrame, required: Iterable[str], name: str = "frame") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def write_subject_metrics(rows: list[dict[str, Any]] | pd.DataFrame, dest: Path | str) -> Path:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    assert_schema(df, SUBJECT_METRICS_COLUMNS, "subject_metrics")
    df = df[SUBJECT_METRICS_COLUMNS]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def write_predictions(frames: list[pd.DataFrame] | pd.DataFrame, dest: Path | str) -> Path:
    df = pd.concat(frames, ignore_index=True) if isinstance(frames, list) else frames.copy()
    assert_schema(df, PREDICTION_COLUMNS, "predictions")
    df = df[PREDICTION_COLUMNS]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return dest


def write_training_summary(rows: list[dict[str, Any]] | pd.DataFrame, dest: Path | str) -> Path:
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def write_selected_hyperparameters(rows: list[dict[str, Any]] | pd.DataFrame, dest: Path | str) -> Path:
    return write_training_summary(rows, dest)


def _stats(values: np.ndarray, *, n_boot: int = 10000, seed: int = 42, alpha: float = 0.05) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {k: float("nan") for k in
                ("mean", "std", "median", "iqr", "min", "max", "ci_low", "ci_high", "n")}
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "mean": float(values.mean()), "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)), "iqr": float(q3 - q1),
        "min": float(values.min()), "max": float(values.max()),
        "ci_low": float(np.percentile(boot, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "n": int(values.size),
    }


def aggregate_participant_metrics(
    per_subject: pd.DataFrame, *, metrics: list[str] | None = None, seed: int = 42
) -> dict[str, Any]:
    """Participant-level aggregate stats (mean/std/median/IQR/min/max/95% bootstrap CI)."""
    metrics = metrics or [m for m in AGGREGATE_METRIC_KEYS if m in per_subject.columns]
    return {
        "level": "participant",
        "n_participants": int(per_subject["subject_id"].nunique()) if "subject_id" in per_subject else len(per_subject),
        "metrics": {m: _stats(per_subject[m].to_numpy(), seed=seed) for m in metrics},
    }


def write_aggregate_metrics(
    per_subject: pd.DataFrame,
    dest: Path | str,
    *,
    pooled: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    seed: int = 42,
) -> Path:
    payload: dict[str, Any] = {"participant_level": aggregate_participant_metrics(per_subject, seed=seed)}
    if pooled is not None:
        payload["pooled_level"] = pooled  # explicitly labelled pooled metrics
    if extra:
        payload.update(extra)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest
