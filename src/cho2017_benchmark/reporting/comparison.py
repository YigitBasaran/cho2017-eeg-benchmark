"""Cross-model loading, consistency checks and comparison tables.

Used by the final comparison notebook, which must verify all models were
evaluated under identical conditions before comparing them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import paths

MODEL_NAMES = ["csp_lda", "eegnet", "atcnet"]


@dataclass
class ModelResults:
    model_name: str
    subject_metrics: pd.DataFrame
    predictions: pd.DataFrame
    aggregate: dict[str, Any]


def load_model_results(model_name: str) -> ModelResults:
    base = paths.results_dir(model_name)
    metrics = pd.read_csv(base / "tables" / "subject_metrics.csv")
    pred_path = base / "predictions" / "test_predictions.parquet"
    predictions = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
    agg_path = base / "metrics" / "aggregate_metrics.json"
    aggregate = json.loads(agg_path.read_text(encoding="utf-8")) if agg_path.exists() else {}
    return ModelResults(model_name, metrics, predictions, aggregate)


def _unique(series: pd.Series) -> set:
    return set(series.dropna().unique().tolist())


def cross_check_consistency(results: dict[str, ModelResults]) -> None:
    """Raise if models used different splits / subjects / test trials / hashes."""
    errors: list[str] = []
    hashes, subjects, test_ids, preproc = {}, {}, {}, {}
    for name, res in results.items():
        sm = res.subject_metrics
        if "split_manifest_hash" in sm:
            hashes[name] = _unique(sm["split_manifest_hash"])
        if "preprocessing_hash" in sm:
            preproc[name] = _unique(sm["preprocessing_hash"])
        subjects[name] = _unique(sm["subject_id"])
        if not res.predictions.empty:
            test = res.predictions[res.predictions["split"] == "test"]
            test_ids[name] = set(test["trial_id"].unique().tolist())

    hash_values = {next(iter(v)) for v in hashes.values() if len(v) == 1}
    if len({tuple(sorted(v)) for v in hashes.values()}) > 1:
        errors.append(f"split_manifest_hash differs across models: {hashes}")
    if len({frozenset(v) for v in subjects.values()}) > 1:
        errors.append("Included subject sets differ across models.")
    if test_ids and len({frozenset(v) for v in test_ids.values()}) > 1:
        errors.append("Evaluated test trial ids differ across models.")
    if len({frozenset(v) for v in preproc.values()}) > 1:
        errors.append(f"preprocessing_hash differs across models: {preproc}")

    if errors:
        raise ValueError("Cross-model consistency checks FAILED:\n  - " + "\n  - ".join(errors))


def build_model_summary_table(results: dict[str, ModelResults], metric: str = "accuracy") -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        part = res.aggregate.get("participant_level", {}).get("metrics", {}).get(metric, {})
        rows.append({"model": name, **{f"{metric}_{k}": v for k, v in part.items()}})
    return pd.DataFrame(rows)


def build_subject_accuracy_table(results: dict[str, ModelResults], metric: str = "accuracy") -> pd.DataFrame:
    """Subject x model table of seed-averaged *metric*."""
    frames = []
    for name, res in results.items():
        sm = res.subject_metrics
        ok = sm[sm.get("status", "ok") == "ok"] if "status" in sm else sm
        agg = ok.groupby("subject_id")[metric].mean().rename(name)
        frames.append(agg)
    return pd.concat(frames, axis=1).reset_index()


def common_hard_subjects(
    results: dict[str, ModelResults], *, threshold: float = 0.6, metric: str = "accuracy"
) -> list[str]:
    table = build_subject_accuracy_table(results, metric).set_index("subject_id")
    hard = table[(table < threshold).all(axis=1)]
    return hard.index.tolist()


def model_rank_per_subject(results: dict[str, ModelResults], metric: str = "accuracy") -> pd.DataFrame:
    table = build_subject_accuracy_table(results, metric).set_index("subject_id")
    ranks = table.rank(axis=1, ascending=False, method="min")
    return ranks.reset_index()
