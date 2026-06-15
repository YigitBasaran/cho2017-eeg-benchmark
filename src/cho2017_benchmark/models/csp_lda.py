"""Experiment 1: CSP + LDA per-subject pipeline.

A small, interpretable validation search over CSP components / regularization /
LDA solver. Following the fair primary protocol (matching the neural models):
fit on train, select on validation, evaluate once on test -- no train+val refit.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ..evaluation.latency import measure_callable_latency
from ..evaluation.metrics import compute_subject_metrics
from ..evaluation.predictions import build_prediction_frame
from ..reproducibility import candidate_config_hash


def build_pipeline(n_components: int = 4, reg: Any = None, lda_solver: str = "svd"):
    """Build an MNE CSP -> scikit-learn LDA pipeline."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import Pipeline

    csp = CSP(n_components=n_components, reg=reg, log=True,
              transform_into="average_power", norm_trace=False)
    if lda_solver == "svd":
        lda = LinearDiscriminantAnalysis(solver="svd")
    elif lda_solver == "lsqr_shrinkage":
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(f"Unknown lda_solver '{lda_solver}'.")
    return Pipeline([("csp", csp), ("lda", lda)])


def candidate_grid(cfg) -> list[dict[str, Any]]:
    n_components = cfg.get("csp.n_components", [4, 6, 8])
    regs = cfg.get("csp.reg", [None, "ledoit_wolf"])
    solvers = cfg.get("lda.solver", ["svd", "lsqr_shrinkage"])
    grid = []
    for nc in n_components:
        for rg in regs:
            for sv in solvers:
                grid.append({"id": f"nc{nc}_reg{rg}_lda{sv}",
                             "n_components": int(nc), "reg": rg, "lda_solver": sv})
    return grid


def run_subject_csp(
    cfg,
    X: np.ndarray,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    *,
    subject_id: str,
    split_method: str,
    split_manifest_hash: str,
    preprocessing_hash: str,
    split_meta_test: pd.DataFrame,
    model_path: str = "",
) -> dict[str, Any]:
    """Select hyperparameters on validation, fit on train, evaluate once on test."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    tr, va, te = split_indices["train"], split_indices["val"], split_indices["test"]
    grid = candidate_grid(cfg)

    records: list[dict[str, Any]] = []
    best, best_score = grid[0], -np.inf
    for cand in grid:
        pipe = build_pipeline(cand["n_components"], cand["reg"], cand["lda_solver"])
        pipe.fit(X[tr], y[tr])
        score = float(balanced_accuracy_score(y[va], pipe.predict(X[va]))) if len(va) else np.nan
        records.append({"subject_id": subject_id, "candidate_id": cand["id"],
                        "val_balanced_accuracy": score,
                        "candidate_config_hash": candidate_config_hash(cand)})
        if score > best_score:
            best_score, best = score, cand

    # Final fit on TRAIN ONLY (no train+val refit in the primary protocol).
    start = time.perf_counter()
    final = build_pipeline(best["n_components"], best["reg"], best["lda_solver"])
    final.fit(X[tr], y[tr])
    train_time = time.perf_counter() - start

    y_pred = final.predict(X[te])
    y_prob = final.predict_proba(X[te])
    latency = measure_callable_latency(lambda a: final.predict(a), X[te][:1], device="cpu")
    params = int(final.named_steps["csp"].filters_.size + final.named_steps["lda"].coef_.size)

    row = compute_subject_metrics(
        y[te], y_pred, y_prob, model_name="csp_lda", subject_id=subject_id,
        split_method=split_method, n_train=len(tr), n_validation=len(va), n_test=len(te),
        best_epoch=np.nan, train_time_seconds=train_time,
        inference=latency.to_inference_dict(), trainable_parameters=params,
        seed=cfg.seed, split_manifest_hash=split_manifest_hash,
        preprocessing_hash=preprocessing_hash,
    )
    predictions = build_prediction_frame(
        model_name="csp_lda", subject_id=subject_id, split_meta=split_meta_test,
        y_pred=y_pred, y_prob=y_prob, split="test", seed=cfg.seed, checkpoint_path=model_path,
    )
    selected = {
        "subject_id": subject_id, **{k: best[k] for k in ("id", "n_components", "reg", "lda_solver")},
        "selected_candidate": best["id"], "candidate_config_hash": candidate_config_hash(best),
        "n_candidates_evaluated": len(grid), "tuning_seed": cfg.tuning_seed,
        "val_balanced_accuracy": best_score,
    }
    return {"metrics": row, "predictions": predictions, "selected": selected,
            "candidates": records, "model": final,
            "patterns": final.named_steps["csp"].patterns_,
            "filters": final.named_steps["csp"].filters_}
