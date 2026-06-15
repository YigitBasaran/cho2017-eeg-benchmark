"""Per-subject neural experiment driver (EEGNet / ATCNet).

Implements the agreed protocol so notebooks 02/03 stay thin:

1. Evaluate every candidate config with ``tuning_seed`` on the validation set
   only; pick exactly one candidate per participant (rejected candidates never
   touch the test set).
2. Train the selected candidate with each training seed (e.g. 42, 43, 44).
3. Evaluate the best-validation checkpoint once on the test set per seed.

Per-seed results are preserved (one metrics row / prediction frame / history /
checkpoint per seed). EEGNet and ATCNet evaluate the same number of candidates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import ExperimentConfig, deep_merge
from ..data.datasets import SubjectSplitData
from ..evaluation.latency import measure_latency
from ..evaluation.metrics import compute_subject_metrics
from ..evaluation.predictions import build_prediction_frame
from ..models import factory
from ..reproducibility import candidate_config_hash, set_seed
from . import checkpoints
from .trainer import Trainer, resolve_device

_TRAINING_OVERRIDE_KEYS = {"lr", "weight_decay", "grad_clip"}


def _candidate_cfg(cfg: ExperimentConfig, candidate: dict[str, Any]) -> ExperimentConfig:
    """Return a config variant with the candidate's overrides applied."""
    model_over = {k: v for k, v in candidate.items()
                  if k != "id" and k not in _TRAINING_OVERRIDE_KEYS}
    train_over = {k: v for k, v in candidate.items() if k in _TRAINING_OVERRIDE_KEYS}
    overrides: dict[str, Any] = {}
    if model_over:
        overrides["model"] = {"params": model_over}
    if train_over:
        overrides["training"] = train_over
    return ExperimentConfig(deep_merge(cfg.raw, overrides))


def _make_trainer(model, ccfg: ExperimentConfig, seed: int, device: str,
                  class_weights: np.ndarray | None = None) -> Trainer:
    tc = dict(ccfg.get("training", {}) or {})
    return Trainer(
        model, device=device, optimizer_cfg=tc,
        scheduler_cfg=tc.get("scheduler", {"name": "reduce_on_plateau"}),
        max_epochs=int(tc.get("max_epochs", 200)), patience=int(tc.get("patience", 25)),
        monitor=str(tc.get("monitor", "val_balanced_accuracy")),
        monitor_mode=str(tc.get("monitor_mode", "max")),
        grad_clip=tc.get("grad_clip"), mixed_precision=bool(tc.get("mixed_precision", False)),
        seed=seed, class_weights=class_weights,
    )


def compute_class_weights(y: np.ndarray, threshold: float = 0.10) -> np.ndarray | None:
    """Inverse-frequency train-only weights, only if class imbalance > threshold."""
    counts = np.bincount(np.asarray(y), minlength=2).astype(float)
    if counts.sum() == 0:
        return None
    frac = counts / counts.sum()
    if abs(frac[0] - frac[1]) > threshold:
        return (counts.sum() / (2.0 * np.maximum(counts, 1))).astype(np.float32)
    return None


def run_subject_neural(
    model_name: str,
    cfg: ExperimentConfig,
    subject_data: SubjectSplitData,
    *,
    split_method: str,
    split_manifest_hash: str,
    preprocessing_hash: str,
    device: str | None = None,
    n_outputs: int = 2,
) -> dict[str, Any]:
    """Run the full candidate-selection + multi-seed protocol for one subject."""
    device = device or resolve_device(cfg.device)
    sid = subject_data.subject_id
    n_chans, n_times = subject_data.n_channels, subject_data.n_times
    candidates = list(cfg.get("candidates", []) or [{"id": "default"}])
    tuning_seed = cfg.tuning_seed
    monitor_mode = str(cfg.get("training.monitor_mode", "max"))

    n_train = len(subject_data.split_meta["train"])
    n_val = len(subject_data.split_meta["val"])
    n_test = len(subject_data.split_meta["test"])

    # class weights from training labels only
    threshold = float(cfg.get("class_balance.imbalance_threshold", 0.10))
    use_weights = bool(cfg.get("class_balance.use_weights_default", False))
    cw = compute_class_weights(subject_data.split_meta["train"]["true_label"].to_numpy(), threshold)
    class_weights = cw if (use_weights or cw is not None) else None

    # --- 1. candidate selection (tuning_seed, validation only) ---
    candidate_records: list[dict[str, Any]] = []
    best_candidate, best_val = candidates[0], (-np.inf if monitor_mode == "max" else np.inf)
    for candidate in candidates:
        ccfg = _candidate_cfg(cfg, candidate)
        set_seed(tuning_seed)
        model, _ = factory.build_model(model_name, ccfg, n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
        trainer = _make_trainer(model, ccfg, tuning_seed, device, class_weights)
        result = trainer.fit(subject_data.train_loader, subject_data.val_loader)
        candidate_records.append({
            "subject_id": sid, "candidate_id": candidate.get("id", "default"),
            "val_metric": result.best_val_metric,
            "candidate_config_hash": candidate_config_hash(candidate),
        })
        better = (result.best_val_metric > best_val if monitor_mode == "max"
                  else result.best_val_metric < best_val)
        if better:
            best_val, best_candidate = result.best_val_metric, candidate
    n_candidates = len(candidates)

    # --- 2 & 3. final training per seed + test evaluation ---
    metrics_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        sccfg = _candidate_cfg(cfg, best_candidate)
        set_seed(seed)
        model, info = factory.build_model(model_name, sccfg, n_chans=n_chans, n_times=n_times, n_outputs=n_outputs)
        params = factory.count_trainable_parameters(model)
        trainer = _make_trainer(model, sccfg, seed, device, class_weights)
        result = trainer.fit(subject_data.train_loader, subject_data.val_loader)

        ckpt = checkpoints.checkpoint_path(model_name, sid, seed)
        checkpoints.save_checkpoint(
            ckpt, model=model, normalization=subject_data.stats.to_dict(),
            meta={
                "model_config": info, "class_mapping": cfg.class_mapping,
                "sfreq": cfg.srate_out, "channel_names": None,
                "time_window": [cfg.get("window.crop_start_s"), cfg.get("window.crop_end_s")],
                "best_epoch": result.best_epoch, "val_metric": result.best_val_metric,
                "seed": seed, "subject_id": sid, "model_source": info.get("source"),
                "candidate_id": best_candidate.get("id", "default"),
                "split_manifest_hash": split_manifest_hash,
                "preprocessing_hash": preprocessing_hash,
            },
        )
        result.history.to_csv(checkpoints.history_path(model_name, sid, seed), index=False)

        y_true, y_pred, y_prob = trainer.evaluate(subject_data.test_loader)
        latency = measure_latency(model, subject_data.sample_input(), device=device)

        metrics_rows.append(compute_subject_metrics(
            y_true, y_pred, y_prob, model_name=model_name, subject_id=sid,
            split_method=split_method, n_train=n_train, n_validation=n_val, n_test=n_test,
            best_epoch=result.best_epoch, train_time_seconds=result.train_time_seconds,
            inference=latency.to_inference_dict(), trainable_parameters=params, seed=seed,
            split_manifest_hash=split_manifest_hash, preprocessing_hash=preprocessing_hash,
        ))
        prediction_frames.append(build_prediction_frame(
            model_name=model_name, subject_id=sid, split_meta=subject_data.split_meta["test"],
            y_pred=y_pred, y_prob=y_prob, split="test", seed=seed, checkpoint_path=str(ckpt),
        ))
        summary_rows.append({
            "subject_id": sid, "seed": seed, "best_epoch": result.best_epoch,
            "train_time_seconds": result.train_time_seconds,
            "val_metric": result.best_val_metric, "n_candidates_evaluated": n_candidates,
            "selected_candidate": best_candidate.get("id", "default"),
            "candidate_config_hash": candidate_config_hash(best_candidate),
            "tuning_seed": tuning_seed, "model_source": info.get("source"),
            "trainable_parameters": params, "device": device,
            "class_weights_used": class_weights is not None,
            "latency_device": latency.device,
        })

    return {
        "metrics": metrics_rows,
        "predictions": prediction_frames,
        "training_summary": summary_rows,
        "candidates": candidate_records,
        "selected_candidate": best_candidate.get("id", "default"),
        "model_source": summary_rows[0]["model_source"] if summary_rows else None,
    }
