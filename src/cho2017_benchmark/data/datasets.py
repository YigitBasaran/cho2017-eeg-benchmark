"""PyTorch dataset and per-subject loaders with train-only normalization.

Normalization statistics are computed from the *training* trials only and then
frozen and applied to validation and test (and saved with the checkpoint), so no
validation/test information ever leaks into the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class NormalizationStats:
    mean: np.ndarray  # (n_channels, 1)
    std: np.ndarray   # (n_channels, 1)
    eps: float = 1e-7

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.astype(np.float32),
                "std": self.std.astype(np.float32), "eps": float(self.eps)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NormalizationStats":
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32), eps=float(d["eps"]))


def compute_normalization(X_train: np.ndarray, eps: float = 1e-7) -> NormalizationStats:
    """Per-channel mean/std over all training trials and time samples.

    *X_train* has shape ``(n_trials, n_channels, n_times)``.
    """
    if X_train.ndim != 3:
        raise ValueError(f"Expected (n_trials, n_channels, n_times), got {X_train.shape}.")
    mean = X_train.mean(axis=(0, 2), keepdims=False).reshape(-1, 1)
    std = X_train.std(axis=(0, 2), keepdims=False).reshape(-1, 1)
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32), eps=eps)


def apply_normalization(X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply frozen per-channel normalization: ``(X - mean) / (std + eps)``."""
    return ((X - stats.mean) / (stats.std + stats.eps)).astype(np.float32)


class Cho2017TrialDataset(Dataset):
    """Tensor dataset of ``(n_trials, n_channels, n_times)`` EEG and integer labels."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        stats: NormalizationStats | None = None,
        trial_ids: np.ndarray | None = None,
    ) -> None:
        if stats is not None:
            X = apply_normalization(X, stats)
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self.trial_ids = np.asarray(trial_ids) if trial_ids is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


@dataclass
class SubjectSplitData:
    subject_id: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    stats: NormalizationStats
    split_meta: dict[str, pd.DataFrame]  # split -> (trial_id, original_trial_index, run_id, true_label)
    n_channels: int
    n_times: int

    def sample_input(self) -> torch.Tensor:
        """A single (1, n_channels, n_times) tensor for latency / shape checks."""
        x, _ = self.test_loader.dataset[0]
        return x.unsqueeze(0)


def subject_split_indices(
    record: dict[str, Any], split_manifest: pd.DataFrame, subject_id: str
) -> dict[str, np.ndarray]:
    """Map this subject's included trials to train/val/test positional indices."""
    trial_ids = np.asarray(record["trial_id"]).astype(str)
    included = np.asarray(record["included"]).astype(bool)
    pos_by_trial = {tid: i for i, tid in enumerate(trial_ids)}

    sub = split_manifest[split_manifest["subject_id"] == subject_id]
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for _, row in sub.iterrows():
        if not row["included"]:
            continue
        split = row["split"]
        if split not in out:
            continue
        pos = pos_by_trial.get(str(row["trial_id"]))
        if pos is not None and included[pos]:
            out[split].append(pos)
    return {k: np.array(v, dtype=int) for k, v in out.items()}


def make_subject_loaders(
    subject_id: str,
    record: dict[str, Any],
    split_manifest: pd.DataFrame,
    *,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> SubjectSplitData:
    """Build train/val/test loaders for one subject with train-only normalization."""
    X = np.asarray(record["X_eeg"], dtype=np.float32)
    y = np.asarray(record["y"], dtype=np.int64)
    trial_ids = np.asarray(record["trial_id"]).astype(str)
    original_idx = np.asarray(record["original_trial_index"])
    run_id = np.asarray(record["run_id"])
    eps = 1e-7

    indices = subject_split_indices(record, split_manifest, subject_id)
    if len(indices["train"]) == 0 or len(indices["test"]) == 0:
        raise ValueError(f"{subject_id}: empty train or test split.")

    stats = compute_normalization(X[indices["train"]], eps=eps)

    split_meta: dict[str, pd.DataFrame] = {}
    loaders: dict[str, DataLoader] = {}
    generator = torch.Generator().manual_seed(seed)
    for split in ("train", "val", "test"):
        idx = indices[split]
        ds = Cho2017TrialDataset(X[idx], y[idx], stats=stats, trial_ids=trial_ids[idx])
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, drop_last=False,
            generator=generator if split == "train" else None,
        )
        split_meta[split] = pd.DataFrame({
            "trial_id": trial_ids[idx],
            "original_trial_index": original_idx[idx],
            "run_id": run_id[idx],
            "true_label": y[idx],
        })

    return SubjectSplitData(
        subject_id=subject_id, train_loader=loaders["train"], val_loader=loaders["val"],
        test_loader=loaders["test"], stats=stats, split_meta=split_meta,
        n_channels=X.shape[1], n_times=X.shape[2],
    )


def subject_split_meta(record: dict[str, Any], indices: dict[str, np.ndarray]) -> dict[str, pd.DataFrame]:
    """Per-split metadata frames (trial_id, original_trial_index, run_id, true_label).

    Reusable by the CSP+LDA experiment (which does not need normalized loaders).
    """
    trial_ids = np.asarray(record["trial_id"]).astype(str)
    original_idx = np.asarray(record["original_trial_index"])
    run_id = np.asarray(record["run_id"])
    y = np.asarray(record["y"], dtype=np.int64)
    out: dict[str, pd.DataFrame] = {}
    for split, idx in indices.items():
        out[split] = pd.DataFrame({
            "trial_id": trial_ids[idx], "original_trial_index": original_idx[idx],
            "run_id": run_id[idx], "true_label": y[idx],
        })
    return out
