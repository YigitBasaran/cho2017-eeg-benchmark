"""Seed-aware checkpoint IO.

Checkpoints live at ``results/<model>/checkpoints/<subject_id>/seed_<seed>/best.pt``
and embed everything needed to reproduce an evaluation: the model state, resolved
model config, frozen normalization statistics, class mapping, sampling frequency,
channel names, input time window, best epoch, validation metric, seed, subject id,
model source and the split/preprocessing hashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths

CHECKPOINT_KEYS = [
    "state_dict", "model_config", "normalization", "class_mapping", "sfreq",
    "channel_names", "time_window", "best_epoch", "val_metric", "seed",
    "subject_id", "model_source", "candidate_id", "split_manifest_hash",
    "preprocessing_hash",
]


def checkpoint_path(model_name: str, subject_id: str, seed: int) -> Path:
    return paths.results_dir(model_name) / "checkpoints" / subject_id / f"seed_{seed}" / "best.pt"


def history_path(model_name: str, subject_id: str, seed: int) -> Path:
    return paths.results_dir(model_name) / "histories" / f"{subject_id}_seed_{seed}.csv"


def save_checkpoint(path: Path | str, *, model, normalization: dict, meta: dict[str, Any]) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "normalization": normalization}
    payload.update(meta)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path | str, map_location: str = "cpu") -> dict[str, Any]:
    import torch

    return torch.load(Path(path), map_location=map_location, weights_only=False)


def checkpoint_size_bytes(path: Path | str) -> int:
    p = Path(path)
    return int(p.stat().st_size) if p.exists() else 0
