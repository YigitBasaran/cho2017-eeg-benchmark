"""Reproducibility utilities: seeding, deterministic hashing, environment capture.

A single :func:`set_seed` configures every relevant RNG. A single
:func:`stable_hash` canonicaliser backs all of the named hashes
(:func:`preprocessing_hash`, :func:`split_manifest_hash`,
:func:`model_config_hash`, :func:`candidate_config_hash`) so identical inputs
always hash identically, across processes and machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
_FLOAT_DECIMALS = 10


# --- Seeding ---------------------------------------------------------------


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if available) PyTorch CPU + CUDA.

    When *deterministic* is True we additionally request deterministic
    algorithms from cuDNN/PyTorch. Strict determinism can measurably slow
    training (cuDNN cannot pick the fastest non-deterministic kernels) and a
    few ops have no deterministic implementation; we therefore use
    ``warn_only=True`` so those ops warn instead of raising.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Required for deterministic CUDA matmul on some CUDA versions.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # older torch without warn_only
                torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        # PyTorch is optional for the non-neural parts of the pipeline.
        pass


# --- Deterministic hashing -------------------------------------------------


def _canonical(obj: Any) -> Any:
    """Recursively convert *obj* into a JSON-serialisable canonical form."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return round(obj, _FLOAT_DECIMALS)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), _FLOAT_DECIMALS)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape),
                "dtype": str(obj.dtype), "data": _canonical(obj.tolist())}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return [[str(k), _canonical(obj[k])] for k in sorted(obj, key=str)]
    if isinstance(obj, (set, frozenset)):
        return sorted((_canonical(v) for v in obj), key=lambda x: json.dumps(x, default=str))
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        ordered = obj.sort_values(list(obj.columns)).reset_index(drop=True)
        return {"__dataframe__": True, "columns": list(map(str, obj.columns)),
                "records": [_canonical(rec) for rec in ordered.to_dict("records")]}
    if isinstance(obj, pd.Series):
        return _canonical(obj.tolist())
    if is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(asdict(obj))
    if hasattr(obj, "hashable") and callable(obj.hashable):
        return _canonical(obj.hashable())
    return str(obj)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Return a deterministic short hex digest of an arbitrary object."""
    payload = json.dumps(_canonical(obj), separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]


def preprocessing_hash(preprocessing: Any, length: int = 16) -> str:
    return stable_hash(preprocessing, length=length)


def model_config_hash(model_config: Any, length: int = 16) -> str:
    return stable_hash(model_config, length=length)


def candidate_config_hash(candidate_config: Any, length: int = 16) -> str:
    return stable_hash(candidate_config, length=length)


_SPLIT_IDENTITY_COLUMNS = [
    "subject_id", "trial_id", "class_label", "original_trial_index",
    "run_id", "run_id_source", "included", "split", "split_method",
]


def split_manifest_hash(manifest: pd.DataFrame, length: int = 16) -> str:
    """Hash the split assignment of a manifest (identity columns only).

    Only columns that define the split identity are hashed, so cosmetic column
    additions do not change the hash while any change to trial-to-split
    assignment does.
    """
    cols = [c for c in _SPLIT_IDENTITY_COLUMNS if c in manifest.columns]
    subset = manifest[cols].copy()
    return stable_hash(subset, length=length)


# --- Environment capture ---------------------------------------------------

_TRACKED_PACKAGES = [
    "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "mne", "moabb",
    "torch", "braindecode", "pyyaml", "pyarrow", "joblib", "tqdm", "requests",
]


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # pragma: no cover - defensive
        return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def collect_environment() -> dict[str, Any]:
    """Collect a machine-readable description of the execution environment."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": _git_commit(),
        "packages": {name: _package_version(name) for name in _TRACKED_PACKAGES},
        "gpu": None,
        "cuda": None,
        "cudnn": None,
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            cudnn = getattr(torch.backends, "cudnn", None)
            info["cudnn"] = cudnn.version() if cudnn is not None else None
    except ImportError:
        info["torch_cuda_available"] = False
    return info


def save_environment(dest_dir: Path | str, filename: str = "environment.json") -> Path:
    """Write :func:`collect_environment` output as JSON under *dest_dir*."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    path.write_text(json.dumps(collect_environment(), indent=2), encoding="utf-8")
    return path
