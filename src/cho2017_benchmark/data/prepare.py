"""Build the per-subject processed cache (``data/processed/common/sNN.npz``).

Each ``.npz`` stores the model-ready EEG (CAR + 8-30 Hz + cropped + resampled to
64 x 512 @ 256 Hz) for *all* trials, plus raw EMG for EDA, labels, trial/run ids,
artifact flags and the preprocessing hash. The cache is idempotent (skips valid
files unless ``force=True``) and **gated**: it refuses to run without a present,
validated ``data/metadata/mat_layout_resolution.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .. import paths
from ..reproducibility import preprocessing_hash
from . import artifacts, epoching, loader, preprocessing
from .download import subject_id_str
from .inspect_mat import load_layout_resolution

PROCESSED_ARRAY_KEYS = [
    "X_eeg", "X_emg", "y", "trial_id", "class_label", "original_trial_index",
    "event_sample", "run_id", "included", "bad_amplitude", "bad_emg",
    "exclusion_reason", "class_names", "channel_names_eeg", "channel_names_emg",
]
PROCESSED_SCALAR_KEYS = [
    "run_id_source", "sampling_rate", "sampling_rate_emg", "epoch_tmin_s",
    "epoch_tmax_s", "subject_id", "preprocessing_json", "preprocessing_hash",
]


def is_processed_valid(path: Path | str, pp: preprocessing.PreprocessingConfig) -> bool:
    """True if *path* exists and was produced with the same preprocessing hash."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=True) as data:
            return str(data["preprocessing_hash"]) == preprocessing_hash(pp.hashable())
    except Exception:
        return False


def prepare_subject(
    index: int,
    *,
    cfg: Any,
    pp: preprocessing.PreprocessingConfig | None = None,
    resolution: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare one subject's processed cache file. Returns a status dict."""
    pp = pp or preprocessing.PreprocessingConfig.from_config(cfg)
    resolution = resolution or load_layout_resolution()  # hard gate
    sid = subject_id_str(index)
    dest = paths.subject_processed_path(sid)
    raw_path = paths.raw_dir() / f"{sid}.mat"

    if not raw_path.exists():
        return {"subject_id": sid, "status": "missing_raw", "path": str(dest),
                "n_trials": 0, "n_included": 0, "run_id_source": ""}
    if not force and is_processed_valid(dest, pp):
        with np.load(dest, allow_pickle=True) as d:
            n_trials = int(np.asarray(d["y"]).shape[0])
            n_included = int(np.asarray(d["included"]).sum())
            src = str(d["run_id_source"])
        return {"subject_id": sid, "status": "skipped", "path": str(dest),
                "n_trials": n_trials, "n_included": n_included, "run_id_source": src}

    rec = loader.load_subject(index, resolution=resolution)
    ep = epoching.epoch_subject(
        rec,
        tmin_s=float(cfg.get("epoch.tmin_s", -1.5)),
        tmax_s=float(cfg.get("epoch.tmax_s", 4.0)),
        class_mapping=cfg.class_mapping,
        trials_per_run=int(cfg.get("runs.trials_per_run", 20)),
        valid_trials_per_class=cfg.get("runs.valid_trials_per_class", [100, 120]),
        valid_n_runs=cfg.get("runs.valid_n_runs", [5, 6]),
        resolution=resolution,
    )
    flags = artifacts.compute_trial_flags(rec, ep, resolution)
    X_eeg = preprocessing.preprocess_epochs(ep, pp, resolution)

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest,
        X_eeg=X_eeg,
        X_emg=ep.X_emg.astype(np.float32),
        y=ep.y,
        trial_id=ep.trial_id.astype(str),
        class_label=ep.class_label.astype(str),
        original_trial_index=ep.original_trial_index,
        event_sample=ep.event_sample,
        run_id=ep.run_id,
        included=flags["included"],
        bad_amplitude=flags["bad_amplitude"],
        bad_emg=flags["bad_emg"],
        exclusion_reason=flags["exclusion_reason"].astype(str),
        class_names=np.array(ep.class_names, dtype=str),
        channel_names_eeg=np.array(loader.CHO2017_EEG_CHANNELS, dtype=str),
        channel_names_emg=np.array(loader.CHO2017_EMG_CHANNELS, dtype=str),
        run_id_source=np.array(ep.run_id_source),
        sampling_rate=np.array(pp.srate_out),
        sampling_rate_emg=np.array(int(ep.srate)),
        epoch_tmin_s=np.array(ep.epoch_tmin_s),
        epoch_tmax_s=np.array(ep.epoch_tmax_s),
        subject_id=np.array(sid),
        preprocessing_json=np.array(json.dumps(pp.hashable())),
        preprocessing_hash=np.array(preprocessing_hash(pp.hashable())),
    )
    return {"subject_id": sid, "status": "ok", "path": str(dest),
            "n_trials": int(ep.n_trials), "n_included": int(flags["included"].sum()),
            "run_id_source": ep.run_id_source}


def prepare_all(
    cfg: Any,
    *,
    subjects: Sequence[int] | None = None,
    pp: preprocessing.PreprocessingConfig | None = None,
    force: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Prepare all (or given) subjects. Processes excluded subjects too (for EDA)."""
    pp = pp or preprocessing.PreprocessingConfig.from_config(cfg)
    resolution = load_layout_resolution()  # gate once
    if subjects is None:
        subjects = cfg.quick_subjects if cfg.quick_mode else cfg.all_subjects
    subjects = list(subjects)

    iterator = subjects
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(subjects, desc="prepare", unit="subj")
        except ImportError:
            pass

    rows = []
    for index in iterator:
        rows.append(prepare_subject(index, cfg=cfg, pp=pp, resolution=resolution, force=force))
    return pd.DataFrame(rows)


def write_dataset_metadata(
    cfg: Any, pp: preprocessing.PreprocessingConfig, summary: pd.DataFrame
) -> list[Path]:
    """Write structural metadata JSON files under ``data/metadata``."""
    md = paths.metadata_dir()
    md.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    p1 = md / "channel_names.json"
    p1.write_text(json.dumps(
        {"eeg": list(loader.CHO2017_EEG_CHANNELS), "emg": list(loader.CHO2017_EMG_CHANNELS)},
        indent=2), encoding="utf-8")
    written.append(p1)

    p2 = md / "preprocessing_summary.json"
    p2.write_text(json.dumps({
        "config": pp.hashable(),
        "preprocessing_hash": preprocessing_hash(pp.hashable()),
        "expected_n_times": pp.expected_n_times(),
        "description": (
            "per-epoch: common average reference -> zero-phase 8-30 Hz Butterworth "
            "(sosfiltfilt) -> resample 512->256 Hz -> crop 0.5-2.5 s -> 64 x 512"
        ),
    }, indent=2), encoding="utf-8")
    written.append(p2)

    ok = summary[summary["status"].isin(["ok", "skipped"])]
    p3 = md / "dataset_summary.json"
    p3.write_text(json.dumps({
        "n_subjects_requested": int(len(summary)),
        "n_subjects_processed": int(len(ok)),
        "total_trials": int(ok["n_trials"].sum()) if len(ok) else 0,
        "total_included": int(ok["n_included"].sum()) if len(ok) else 0,
        "status_distribution": summary["status"].value_counts().to_dict(),
        "run_id_source_distribution": ok["run_id_source"].value_counts().to_dict() if len(ok) else {},
        "srate_in": pp.srate_in,
        "srate_out": pp.srate_out,
        "per_subject": ok[["subject_id", "n_trials", "n_included", "run_id_source"]].to_dict("records"),
    }, indent=2), encoding="utf-8")
    written.append(p3)
    return written


def load_processed(subject: int | str) -> dict[str, Any]:
    """Load a processed cache file into a dict (scalars unwrapped)."""
    sid = subject if isinstance(subject, str) else subject_id_str(subject)
    path = paths.subject_processed_path(sid)
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found: {path}. Run prepare_dataset.py.")
    with np.load(path, allow_pickle=True) as data:
        out: dict[str, Any] = {k: data[k] for k in data.files}
    for key in PROCESSED_SCALAR_KEYS:
        if key in out and np.ndim(out[key]) == 0:
            out[key] = out[key].item()
    return out
