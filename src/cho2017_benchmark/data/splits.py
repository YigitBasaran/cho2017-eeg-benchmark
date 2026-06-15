"""Shared, run-aware train/validation/test split manifest.

The split is derived from the complete ``trial_manifest`` (every trial, before
bad-trial removal). The preferred split is a chronological later-run holdout
(train = early runs, validation = penultimate run, test = last run); when run
structure is unrecoverable a deterministic class-stratified 70/15/15 fallback is
used. Split membership is assigned on the original trial/run identity, then
excluded trials are simply marked ``included == False`` (no replacement trials
are moved between splits).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .. import paths

SPLIT_MANIFEST_COLUMNS = [
    "subject_id", "trial_id", "class_label", "original_trial_index", "run_id",
    "run_id_source", "included", "exclusion_reason", "split", "split_method",
    "random_seed",
]

_RUN_HOLDOUT_MAP = {
    5: ["train", "train", "train", "val", "test"],
    6: ["train", "train", "train", "train", "val", "test"],
}


def run_holdout_split(
    run_id: np.ndarray, run_id_source: str
) -> np.ndarray | None:
    """Assign train/val/test by run id, or None if not applicable."""
    if run_id_source not in ("inferred_from_protocol", "explicit"):
        return None
    runs = np.asarray(run_id)
    unique = sorted(int(r) for r in np.unique(runs) if r >= 1)
    if len(unique) not in _RUN_HOLDOUT_MAP:
        return None
    mapping = {run: label for run, label in zip(unique, _RUN_HOLDOUT_MAP[len(unique)])}
    return np.array([mapping.get(int(r), "") for r in runs], dtype=object)


def stratified_fallback_split(
    class_label: np.ndarray,
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.70, 0.15, 0.15),
) -> np.ndarray:
    """Deterministic class-stratified train/val/test labels for every trial."""
    from sklearn.model_selection import train_test_split

    train_r, val_r, test_r = ratios
    n = len(class_label)
    idx = np.arange(n)
    labels = np.array([str(c) for c in class_label])

    train_idx, temp_idx = train_test_split(
        idx, test_size=(val_r + test_r), stratify=labels, random_state=seed,
    )
    test_frac_of_temp = test_r / (val_r + test_r)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_frac_of_temp, stratify=labels[temp_idx],
        random_state=seed,
    )
    out = np.empty(n, dtype=object)
    out[train_idx] = "train"
    out[val_idx] = "val"
    out[test_idx] = "test"
    return out


def build_split_for_subject(
    trials: pd.DataFrame, *, seed: int = 42, ratios: Sequence[float] = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    """Add ``split`` / ``split_method`` / ``random_seed`` to one subject's trials."""
    trials = trials.reset_index(drop=True).copy()
    run_id_source = str(trials["run_id_source"].iloc[0])
    split = run_holdout_split(trials["run_id"].to_numpy(), run_id_source)
    if split is not None:
        method = "run_holdout"
        random_seed = -1
    else:
        split = stratified_fallback_split(
            trials["class_label"].to_numpy(), seed=seed, ratios=ratios
        )
        method = "stratified_fallback"
        random_seed = seed
    trials["split"] = split
    trials["split_method"] = method
    trials["random_seed"] = random_seed
    return trials


def build_split_manifest(
    trial_manifest: pd.DataFrame, cfg, *, write: bool = False, dest: Path | str | None = None
) -> pd.DataFrame:
    """Build the split manifest from the complete trial manifest."""
    seed = int(cfg.get("split.random_seed", 42))
    ratios = tuple(cfg.get("split.fallback_ratios", [0.70, 0.15, 0.15]))
    parts = [
        build_split_for_subject(group, seed=seed, ratios=ratios)
        for _, group in trial_manifest.groupby("subject_id", sort=True)
    ]
    manifest = pd.concat(parts, ignore_index=True)[SPLIT_MANIFEST_COLUMNS]
    if write:
        write_split_manifest(manifest, dest)
    return manifest


def write_split_manifest(manifest: pd.DataFrame, dest: Path | str | None = None) -> Path:
    dest = Path(dest) if dest is not None else paths.manifests_dir() / "split_manifest.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(dest, index=False)
    return dest


def assert_no_leakage(manifest: pd.DataFrame, *, excluded_subjects: Sequence[int] = (29, 34)) -> None:
    """Raise AssertionError if any split-integrity invariant is violated."""
    errors: list[str] = []

    # Excluded subjects must be absent from the (model-training) manifest.
    excluded_ids = {f"s{int(s):02d}" for s in excluded_subjects}
    present_excluded = excluded_ids & set(manifest["subject_id"].unique())
    if present_excluded:
        errors.append(f"Excluded subjects present in split manifest: {sorted(present_excluded)}")

    # Global trial_id uniqueness.
    dup = manifest["trial_id"][manifest["trial_id"].duplicated()].unique()
    if len(dup):
        errors.append(f"Duplicate trial_ids: {dup[:5].tolist()} ...")

    usable = manifest[(manifest["included"]) & (manifest["split"].isin(["train", "val", "test"]))]

    # Excluded trials must never be in a usable split.
    bad_excluded = manifest[(~manifest["included"]) & (manifest["split"].isin(["train", "val", "test"]))]
    # (Excluded trials may carry a split label by run identity; they are fine as
    # long as they are not in the *usable* set above. We only forbid included
    # trials with no split.)
    no_split = manifest[(manifest["included"]) & (~manifest["split"].isin(["train", "val", "test"]))]
    if len(no_split):
        errors.append(f"{len(no_split)} included trials have no valid split.")

    for sid, group in usable.groupby("subject_id"):
        # No trial in more than one split (within subject).
        if group["trial_id"].duplicated().any():
            errors.append(f"{sid}: a trial appears in multiple splits.")
        # test disjoint from train/val.
        test_ids = set(group[group["split"] == "test"]["trial_id"])
        trainval_ids = set(group[group["split"].isin(["train", "val"])]["trial_id"])
        if test_ids & trainval_ids:
            errors.append(f"{sid}: test trials overlap train/val.")
        # Both classes in every non-empty usable split.
        for split_name in ("train", "val", "test"):
            sub = group[group["split"] == split_name]
            if len(sub) and sub["class_label"].nunique() < 2:
                errors.append(f"{sid}: split '{split_name}' is missing a class.")

    if errors:
        raise AssertionError("Split leakage / integrity checks failed:\n  - " + "\n  - ".join(errors))


def split_summary_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-subject train/val/test class counts (included only) + exclusions."""
    rows = []
    for sid, group in manifest.groupby("subject_id", sort=True):
        inc = group[group["included"]]
        row = {"subject_id": sid, "split_method": group["split_method"].iloc[0]}
        for split_name in ("train", "val", "test"):
            sub = inc[inc["split"] == split_name]
            row[f"{split_name}_left"] = int((sub["class_label"] == "left_hand").sum())
            row[f"{split_name}_right"] = int((sub["class_label"] == "right_hand").sum())
        row["n_excluded"] = int((~group["included"]).sum())
        row["n_total"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows)
