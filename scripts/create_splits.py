#!/usr/bin/env python
"""Build the shared trial / split / excluded-trial manifests from the cache.

Reads the per-subject processed ``.npz`` files (run prepare_dataset.py first),
builds:
  - data/manifests/trial_manifest.csv     (every trial, before exclusion)
  - data/manifests/split_manifest.csv     (run-aware holdout or stratified fallback)
  - data/manifests/excluded_trials.csv    (bad trials only)
  - data/manifests/split_summary.csv      (per-subject class counts per split)
then verifies there is no split leakage.

Examples:
    python scripts/create_splits.py
    python scripts/create_splits.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from cho2017_benchmark import paths  # noqa: E402
from cho2017_benchmark.config import resolve_config  # noqa: E402
from cho2017_benchmark.data import manifests, prepare, splits  # noqa: E402
from cho2017_benchmark.data.artifacts import EXCLUDED_TRIALS_COLUMNS  # noqa: E402
from cho2017_benchmark.data.preprocessing import PreprocessingConfig  # noqa: E402
from cho2017_benchmark.reproducibility import preprocessing_hash, split_manifest_hash  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = resolve_config(quick_mode=args.quick or None)
    subjects = cfg.active_subjects()  # excludes s29/s34

    trial_frames = []
    missing = []
    for index in subjects:
        try:
            record = prepare.load_processed(index)
        except FileNotFoundError:
            missing.append(index)
            continue
        trial_frames.append(manifests.build_trial_manifest(record))

    if not trial_frames:
        print("ERROR: no processed subjects found. Run prepare_dataset.py first.",
              file=sys.stderr)
        return 2
    if missing:
        print(f"WARNING: {len(missing)} active subjects missing processed cache: {missing}")

    trial_manifest = pd.concat(trial_frames, ignore_index=True)
    manifests.write_trial_manifest(trial_manifest)

    split_manifest = splits.build_split_manifest(trial_manifest, cfg, write=True)

    excluded = trial_manifest[~trial_manifest["included"]].copy()
    excluded = excluded[["subject_id", "class_label", "original_trial_index", "run_id", "exclusion_reason"]]
    excluded["source"] = "provided_bad_trials"
    excluded["notes"] = ""
    excluded = excluded[EXCLUDED_TRIALS_COLUMNS]
    excluded.to_csv(paths.manifests_dir() / "excluded_trials.csv", index=False)

    splits.assert_no_leakage(split_manifest, excluded_subjects=cfg.excluded_subjects)

    summary = splits.split_summary_table(split_manifest)
    summary.to_csv(paths.manifests_dir() / "split_summary.csv", index=False)

    pp = PreprocessingConfig.from_config(cfg)
    hashes = {
        "split_manifest_hash": split_manifest_hash(split_manifest),
        "preprocessing_hash": preprocessing_hash(pp.hashable()),
        "n_subjects": int(split_manifest["subject_id"].nunique()),
    }
    (paths.manifests_dir() / "manifest_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8")

    print(f"Subjects: {split_manifest['subject_id'].nunique()} | "
          f"trials: {len(split_manifest)} | "
          f"methods: {split_manifest['split_method'].value_counts().to_dict()}")
    print(f"split_manifest_hash: {hashes['split_manifest_hash']}")
    print("Leakage checks: PASSED")
    print("Manifests written to", paths.manifests_dir())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
