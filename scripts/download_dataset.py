#!/usr/bin/env python
"""Download the Cho2017 dataset into ``data/raw/cho2017`` (idempotent, disk-safe).

Examples:
    python scripts/download_dataset.py                 # all 52 subjects
    python scripts/download_dataset.py --inspection    # only s01,s20,s29,s33,s34,s52
    python scripts/download_dataset.py --subjects 1 2 3
    python scripts/download_dataset.py --force         # re-download everything
    python scripts/download_dataset.py --inspect       # download all + write layout resolution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running before `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cho2017_benchmark.data import download as dl  # noqa: E402
from cho2017_benchmark.data import inspect_mat  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--subjects", type=int, nargs="+", help="explicit subject indices (1..52)")
    g.add_argument("--inspection", action="store_true",
                   help="download only the structural-inspection subset")
    p.add_argument("--force", action="store_true", help="re-download even if valid")
    p.add_argument("--no-space-check", action="store_true", help="skip the disk-space check")
    p.add_argument("--no-progress", action="store_true", help="disable progress bars")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--inspect", action="store_true",
                   help="after downloading, inspect files and write the layout resolution")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inspection:
        indices = list(inspect_mat.DEFAULT_INSPECTION_SUBJECTS)
    elif args.subjects:
        indices = args.subjects
    else:
        indices = list(range(1, dl.N_SUBJECTS + 1))

    print(f"Downloading {len(indices)} subject file(s) into {dl.paths.raw_dir()}")
    try:
        manifest = dl.download_dataset(
            indices,
            force=args.force,
            timeout=args.timeout,
            max_retries=args.max_retries,
            progress=not args.no_progress,
            check_space=not args.no_space_check,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ok = (manifest["validation_status"] == "ok").sum()
    failed = manifest[manifest["validation_status"] != "ok"]
    print(f"\nValidated OK: {ok}/{len(manifest)}")
    if len(failed):
        print("Files needing attention:")
        for _, row in failed.iterrows():
            print(f"  {row['subject_id']}: {row['download_status']} / "
                  f"{row['validation_status']} -- {row['error_message']}")
    print(f"Manifest: {dl.paths.manifests_dir() / 'file_manifest.csv'}")

    if args.inspect:
        print("\nInspecting structure and writing layout resolution ...")
        # Inspect the default subset if present, else whatever was downloaded.
        present = [i for i in inspect_mat.DEFAULT_INSPECTION_SUBJECTS
                   if (dl.paths.raw_dir() / f"{dl.subject_id_str(i)}.mat").exists()]
        subjects = present or indices
        inspections, resolution = inspect_mat.inspect_and_resolve(subjects)
        print(f"  Report:     {inspect_mat.paths.eda_dir() / 'reports' / inspect_mat.REPORT_FILENAME}")
        print(f"  Resolution: {inspect_mat.paths.metadata_dir() / inspect_mat.RESOLUTION_FILENAME} "
              f"(validated={resolution.validated})")
        for w in resolution.warnings:
            print(f"  WARNING: {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
