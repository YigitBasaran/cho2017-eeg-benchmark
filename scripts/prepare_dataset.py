#!/usr/bin/env python
"""Build the per-subject processed cache and dataset metadata.

Requires a validated data/metadata/mat_layout_resolution.json (run
scripts/download_dataset.py --inspect or scripts/validate_dataset.py first).

Examples:
    python scripts/prepare_dataset.py                 # all subjects (incl. s29/s34 for EDA)
    python scripts/prepare_dataset.py --quick         # quick_subjects only
    python scripts/prepare_dataset.py --subjects 1 2 3
    python scripts/prepare_dataset.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cho2017_benchmark.config import resolve_config  # noqa: E402
from cho2017_benchmark.data import prepare  # noqa: E402
from cho2017_benchmark.data.preprocessing import PreprocessingConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", help="explicit subject indices")
    p.add_argument("--quick", action="store_true", help="quick mode (quick_subjects only)")
    p.add_argument("--force", action="store_true", help="re-prepare even if valid")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = resolve_config(quick_mode=args.quick or None)
    pp = PreprocessingConfig.from_config(cfg)

    try:
        summary = prepare.prepare_all(
            cfg, subjects=args.subjects, pp=pp, force=args.force,
            progress=not args.no_progress,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    counts = summary["status"].value_counts().to_dict()
    print("\nPrepare summary:", counts)
    bad = summary[~summary["status"].isin(["ok", "skipped"])]
    for _, row in bad.iterrows():
        print(f"  {row['subject_id']}: {row['status']}")

    written = prepare.write_dataset_metadata(cfg, pp, summary)
    print("Metadata written:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
