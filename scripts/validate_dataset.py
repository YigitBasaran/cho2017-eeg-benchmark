#!/usr/bin/env python
"""Deep-validate downloaded .mat files and (re)generate the layout resolution.

For every present raw file this confirms it loads, exposes the ``eeg`` struct,
has the expected sampling rate / channel count / fields and a sensible number of
event onsets. It refreshes ``file_manifest.csv`` validation status and writes the
structural report + ``mat_layout_resolution.json``.

Examples:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --subjects 1 20 29 34
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from cho2017_benchmark import paths  # noqa: E402
from cho2017_benchmark.data import download as dl  # noqa: E402
from cho2017_benchmark.data import inspect_mat  # noqa: E402

EXPECTED_SRATE = 512


def _validate_one(index: int) -> dict:
    sid = dl.subject_id_str(index)
    path = paths.raw_dir() / f"{sid}.mat"
    insp = inspect_mat.inspect_subject(path)
    checks = {
        "loads": insp.ok,
        "srate_ok": insp.srate == EXPECTED_SRATE,
        "channels_ok": insp.n_channels in (64, 68),
        "has_event": (insp.n_event_onsets_binary or 0) > 0,
        "has_trials": (insp.n_imagery_trials or 0) > 0,
    }
    ok = all(checks.values())
    return {"subject_id": sid, "validation_status": "ok" if ok else "failed",
            "srate": insp.srate, "n_channels": insp.n_channels,
            "n_onsets": insp.n_event_onsets_binary, "n_trials": insp.n_imagery_trials,
            "error": insp.error or ("; ".join(k for k, v in checks.items() if not v))}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    present = [i for i in (args.subjects or range(1, dl.N_SUBJECTS + 1))
               if (paths.raw_dir() / f"{dl.subject_id_str(i)}.mat").exists()]
    if not present:
        print("ERROR: no raw .mat files found. Run download_dataset.py first.", file=sys.stderr)
        return 2

    rows = [_validate_one(i) for i in present]
    table = pd.DataFrame(rows)
    n_ok = int((table["validation_status"] == "ok").sum())
    print(f"Validated {n_ok}/{len(table)} present files.")
    for _, row in table[table["validation_status"] != "ok"].iterrows():
        print(f"  {row['subject_id']}: {row['error']}")

    # Refresh file_manifest validation status for present files.
    fm_path = paths.manifests_dir() / "file_manifest.csv"
    if fm_path.exists():
        fm = pd.read_csv(fm_path)
        status = dict(zip(table["subject_id"], table["validation_status"]))
        fm["validation_status"] = fm.apply(
            lambda r: status.get(r["subject_id"], r["validation_status"]), axis=1)
        fm.to_csv(fm_path, index=False)

    # Inspection + resolution on the inspection subset (or whatever is present).
    subset = [i for i in inspect_mat.DEFAULT_INSPECTION_SUBJECTS if i in present] or present
    _, resolution = inspect_mat.inspect_and_resolve(subset)
    print(f"Layout resolution validated={resolution.validated} -> "
          f"{paths.metadata_dir() / inspect_mat.RESOLUTION_FILENAME}")
    for w in resolution.warnings:
        print(f"  WARNING: {w}")
    return 0 if n_ok == len(table) else 1


if __name__ == "__main__":
    raise SystemExit(main())
