#!/usr/bin/env python
"""Print a concise project status: data, manifests, results and environment."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cho2017_benchmark import paths  # noqa: E402
from cho2017_benchmark.reproducibility import collect_environment  # noqa: E402


def _count(glob_dir: Path, pattern: str) -> int:
    return len(list(glob_dir.glob(pattern))) if glob_dir.exists() else 0


def main() -> int:
    root = paths.PROJECT_ROOT
    print("=" * 64)
    print("Cho2017 Motor Imagery Benchmark - project status")
    print("=" * 64)
    print(f"Project root: {root}")

    print("\n[Data]")
    print(f"  raw .mat files:        {_count(paths.raw_dir(), '*.mat')} / 52")
    print(f"  partial (.part):       {_count(paths.raw_dir(), '*.part')}")
    print(f"  processed .npz:        {_count(paths.processed_dir(), '*.npz')}")

    print("\n[Manifests]")
    for name in ("file_manifest.csv", "trial_manifest.csv", "split_manifest.csv",
                 "excluded_trials.csv", "split_summary.csv"):
        p = paths.manifests_dir() / name
        print(f"  {name:24s} {'present' if p.exists() else 'missing'}")
    res = paths.metadata_dir() / "mat_layout_resolution.json"
    print(f"  mat_layout_resolution.json {'present' if res.exists() else 'missing'}")

    print("\n[Results] (should be empty until the notebooks are run)")
    for model in ("eda", "csp_lda", "eegnet", "atcnet", "comparison"):
        base = paths.results_dir(model)
        n_files = sum(1 for p in base.rglob("*") if p.is_file() and p.name != ".gitkeep")
        print(f"  results/{model:11s} {n_files} generated file(s)")

    print("\n[Notebooks]")
    for nb in sorted(paths.notebooks_dir().glob("*.ipynb")):
        print(f"  {nb.name}")

    print("\n[Environment]")
    env = collect_environment()
    print(f"  python: {env['python_version']} | platform: {env['platform']}")
    print(f"  CUDA available: {env.get('torch_cuda_available')}")
    for pkg in ("numpy", "scipy", "scikit-learn", "torch", "mne", "moabb", "braindecode", "pyarrow"):
        print(f"    {pkg:14s} {env['packages'].get(pkg) or 'NOT INSTALLED'}")

    print("\n[Next steps]")
    steps = [
        ("Download dataset", _count(paths.raw_dir(), "*.mat") >= 1),
        ("Validate + resolve layout", res.exists()),
        ("Prepare processed cache", _count(paths.processed_dir(), "*.npz") >= 1),
        ("Create splits", (paths.manifests_dir() / "split_manifest.csv").exists()),
    ]
    for label, done in steps:
        print(f"  [{'x' if done else ' '}] {label}")
    print("  [ ] Run notebooks 00-04 (manual)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
