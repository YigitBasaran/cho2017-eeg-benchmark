"""Centralized, OS-independent path discovery for the project.

All file-system locations used across the project flow through this module so
that notebooks, scripts and the library never hard-code absolute paths. Project
root discovery is robust to being called from a notebook, a script, an editable
install or via the ``CHO2017_PROJECT_ROOT`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT_MARKER = "pyproject.toml"
_PACKAGE_DIRNAME = "cho2017_benchmark"


def _looks_like_root(path: Path) -> bool:
    """Return True if *path* is the project root.

    We require both the ``pyproject.toml`` marker and the source package, so an
    unrelated ``pyproject.toml`` higher up the tree cannot be mistaken for our
    root.
    """
    if not (path / _ROOT_MARKER).is_file():
        return False
    return (path / "src" / _PACKAGE_DIRNAME).is_dir() or (path / _PACKAGE_DIRNAME).is_dir()


def find_project_root(start: Path | str | None = None) -> Path:
    """Locate the project root directory.

    Search order: explicit *start*, the ``CHO2017_PROJECT_ROOT`` env var, the
    current working directory, and finally the directory of this file. For each
    candidate we walk upward until a directory that :func:`_looks_like_root`.

    Raises:
        FileNotFoundError: if no project root can be located.
    """
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start))
    env = os.environ.get("CHO2017_PROJECT_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parent)

    for candidate in candidates:
        candidate = candidate.resolve()
        for parent in (candidate, *candidate.parents):
            if _looks_like_root(parent):
                return parent
    raise FileNotFoundError(
        "Could not locate the project root. Expected a directory containing "
        f"'{_ROOT_MARKER}' and 'src/{_PACKAGE_DIRNAME}'. Set the "
        "CHO2017_PROJECT_ROOT environment variable to point at the project."
    )


try:
    PROJECT_ROOT: Path = find_project_root()
except FileNotFoundError:
    # Editable install fallback: src/cho2017_benchmark/paths.py -> repo root.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- Directory helpers -----------------------------------------------------


def data_dir() -> Path:
    return PROJECT_ROOT / "data"


def raw_dir() -> Path:
    """Directory holding the raw ``sNN.mat`` recordings."""
    return data_dir() / "raw" / "cho2017"


def interim_dir() -> Path:
    return data_dir() / "interim"


def processed_dir() -> Path:
    """Directory holding the per-subject processed ``.npz`` model inputs."""
    return data_dir() / "processed" / "common"


def manifests_dir() -> Path:
    return data_dir() / "manifests"


def metadata_dir() -> Path:
    return data_dir() / "metadata"


def configs_dir() -> Path:
    return PROJECT_ROOT / "configs"


def notebooks_dir() -> Path:
    return PROJECT_ROOT / "notebooks"


def results_dir(model: str | None = None) -> Path:
    """Root results directory, or a model-specific sub-directory if given."""
    base = PROJECT_ROOT / "results"
    return base / model if model else base


def eda_dir() -> Path:
    return results_dir() / "eda"


def subject_processed_path(subject_id: str) -> Path:
    """Path of the processed cache file for a subject id like ``s01``."""
    return processed_dir() / f"{subject_id}.npz"


def ensure_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if missing and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
