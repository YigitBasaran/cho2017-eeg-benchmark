"""Idempotent, disk-safe downloader for the Cho2017 GigaDB dataset.

Files are fetched from the GigaDB Wasabi S3 mirror used by MOABB. The downloader
is deliberately self-contained (``requests`` only) rather than going through
MOABB, because MOABB converts the recordings into a pseudo-continuous RawArray;
here we want the original ``.mat`` files untouched.

Safety properties (see project plan):
- estimate total size and check free disk space before starting;
- download to a ``*.part`` file and resume via HTTP Range;
- retry with exponential backoff;
- validate the ``.mat`` header (``scipy.io.whosmat``) before atomically renaming
  the ``*.part`` to its final name -- a partial/corrupt file never masquerades
  as a valid one;
- skip already-valid files unless ``force=True``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from .. import paths

BASE_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/100001_101000/100295/mat_data"
)

N_SUBJECTS = 52

FILE_MANIFEST_COLUMNS = [
    "subject_id",
    "filename",
    "source_url",
    "local_path",
    "size_bytes",
    "download_status",
    "validation_status",
    "downloaded_at",
    "error_message",
]


def subject_id_str(index: int) -> str:
    """Return the canonical subject id, e.g. ``5 -> 's05'``."""
    if not 1 <= index <= N_SUBJECTS:
        raise ValueError(f"Subject index must be in 1..{N_SUBJECTS}, got {index}.")
    return f"s{index:02d}"


def subject_filename(index: int) -> str:
    return f"{subject_id_str(index)}.mat"


def subject_url(index: int) -> str:
    return f"{BASE_URL}/{subject_filename(index)}"


@dataclass
class DownloadResult:
    subject_id: str
    filename: str
    source_url: str
    local_path: str
    size_bytes: int = 0
    download_status: str = "failed"   # ok | skipped_exists | failed
    validation_status: str = "not_validated"  # ok | failed | not_validated
    downloaded_at: str = ""
    error_message: str = ""

    def as_row(self) -> dict[str, object]:
        return {c: getattr(self, c) for c in FILE_MANIFEST_COLUMNS}


@dataclass
class DiskCheck:
    path: str
    free_bytes: int
    needed_bytes: int
    ok: bool


# --- Header validation -----------------------------------------------------


def validate_mat_header(path: Path | str) -> tuple[bool, str | None]:
    """Lightweight validity check: the ``.mat`` header parses and contains ``eeg``.

    Uses :func:`scipy.io.whosmat`, which reads only the file header (variable
    names/shapes) without loading the (large) data arrays.
    """
    path = Path(path)
    if not path.exists():
        return False, "file does not exist"
    if path.stat().st_size == 0:
        return False, "file is empty"
    try:
        from scipy.io import whosmat

        variables = whosmat(str(path))
    except Exception as exc:  # noqa: BLE001 - report any parse failure
        return False, f"whosmat failed: {exc}"
    names = {name for name, *_ in variables}
    if "eeg" not in names:
        return False, f"no 'eeg' variable (found: {sorted(names)})"
    return True, None


# --- Size estimation / disk check -----------------------------------------


def _remote_size(url: str, timeout: float) -> int | None:
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None
    except requests.RequestException:
        return None


def estimate_total_bytes(
    indices: Sequence[int], *, timeout: float = 30.0
) -> tuple[int, dict[str, int | None]]:
    """Return (sum of known sizes, per-subject sizes) via HEAD requests."""
    per_subject: dict[str, int | None] = {}
    total = 0
    for i in indices:
        size = _remote_size(subject_url(i), timeout)
        per_subject[subject_id_str(i)] = size
        if size:
            total += size
    return total, per_subject


def check_disk_space(
    dest: Path | str, needed_bytes: int, *, safety_factor: float = 1.1
) -> DiskCheck:
    """Check that *dest* has room for *needed_bytes* (times a safety factor)."""
    import shutil

    dest = Path(dest)
    probe = dest if dest.exists() else dest.parent
    usage = shutil.disk_usage(probe)
    needed = int(needed_bytes * safety_factor)
    return DiskCheck(
        path=str(probe), free_bytes=usage.free, needed_bytes=needed,
        ok=usage.free >= needed,
    )


# --- Single-file download --------------------------------------------------


def download_file(
    url: str,
    dest: Path | str,
    *,
    timeout: float = 60.0,
    max_retries: int = 5,
    resume: bool = True,
    force: bool = False,
    progress: bool = True,
    chunk_size: int = 1 << 20,
) -> tuple[bool, str | None]:
    """Download *url* to *dest* with resume + retries + atomic, validated rename.

    Returns ``(ok, error_message)``. The final file only appears at *dest* after
    its header validates.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if dest.exists() and not force:
        ok, msg = validate_mat_header(dest)
        if ok:
            return True, None
        # Existing file is corrupt -> remove and re-download.
        dest.unlink(missing_ok=True)

    if force:
        part.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)

    total = _remote_size(url, timeout)
    last_error: str | None = None

    for attempt in range(max_retries):
        try:
            existing = part.stat().st_size if part.exists() else 0
            if total is not None and existing >= total > 0:
                break  # already fully fetched, just needs validation
            headers = {}
            if resume and existing and total and existing < total:
                headers["Range"] = f"bytes={existing}-"
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
                resp.raise_for_status()
                # If we asked for a range but the server ignored it, restart.
                append = bool(headers.get("Range")) and resp.status_code == 206
                mode = "ab" if append else "wb"
                bar = _make_progress(progress, total, existing if append else 0, dest.name)
                with open(part, mode) as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                            if bar is not None:
                                bar.update(len(chunk))
                if bar is not None:
                    bar.close()
            if total is None or part.stat().st_size >= total:
                break
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(60.0, 2.0 ** attempt))
    else:
        return False, last_error or "exhausted retries"

    if total is not None and part.stat().st_size < total:
        return False, f"incomplete download ({part.stat().st_size}/{total} bytes)"

    ok, msg = validate_mat_header(part)
    if not ok:
        return False, f"validation failed: {msg}"
    os.replace(part, dest)
    return True, None


def _make_progress(enabled: bool, total: int | None, initial: int, desc: str):
    if not enabled:
        return None
    try:
        from tqdm import tqdm

        return tqdm(
            total=total, initial=initial, unit="B", unit_scale=True,
            desc=desc, leave=False,
        )
    except ImportError:
        return None


# --- Whole-dataset download ------------------------------------------------


def download_dataset(
    subject_indices: Iterable[int] | None = None,
    *,
    force: bool = False,
    timeout: float = 60.0,
    max_retries: int = 5,
    resume: bool = True,
    progress: bool = True,
    check_space: bool = True,
    dest_dir: Path | str | None = None,
    write_manifest: bool = True,
) -> pd.DataFrame:
    """Download the requested subjects and return/update the file manifest.

    Raises:
        RuntimeError: if a disk-space check fails before any download starts.
    """
    indices = list(subject_indices) if subject_indices is not None else list(range(1, N_SUBJECTS + 1))
    dest_dir = Path(dest_dir) if dest_dir is not None else paths.raw_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    if check_space:
        total, _ = estimate_total_bytes(indices, timeout=timeout)
        if total > 0:
            disk = check_disk_space(dest_dir, total)
            if not disk.ok:
                raise RuntimeError(
                    "Insufficient disk space for download. "
                    f"Need ~{disk.needed_bytes / 1e9:.1f} GB (incl. safety margin), "
                    f"only {disk.free_bytes / 1e9:.1f} GB free at {disk.path}."
                )

    results: list[DownloadResult] = []
    for i in indices:
        sid = subject_id_str(i)
        fname = subject_filename(i)
        url = subject_url(i)
        dest = dest_dir / fname
        result = DownloadResult(
            subject_id=sid, filename=fname, source_url=url, local_path=str(dest),
        )

        existed_valid = dest.exists() and not force and validate_mat_header(dest)[0]
        ok, err = download_file(
            url, dest, timeout=timeout, max_retries=max_retries,
            resume=resume, force=force, progress=progress,
        )
        result.downloaded_at = datetime.now(timezone.utc).isoformat()
        if ok:
            result.download_status = "skipped_exists" if existed_valid else "ok"
            result.size_bytes = dest.stat().st_size
            v_ok, v_msg = validate_mat_header(dest)
            result.validation_status = "ok" if v_ok else "failed"
            if not v_ok:
                result.error_message = v_msg or ""
        else:
            result.download_status = "failed"
            result.validation_status = "failed"
            result.error_message = err or ""
        results.append(result)

    manifest = pd.DataFrame([r.as_row() for r in results], columns=FILE_MANIFEST_COLUMNS)
    if write_manifest:
        out = paths.manifests_dir() / "file_manifest.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest = _merge_manifest(out, manifest)
        manifest.to_csv(out, index=False)
    return manifest


def _merge_manifest(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Merge freshly downloaded rows into any existing manifest (new wins)."""
    if not path.exists():
        return new
    try:
        old = pd.read_csv(path)
    except Exception:
        return new
    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset="subject_id", keep="last")
    combined = combined.sort_values("subject_id").reset_index(drop=True)
    return combined[FILE_MANIFEST_COLUMNS]
