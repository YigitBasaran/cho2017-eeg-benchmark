"""Direct, canonical loader for the raw Cho2017 ``.mat`` files.

We parse the original MATLAB files ourselves (not via MOABB's RawArray, which
concatenates class recordings with zero buffers and introduces artificial edge
effects). The structure was verified by inspection (see
``results/eda/reports/mat_structure_report.md`` /
``data/metadata/mat_layout_resolution.json``):

- top-level ``eeg`` struct;
- ``imagery_left`` / ``imagery_right``: float32 ``(68, T)`` continuous per-class
  signals (first 64 channels EEG, next 4 EMG);
- ``imagery_event``: a binary marker vector (length ``T``) shared by both class
  arrays, with one ``1`` per trial onset;
- ``srate`` = 512 Hz, ``frame`` = ``[-2000, 5000]`` ms, ``bad_trial_indices`` a
  struct of class-specific (left/right) one-based index cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import paths

N_EEG = 64
N_EMG = 4
N_TOTAL = 68

# Canonical 64-channel EEG montage for Cho2017 (standard Biosemi-64 / MOABB
# order). Verified against the file's ``senloc`` coordinates: C3 (idx 12) has
# x < 0 (left), C4 (idx 49) x > 0 (right), Cz (idx 47) ~ 0.
CHO2017_EEG_CHANNELS: list[str] = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]
CHO2017_EMG_CHANNELS: list[str] = ["EMG1", "EMG2", "EMG3", "EMG4"]

assert len(CHO2017_EEG_CHANNELS) == N_EEG


@dataclass
class SubjectRecording:
    """Raw, untouched signals and metadata for one subject."""

    subject_id: str
    imagery_left: np.ndarray   # (68, T) float32
    imagery_right: np.ndarray  # (68, T) float32
    imagery_event: np.ndarray  # (T,)
    srate: float
    frame_ms: list[float]
    n_imagery_trials: int
    bad_trial_indices: dict[str, dict[str, np.ndarray]]  # {field: {"left":..,"right":..}} raw
    senloc: np.ndarray | None
    psenloc: np.ndarray | None
    comment: str
    subject_label: str

    @property
    def n_samples(self) -> int:
        return int(self.imagery_left.shape[1])

    def class_signal(self, class_name: str) -> np.ndarray:
        if class_name in ("left", "left_hand"):
            return self.imagery_left
        if class_name in ("right", "right_hand"):
            return self.imagery_right
        raise KeyError(f"Unknown class '{class_name}'.")


def split_eeg_emg(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``(68, T)`` (or ``(68, ...)``) array into EEG (first 64) + EMG (next 4)."""
    if signal.shape[0] != N_TOTAL:
        raise ValueError(
            f"Expected {N_TOTAL} channels on axis 0, got shape {signal.shape}. "
            "The EEG/EMG split assumes channel-major orientation."
        )
    return signal[:N_EEG], signal[N_EEG:N_TOTAL]


def _as_int_indices(value: Any) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value)).ravel()
    if arr.size == 0:
        return np.array([], dtype=int)
    if arr.dtype.kind == "f":
        arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.array([], dtype=int)
    return arr.astype(int)


def _cell_left_right(cell: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split a MATLAB 2-cell ``{left, right}`` into two index arrays."""
    obj = np.asarray(cell, dtype=object)
    flat = list(obj.ravel()) if obj.ndim else [obj.item()]
    if len(flat) >= 2:
        return _as_int_indices(flat[0]), _as_int_indices(flat[1])
    if len(flat) == 1:
        return _as_int_indices(flat[0]), np.array([], dtype=int)
    return np.array([], dtype=int), np.array([], dtype=int)


def parse_bad_trial_indices(struct: Any) -> dict[str, dict[str, np.ndarray]]:
    """Parse the ``bad_trial_indices`` struct into ``{field: {left, right}}`` (raw)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    if struct is None or not hasattr(struct, "_fieldnames"):
        return out
    for fname in struct._fieldnames:
        left, right = _cell_left_right(getattr(struct, fname))
        out[fname] = {"left": left, "right": right}
    return out


def _scalar(value: Any, default: Any = None) -> Any:
    try:
        arr = np.asarray(value).ravel()
        return arr[0] if arr.size else default
    except Exception:
        return default


def load_subject(
    path: Path | str | int, *, resolution: dict[str, Any] | None = None
) -> SubjectRecording:
    """Load one subject's recording from its ``.mat`` file.

    *path* may be a file path or an integer subject index (1..52). When a
    layout *resolution* is provided, the loader asserts the file matches the
    resolved orientation / channel count / sampling rate.
    """
    if isinstance(path, int):
        from .download import subject_id_str

        path = paths.raw_dir() / f"{subject_id_str(path)}.mat"
    path = Path(path)
    from scipy.io import loadmat

    mat = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    if "eeg" not in mat:
        raise ValueError(f"{path} has no 'eeg' struct.")
    eeg = mat["eeg"]

    imagery_left = np.asarray(eeg.imagery_left, dtype=np.float32)
    imagery_right = np.asarray(eeg.imagery_right, dtype=np.float32)
    imagery_event = np.asarray(eeg.imagery_event).ravel()
    srate = float(_scalar(eeg.srate, 512.0))

    # Validate against expectations / resolution.
    if imagery_left.ndim != 2 or imagery_left.shape[0] != N_TOTAL:
        raise ValueError(
            f"{path}: imagery_left shape {imagery_left.shape} is not (68, T). "
            "Channel-major orientation assumed; re-run inspection if this fails."
        )
    if imagery_right.shape != imagery_left.shape:
        raise ValueError(f"{path}: imagery_left/right shapes differ "
                         f"({imagery_left.shape} vs {imagery_right.shape}).")
    if abs(imagery_event.size - imagery_left.shape[1]) > 1:
        raise ValueError(
            f"{path}: imagery_event length {imagery_event.size} does not match "
            f"signal length {imagery_left.shape[1]} (expected a binary marker vector)."
        )
    if resolution is not None:
        exp_srate = resolution.get("srate")
        if exp_srate and abs(srate - float(exp_srate)) > 1e-6:
            raise ValueError(f"{path}: srate {srate} != resolved {exp_srate}.")

    frame = np.asarray(getattr(eeg, "frame", [-2000.0, 5000.0])).ravel().astype(float).tolist()
    n_imagery_trials = int(_scalar(getattr(eeg, "n_imagery_trials", 0), 0))
    bad = parse_bad_trial_indices(getattr(eeg, "bad_trial_indices", None))
    senloc = _safe_array(getattr(eeg, "senloc", None))
    psenloc = _safe_array(getattr(eeg, "psenloc", None))
    comment = str(_scalar(getattr(eeg, "comment", ""), ""))
    subject_label = str(_scalar(getattr(eeg, "subject", path.stem), path.stem))

    return SubjectRecording(
        subject_id=path.stem,
        imagery_left=imagery_left,
        imagery_right=imagery_right,
        imagery_event=imagery_event,
        srate=srate,
        frame_ms=frame,
        n_imagery_trials=n_imagery_trials,
        bad_trial_indices=bad,
        senloc=senloc,
        psenloc=psenloc,
        comment=comment,
        subject_label=subject_label,
    )


def _safe_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return None


def load_channel_names() -> dict[str, list[str]]:
    """Return the canonical EEG and EMG channel names."""
    return {"eeg": list(CHO2017_EEG_CHANNELS), "emg": list(CHO2017_EMG_CHANNELS)}
