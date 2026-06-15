"""Epoch extraction from the continuous per-class arrays.

Trials are located from the ``imagery_event`` binary marker (the ground truth
for onset positions), not by blindly reshaping the signal. A generous *raw*
epoch is extracted around each onset (filtering happens later, per epoch, so it
cannot leak across split boundaries). The full padded interval -- including the
pre-stimulus baseline -- is preserved for EDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .loader import N_EEG, N_EMG, SubjectRecording, split_eeg_emg

DEFAULT_CLASS_MAPPING = {"left_hand": 0, "right_hand": 1}


@dataclass
class EpochedSubject:
    subject_id: str
    X_eeg: np.ndarray            # (n_trials, 64, n_epoch_samples) float32 (raw)
    X_emg: np.ndarray            # (n_trials, 4, n_epoch_samples) float32 (raw)
    y: np.ndarray                # (n_trials,) int64 in {0, 1}
    class_label: np.ndarray      # (n_trials,) str
    original_trial_index: np.ndarray  # (n_trials,) per-class 0-based index
    event_sample: np.ndarray     # (n_trials,) onset sample in the continuous array
    run_id: np.ndarray           # (n_trials,) int (>=1 or -1 if unrecoverable)
    run_id_source: str           # explicit | inferred_from_protocol | unrecoverable
    trial_id: np.ndarray         # (n_trials,) str, deterministic
    srate: float
    epoch_tmin_s: float
    epoch_tmax_s: float
    class_names: tuple[str, ...] = ("left_hand", "right_hand")

    @property
    def n_trials(self) -> int:
        return int(self.X_eeg.shape[0])

    @property
    def n_epoch_samples(self) -> int:
        return int(self.X_eeg.shape[2])

    def epoch_times(self) -> np.ndarray:
        return self.epoch_tmin_s + np.arange(self.n_epoch_samples) / self.srate


def find_onsets(event: np.ndarray, resolution: dict | None = None) -> np.ndarray:
    """Return onset sample indices from the ``imagery_event`` marker.

    For the verified Cho2017 layout the event is a binary marker vector, so
    onsets are the nonzero positions. If a *resolution* declares an
    ``onset_index_array`` representation, the values are treated as indices.
    """
    flat = np.asarray(event).ravel()
    representation = (resolution or {}).get("event_representation", "binary_marker_vector")
    if representation == "onset_index_array":
        return flat.astype(int)
    return np.flatnonzero(flat).astype(int)


def extract_epochs(
    eeg: np.ndarray,
    emg: np.ndarray,
    onsets: Sequence[int],
    *,
    srate: float,
    tmin_s: float,
    tmax_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``[tmin_s, tmax_s)`` windows (relative to each onset)."""
    n_pre = int(round(-tmin_s * srate))
    n_post = int(round(tmax_s * srate))
    n_samples = n_pre + n_post
    if n_samples <= 0:
        raise ValueError(f"Empty epoch window for tmin={tmin_s}, tmax={tmax_s}.")
    total = eeg.shape[1]

    onsets = np.asarray(onsets, dtype=int)
    starts = onsets - n_pre
    ends = onsets + n_post
    if np.any(starts < 0) or np.any(ends > total):
        bad = onsets[(starts < 0) | (ends > total)]
        raise ValueError(
            f"Epoch window [{tmin_s}, {tmax_s}] s does not fit for onsets {bad[:5]} "
            f"(signal length {total}). Reduce the epoch window."
        )

    X_eeg = np.empty((len(onsets), eeg.shape[0], n_samples), dtype=np.float32)
    X_emg = np.empty((len(onsets), emg.shape[0], n_samples), dtype=np.float32)
    for k, (s, e) in enumerate(zip(starts, ends)):
        X_eeg[k] = eeg[:, s:e]
        X_emg[k] = emg[:, s:e]
    return X_eeg, X_emg


def recover_runs(
    n_per_class: int,
    *,
    trials_per_run: int = 20,
    valid_trials_per_class: Sequence[int] = (100, 120),
    valid_n_runs: Sequence[int] = (5, 6),
) -> tuple[np.ndarray, str]:
    """Recover run ids for one class as contiguous blocks, if justified.

    The Cho2017 ``.mat`` files carry no explicit run field. The documented
    protocol uses sequential runs of 20 trials, so contiguous 20-trial blocks
    are accepted as runs *only* when the per-class count is 100 or 120 (=> 5 or
    6 runs) and divides evenly. This assumes chronological trial storage. When
    these conditions do not hold, runs are marked ``unrecoverable``.
    """
    if (
        n_per_class in tuple(valid_trials_per_class)
        and trials_per_run > 0
        and n_per_class % trials_per_run == 0
        and (n_per_class // trials_per_run) in tuple(valid_n_runs)
    ):
        runs = np.arange(n_per_class, dtype=int) // trials_per_run + 1
        return runs, "inferred_from_protocol"
    return np.full(n_per_class, -1, dtype=int), "unrecoverable"


def make_trial_id(subject_id: str, class_name: str, original_index: int) -> str:
    """Deterministic, unique-within-subject trial id, e.g. ``s01_left_hand_000``."""
    return f"{subject_id}_{class_name}_{int(original_index):03d}"


def epoch_subject(
    recording: SubjectRecording,
    *,
    tmin_s: float = -1.5,
    tmax_s: float = 4.0,
    class_mapping: dict[str, int] | None = None,
    trials_per_run: int = 20,
    valid_trials_per_class: Sequence[int] = (100, 120),
    valid_n_runs: Sequence[int] = (5, 6),
    resolution: dict | None = None,
) -> EpochedSubject:
    """Epoch a subject's left + right trials into a single :class:`EpochedSubject`."""
    class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
    left_name, right_name = "left_hand", "right_hand"
    onsets = find_onsets(recording.imagery_event, resolution)
    n = len(onsets)
    srate = recording.srate

    eeg_l, emg_l = split_eeg_emg(recording.imagery_left)
    eeg_r, emg_r = split_eeg_emg(recording.imagery_right)
    Xe_l, Xm_l = extract_epochs(eeg_l, emg_l, onsets, srate=srate, tmin_s=tmin_s, tmax_s=tmax_s)
    Xe_r, Xm_r = extract_epochs(eeg_r, emg_r, onsets, srate=srate, tmin_s=tmin_s, tmax_s=tmax_s)

    runs, source = recover_runs(
        n, trials_per_run=trials_per_run,
        valid_trials_per_class=valid_trials_per_class, valid_n_runs=valid_n_runs,
    )

    idx = np.arange(n, dtype=int)
    X_eeg = np.concatenate([Xe_l, Xe_r], axis=0)
    X_emg = np.concatenate([Xm_l, Xm_r], axis=0)
    y = np.array([class_mapping[left_name]] * n + [class_mapping[right_name]] * n, dtype=np.int64)
    class_label = np.array([left_name] * n + [right_name] * n)
    original_trial_index = np.concatenate([idx, idx])
    event_sample = np.concatenate([onsets, onsets])
    run_id = np.concatenate([runs, runs])
    trial_id = np.array(
        [make_trial_id(recording.subject_id, left_name, i) for i in idx]
        + [make_trial_id(recording.subject_id, right_name, i) for i in idx]
    )

    return EpochedSubject(
        subject_id=recording.subject_id,
        X_eeg=X_eeg, X_emg=X_emg, y=y, class_label=class_label,
        original_trial_index=original_trial_index, event_sample=event_sample,
        run_id=run_id, run_id_source=source, trial_id=trial_id, srate=srate,
        epoch_tmin_s=tmin_s, epoch_tmax_s=tmax_s,
        class_names=(left_name, right_name),
    )
