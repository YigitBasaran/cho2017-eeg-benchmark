"""Shared tiny synthetic fixtures so unit tests never need the 80 GB dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cho2017_benchmark.data.epoching import EpochedSubject, make_trial_id
from cho2017_benchmark.data.splits import SPLIT_MANIFEST_COLUMNS

SRATE = 128


@pytest.fixture
def tiny_resolution() -> dict:
    return {
        "validated": True,
        "srate": SRATE,
        "orientation": "channels_by_time",
        "event_representation": "binary_marker_vector",
        "event_semantics": "event marks cue (t=0)",
        "time_zero_definition": "onset",
        "imagery_interval_relative_to_event_seconds": [-1.0, 2.0],
        "primary_crop_relative_to_event_seconds": [0.0, 0.25],
        "index_base": "one_based",
        "bad_trial_scope": "class_specific",
        "rejection_reason_mapping": {
            "bad_trial_idx_voltage": "amplitude",
            "bad_trial_idx_mi": "emg_correlation",
        },
        "run_metadata_available": False,
        "pre_stimulus_available": True,
        "pre_stimulus_samples": 64,
    }


@pytest.fixture
def tiny_mat(tmp_path):
    """Write a tiny but structurally faithful ``sNN.mat`` and return its path."""
    from scipy.io import savemat

    rng = np.random.RandomState(0)
    n = 4  # trials per class
    total = 240
    onsets = np.array([40, 90, 140, 190])
    event = np.zeros(total)
    event[onsets] = 1.0
    left = rng.randn(68, total).astype(np.float32)
    right = rng.randn(68, total).astype(np.float32)
    # class-specific, one-based bad indices: left trial #1 voltage, right trial #2 EMG.
    bad = {
        "bad_trial_idx_voltage": np.array([np.array([1.0]), np.array([], dtype=float)], dtype=object),
        "bad_trial_idx_mi": np.array([np.array([], dtype=float), np.array([2.0])], dtype=object),
    }
    eeg = {
        "imagery_left": left, "imagery_right": right, "imagery_event": event,
        "srate": float(SRATE), "n_imagery_trials": float(n),
        "frame": np.array([-200.0, 400.0]), "bad_trial_indices": bad,
        "senloc": rng.randn(64, 3), "psenloc": rng.randn(64, 3),
        "comment": "tiny", "subject": "tiny",
    }
    path = tmp_path / "s99.mat"
    savemat(str(path), {"eeg": eeg})
    return path


@pytest.fixture
def tiny_epoched() -> EpochedSubject:
    rng = np.random.RandomState(1)
    n = 4
    n_samples = 192  # tmin=-0.5, tmax=1.0 at 128 Hz
    X_eeg = rng.randn(2 * n, 64, n_samples).astype(np.float32)
    X_emg = rng.randn(2 * n, 4, n_samples).astype(np.float32)
    idx = np.arange(n)
    return EpochedSubject(
        subject_id="s99",
        X_eeg=X_eeg, X_emg=X_emg,
        y=np.array([0] * n + [1] * n, dtype=np.int64),
        class_label=np.array(["left_hand"] * n + ["right_hand"] * n),
        original_trial_index=np.concatenate([idx, idx]),
        event_sample=np.arange(2 * n) * 50 + 40,
        run_id=np.ones(2 * n, dtype=int),
        run_id_source="inferred_from_protocol",
        trial_id=np.array([make_trial_id("s99", "left_hand", i) for i in idx]
                          + [make_trial_id("s99", "right_hand", i) for i in idx]),
        srate=float(SRATE), epoch_tmin_s=-0.5, epoch_tmax_s=1.0,
    )


@pytest.fixture
def tiny_processed() -> dict:
    rng = np.random.RandomState(2)
    n = 4
    trial_id = np.array([make_trial_id("s01", "left_hand", i) for i in range(n)]
                        + [make_trial_id("s01", "right_hand", i) for i in range(n)])
    return {
        "subject_id": "s01",
        "X_eeg": rng.randn(2 * n, 64, 32).astype(np.float32),
        "y": np.array([0] * n + [1] * n, dtype=np.int64),
        "trial_id": trial_id,
        "class_label": np.array(["left_hand"] * n + ["right_hand"] * n),
        "original_trial_index": np.concatenate([np.arange(n), np.arange(n)]),
        "event_sample": np.arange(2 * n),
        "run_id": np.ones(2 * n, dtype=int),
        "run_id_source": "inferred_from_protocol",
        "included": np.ones(2 * n, dtype=bool),
        "bad_amplitude": np.zeros(2 * n, dtype=bool),
        "bad_emg": np.zeros(2 * n, dtype=bool),
        "exclusion_reason": np.array([""] * (2 * n), dtype=object),
        "sampling_rate": 64,
    }


@pytest.fixture
def tiny_split_manifest(tiny_processed) -> pd.DataFrame:
    """A leakage-free split for the tiny_processed trials (2L2R train, 1/1 val, 1/1 test)."""
    tid = tiny_processed["trial_id"]
    cls = tiny_processed["class_label"]
    # left: 0,1->train 2->val 3->test ; right: 0,1->train 2->val 3->test
    split = []
    for c, oi in zip(cls, tiny_processed["original_trial_index"]):
        split.append("train" if oi < 2 else ("val" if oi == 2 else "test"))
    return pd.DataFrame({
        "subject_id": "s01", "trial_id": tid, "class_label": cls,
        "original_trial_index": tiny_processed["original_trial_index"],
        "run_id": tiny_processed["run_id"], "run_id_source": "inferred_from_protocol",
        "included": True, "exclusion_reason": "", "split": split,
        "split_method": "run_holdout", "random_seed": -1,
    })[SPLIT_MANIFEST_COLUMNS]
