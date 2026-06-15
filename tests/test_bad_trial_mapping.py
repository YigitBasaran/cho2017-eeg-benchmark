import numpy as np

from cho2017_benchmark.data import artifacts, epoching, loader
from cho2017_benchmark.data.artifacts import EXCLUDED_TRIALS_COLUMNS


def test_bad_trial_indices_map_to_valid_trials(tiny_mat, tiny_resolution):
    rec = loader.load_subject(tiny_mat, resolution=tiny_resolution)
    ep = epoching.epoch_subject(rec, tmin_s=-0.1, tmax_s=0.2, resolution=tiny_resolution)
    flags = artifacts.compute_trial_flags(rec, ep, tiny_resolution)

    left = ep.class_label == "left_hand"
    right = ep.class_label == "right_hand"
    # one-based: left #1 -> voltage(amplitude); right #2 -> mi(emg)
    assert flags["bad_amplitude"][left][0] == True  # noqa: E712
    assert flags["bad_emg"][right][1] == True  # noqa: E712
    # exactly two excluded, the rest included; flags align to 8 trials
    assert flags["included"].sum() == 6
    assert flags["bad_amplitude"].shape[0] == 8
    assert not flags["notes"]  # no out-of-range indices


def test_excluded_table_schema(tiny_mat, tiny_resolution):
    rec = loader.load_subject(tiny_mat, resolution=tiny_resolution)
    ep = epoching.epoch_subject(rec, tmin_s=-0.1, tmax_s=0.2, resolution=tiny_resolution)
    flags = artifacts.compute_trial_flags(rec, ep, tiny_resolution)
    table = artifacts.build_excluded_trials(ep, flags)
    assert list(table.columns) == EXCLUDED_TRIALS_COLUMNS
    assert len(table) == 2
    assert set(table["exclusion_reason"]) <= {"amplitude", "emg_correlation"}
