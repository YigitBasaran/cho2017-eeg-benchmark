import pandas as pd
import pytest

from cho2017_benchmark.config import resolve_config
from cho2017_benchmark.data import splits
from cho2017_benchmark.data.manifests import TRIAL_MANIFEST_COLUMNS


def test_no_leakage_passes(tiny_split_manifest):
    splits.assert_no_leakage(tiny_split_manifest, excluded_subjects=[29, 34])


def test_excluded_subject_detected(tiny_split_manifest):
    m = tiny_split_manifest.copy()
    m.loc[m.index[0], "subject_id"] = "s29"
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(m, excluded_subjects=[29, 34])


def test_trial_in_two_splits_detected(tiny_split_manifest):
    dup = tiny_split_manifest.iloc[[0]].copy()
    dup["split"] = "test"
    m = pd.concat([tiny_split_manifest, dup], ignore_index=True)
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(m)


def test_missing_class_in_split_detected(tiny_split_manifest):
    m = tiny_split_manifest.copy()
    m.loc[m["split"] == "test", "class_label"] = "left_hand"  # test loses right_hand
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(m)


def _five_run_trial_manifest() -> pd.DataFrame:
    rows = []
    for cls in ("left_hand", "right_hand"):
        for run in range(1, 6):
            for k in range(4):  # 4 trials per run per class -> 20/class
                oi = (run - 1) * 4 + k
                rows.append({
                    "subject_id": "s01", "trial_id": f"s01_{cls}_{oi:03d}",
                    "class_label": cls, "original_trial_index": oi,
                    "event_sample": oi * 10, "run_id": run,
                    "run_id_source": "inferred_from_protocol", "included": True,
                    "bad_amplitude": False, "bad_emg": False, "exclusion_reason": "",
                })
    return pd.DataFrame(rows)[TRIAL_MANIFEST_COLUMNS]


def test_build_split_manifest_run_holdout():
    cfg = resolve_config()
    tm = _five_run_trial_manifest()
    sm = splits.build_split_manifest(tm, cfg)
    assert set(sm.columns) == set(splits.SPLIT_MANIFEST_COLUMNS)
    assert sm["split_method"].iloc[0] == "run_holdout"
    # runs 1-3 train, 4 val, 5 test
    mapping = sm.drop_duplicates("run_id").set_index("run_id")["split"].to_dict()
    assert mapping[1] == "train" and mapping[4] == "val" and mapping[5] == "test"
    splits.assert_no_leakage(sm, excluded_subjects=[29, 34])
