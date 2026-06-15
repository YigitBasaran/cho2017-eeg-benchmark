from cho2017_benchmark.data import manifests
from cho2017_benchmark.data.manifests import TRIAL_MANIFEST_COLUMNS


def test_trial_manifest_contains_all_trials(tiny_processed):
    tm = manifests.build_trial_manifest(tiny_processed)
    assert list(tm.columns) == TRIAL_MANIFEST_COLUMNS
    assert len(tm) == 8  # every original trial (here none excluded)
    assert set(tm["subject_id"]) == {"s01"}
    assert tm["run_id_source"].iloc[0] == "inferred_from_protocol"


def test_trial_manifest_includes_excluded(tiny_processed):
    rec = dict(tiny_processed)
    inc = rec["included"].copy()
    inc[0] = False
    rec["included"] = inc
    rec["exclusion_reason"] = rec["exclusion_reason"].copy()
    rec["exclusion_reason"][0] = "amplitude"
    tm = manifests.build_trial_manifest(rec)
    assert len(tm) == 8  # excluded trial still present in the manifest
    assert (~tm["included"]).sum() == 1
