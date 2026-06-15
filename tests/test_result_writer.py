import pandas as pd
import pytest

from cho2017_benchmark.evaluation.metrics import SUBJECT_METRICS_COLUMNS
from cho2017_benchmark.reporting.result_writer import (
    aggregate_participant_metrics,
    assert_schema,
    write_subject_metrics,
)


def _full_row(**over):
    row = {c: 0 for c in SUBJECT_METRICS_COLUMNS}
    row.update({"model_name": "csp_lda", "subject_id": "s01", "status": "ok",
                "accuracy": 0.7, "split_method": "run_holdout"})
    row.update(over)
    return row


def test_write_subject_metrics_enforces_schema(tmp_path):
    path = write_subject_metrics([_full_row(), _full_row(subject_id="s02")], tmp_path / "sm.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == SUBJECT_METRICS_COLUMNS
    assert len(df) == 2


def test_write_subject_metrics_missing_column_raises(tmp_path):
    with pytest.raises(ValueError):
        write_subject_metrics([{"model_name": "csp_lda"}], tmp_path / "bad.csv")


def test_aggregate_participant_metrics():
    df = pd.DataFrame({
        "subject_id": [f"s{i:02d}" for i in range(6)],
        "model_name": "eegnet",
        "accuracy": [0.6, 0.7, 0.65, 0.55, 0.8, 0.62],
    })
    agg = aggregate_participant_metrics(df)
    assert agg["level"] == "participant"
    acc = agg["metrics"]["accuracy"]
    assert acc["n"] == 6 and acc["ci_low"] <= acc["mean"] <= acc["ci_high"]


def test_assert_schema_helper():
    df = pd.DataFrame({"a": [1]})
    assert_schema(df, ["a"])
    with pytest.raises(ValueError):
        assert_schema(df, ["a", "b"])
