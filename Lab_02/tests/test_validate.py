"""Tests for the data-quality gate."""
from __future__ import annotations

import json

import pytest

from src import config, validate


def test_clean_frame_passes(hourly_frame):
    report = validate.check_frame(hourly_frame)
    assert report["passed"] is True
    assert report["violations"] == []
    assert report["rows"] == len(hourly_frame)


def test_missing_column_is_caught(hourly_frame):
    report = validate.check_frame(hourly_frame.drop(columns=["ozone"]))
    assert report["passed"] is False
    assert any("missing columns" in violation for violation in report["violations"])


def test_too_few_rows_is_caught(hourly_frame):
    report = validate.check_frame(hourly_frame.head(50))
    assert report["passed"] is False
    assert any("rows" in violation for violation in report["violations"])


def test_excess_nulls_are_caught(hourly_frame):
    frame = hourly_frame.copy()
    frame.loc[: len(frame) // 2, "pm2_5"] = None
    report = validate.check_frame(frame)
    assert report["passed"] is False
    assert report["null_fractions"]["pm2_5"] > validate.MAX_NULL_FRACTION


def test_small_null_fraction_is_tolerated(hourly_frame):
    frame = hourly_frame.copy()
    frame.loc[:4, "ozone"] = None  # 5/500 = 1%, under the 10% threshold
    report = validate.check_frame(frame)
    assert report["passed"] is True


def test_impossible_values_are_caught(hourly_frame):
    frame = hourly_frame.copy()
    frame.loc[0, "relative_humidity_2m"] = 250.0
    frame.loc[1, "pm2_5"] = -5.0
    report = validate.check_frame(frame)
    assert report["passed"] is False
    assert report["out_of_range_counts"]["relative_humidity_2m"] == 1
    assert report["out_of_range_counts"]["pm2_5"] == 1


def test_duplicate_timestamps_are_caught(hourly_frame):
    frame = hourly_frame.copy()
    frame.loc[10, "time"] = frame.loc[9, "time"]
    report = validate.check_frame(frame)
    assert report["passed"] is False
    assert report["duplicate_timestamps"] == 1


def test_validate_data_writes_report_and_passes(pipeline_dirs, hourly_frame, tmp_path):
    raw_path = tmp_path / "raw.csv"
    hourly_frame.to_csv(raw_path, index=False)

    returned = validate.validate_data(str(raw_path))

    assert returned == str(raw_path)
    report = json.loads((pipeline_dirs["working"] / config.VALIDATION_REPORT).read_text())
    assert report["passed"] is True
    assert report["source"] == "synthetic"


def test_validate_data_raises_on_bad_data(pipeline_dirs, hourly_frame, tmp_path):
    frame = hourly_frame.copy()
    frame["pm2_5"] = -1.0
    raw_path = tmp_path / "raw.csv"
    frame.to_csv(raw_path, index=False)

    with pytest.raises(validate.DataValidationError):
        validate.validate_data(str(raw_path))

    # The report is still written -- that is what the failure email reads.
    report = json.loads((pipeline_dirs["working"] / config.VALIDATION_REPORT).read_text())
    assert report["passed"] is False
