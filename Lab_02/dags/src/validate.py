"""
Task 2: data quality gate.

Lab 2 had no equivalent -- it trusted a CSV that never changed. Once the source
is a live API, "the data arrived" and "the data is usable" stop being the same
statement, so this task exists to turn silent garbage into a loud failure.

A deliberate design choice: validation failures raise `AirflowFailException`,
which fails the task *without consuming retries*. Retrying a task because the
upstream API returned malformed data just means failing again 5 minutes later;
retries are for transient faults, not for bad data.
"""
from __future__ import annotations

import json
import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

try:  # Keep this module importable (and unit-testable) without Airflow present.
    from airflow.exceptions import AirflowFailException as _BaseFailure
except ImportError:  # pragma: no cover - exercised only outside Airflow
    _BaseFailure = RuntimeError


class DataValidationError(_BaseFailure):
    """Raised when ingested data fails a quality check."""


#: Minimum usable rows. Below this, lag features + a chronological split leave
#: too little signal to train on.
MIN_ROWS = 200

#: Fraction of nulls tolerated in any single column before we refuse the batch.
MAX_NULL_FRACTION = 0.10

#: Physically plausible bounds. Values outside these indicate an upstream bug,
#: not unusual weather.
EXPECTED_RANGES: dict[str, tuple[float, float]] = {
    "pm2_5": (0.0, 1000.0),
    "pm10": (0.0, 2000.0),
    "carbon_monoxide": (0.0, 20000.0),
    "nitrogen_dioxide": (0.0, 1000.0),
    "sulphur_dioxide": (0.0, 1000.0),
    "ozone": (0.0, 1000.0),
    "temperature_2m": (-60.0, 60.0),
    "relative_humidity_2m": (0.0, 100.0),
    "wind_speed_10m": (0.0, 200.0),
    "surface_pressure": (800.0, 1100.0),
}


def check_frame(frame: pd.DataFrame) -> dict:
    """
    Run every quality check and return a structured report.

    Pure function: it does no IO and raises nothing, so the tests can assert on
    the report directly. `validate_data()` is the thin Airflow-facing wrapper
    that decides what to *do* about a failing report.
    """
    required = ["time", *config.POLLUTANT_COLUMNS, *config.WEATHER_COLUMNS]
    violations: list[str] = []

    missing = [column for column in required if column not in frame.columns]
    if missing:
        violations.append(f"missing columns: {sorted(missing)}")

    if len(frame) < MIN_ROWS:
        violations.append(f"only {len(frame)} rows, need >= {MIN_ROWS}")

    null_fractions: dict[str, float] = {}
    for column in required:
        if column not in frame.columns:
            continue
        fraction = float(frame[column].isna().mean())
        null_fractions[column] = round(fraction, 4)
        if fraction > MAX_NULL_FRACTION:
            violations.append(f"{column}: {fraction:.1%} null (max {MAX_NULL_FRACTION:.0%})")

    out_of_range: dict[str, int] = {}
    for column, (low, high) in EXPECTED_RANGES.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        count = int(((values < low) | (values > high)).sum())
        if count:
            out_of_range[column] = count
            violations.append(f"{column}: {count} values outside [{low}, {high}]")

    duplicate_timestamps = 0
    if "time" in frame.columns:
        duplicate_timestamps = int(frame["time"].duplicated().sum())
        if duplicate_timestamps:
            violations.append(f"{duplicate_timestamps} duplicate timestamps")

    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "null_fractions": null_fractions,
        "out_of_range_counts": out_of_range,
        "duplicate_timestamps": duplicate_timestamps,
        "violations": violations,
        "passed": not violations,
    }


def validate_data(raw_path: str) -> str:
    """Airflow task entry point. Returns the input path so it can chain onward."""
    config.ensure_dirs()

    frame = pd.read_csv(raw_path, parse_dates=["time"])
    report = check_frame(frame)
    report["source"] = str(frame["source"].iloc[0]) if "source" in frame.columns and len(frame) else "unknown"

    report_path = config.working_dir() / config.VALIDATION_REPORT
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Validation report -> %s", report_path)

    if not report["passed"]:
        raise DataValidationError(
            f"Data validation failed with {len(report['violations'])} violation(s): "
            + "; ".join(report["violations"])
        )

    logger.info("Validation passed: %s rows, source=%s", report["rows"], report["source"])
    return raw_path
