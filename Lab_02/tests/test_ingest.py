"""
Tests for ingestion.

These never hit the network. A test that calls Open-Meteo for real is not a
unit test -- it is a monitor for somebody else's uptime, and it will fail your
CI on a day when nothing you own is broken.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config, ingest


def test_hourly_frame_parses_open_meteo_shape():
    payload = {
        "hourly": {
            "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
            "pm2_5": [7.1, 8.4],
            "pm10": [11.0, 12.5],
        }
    }
    frame = ingest._hourly_frame(payload, ["pm2_5", "pm10"])
    assert list(frame.columns) == ["time", "pm2_5", "pm10"]
    assert len(frame) == 2
    assert pd.api.types.is_datetime64_any_dtype(frame["time"])


def test_hourly_frame_fills_absent_columns_with_nulls():
    payload = {"hourly": {"time": ["2025-01-01T00:00"], "pm2_5": [7.1]}}
    frame = ingest._hourly_frame(payload, ["pm2_5", "ozone"])
    # A column the API dropped becomes NaN; the validation task decides whether
    # that is fatal, so ingestion itself stays permissive.
    assert frame["ozone"].isna().all()


def test_hourly_frame_rejects_malformed_payload():
    with pytest.raises(ValueError, match="hourly.time"):
        ingest._hourly_frame({"hourly": {}}, ["pm2_5"])


def test_snapshot_is_present_and_usable():
    """The bundled fallback must be committed, or the offline path is a lie."""
    snapshot = config.data_dir() / config.SNAPSHOT_CSV
    assert snapshot.exists(), f"missing fallback snapshot at {snapshot}"

    frame = ingest.load_snapshot()
    assert len(frame) > 200
    for column in config.POLLUTANT_COLUMNS + config.WEATHER_COLUMNS:
        assert column in frame.columns


def test_ingest_falls_back_to_snapshot_when_api_fails(pipeline_dirs, monkeypatch):
    def explode(**_kwargs):
        raise ConnectionError("simulated outage")

    monkeypatch.setattr(ingest, "fetch_live", explode)

    raw_path = ingest.ingest_data()
    frame = pd.read_csv(raw_path)

    assert (frame["source"] == "snapshot").all()
    assert len(frame) > 200


def test_ingest_marks_live_data_as_live(pipeline_dirs, monkeypatch, hourly_frame):
    monkeypatch.setattr(ingest, "fetch_live", lambda **_kwargs: hourly_frame.copy())

    raw_path = ingest.ingest_data()
    frame = pd.read_csv(raw_path)

    assert (frame["source"] == "open-meteo").all()
    assert raw_path.endswith(config.RAW_CSV)
