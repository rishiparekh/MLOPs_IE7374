from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 30


def _hourly_frame(payload: dict[str, Any], columns: list[str]) -> pd.DataFrame:
    hourly = payload.get("hourly") or {}
    if "time" not in hourly:
        raise ValueError("Open-Meteo response contained no 'hourly.time' block")

    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"], utc=True)})
    for column in columns:
        # Missing columns become NaN; validate_data decides if that is tolerable.
        frame[column] = hourly.get(column, [None] * len(frame))
    return frame


def fetch_live(past_days: int = 30) -> pd.DataFrame:
    common = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "UTC",
    }

    logger.info("Fetching air quality for %s (past_days=%s)", config.LOCATION_NAME, past_days)
    aq_response = requests.get(
        AIR_QUALITY_URL,
        params={**common, "hourly": ",".join(config.POLLUTANT_COLUMNS)},
        timeout=REQUEST_TIMEOUT,
    )
    aq_response.raise_for_status()

    logger.info("Fetching weather for %s", config.LOCATION_NAME)
    wx_response = requests.get(
        WEATHER_URL,
        params={**common, "hourly": ",".join(config.WEATHER_COLUMNS)},
        timeout=REQUEST_TIMEOUT,
    )
    wx_response.raise_for_status()

    air_quality = _hourly_frame(aq_response.json(), config.POLLUTANT_COLUMNS)
    weather = _hourly_frame(wx_response.json(), config.WEATHER_COLUMNS)

    merged = air_quality.merge(weather, on="time", how="inner").sort_values("time")
    return merged.reset_index(drop=True)


def load_snapshot() -> pd.DataFrame:
    snapshot_path = config.data_dir() / config.SNAPSHOT_CSV
    logger.warning("Falling back to bundled snapshot at %s", snapshot_path)
    frame = pd.read_csv(snapshot_path, parse_dates=["time"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


def ingest_data(past_days: int = 30) -> str:
    config.ensure_dirs()

    try:
        frame = fetch_live(past_days=past_days)
        source = "open-meteo"
    except Exception as exc:
        logger.warning("Live fetch failed (%s: %s); using snapshot", type(exc).__name__, exc)
        frame = load_snapshot()
        source = "snapshot"

    frame["source"] = source

    out_path = config.working_dir() / config.RAW_CSV
    frame.to_csv(out_path, index=False)
    logger.info("Ingested %s rows from %s -> %s", len(frame), source, out_path)
    return str(out_path)
