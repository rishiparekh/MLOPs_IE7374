from __future__ import annotations

import os
from pathlib import Path


def working_dir() -> Path:
    return Path(os.getenv("AQ_WORKING_DIR", "/opt/airflow/working_data"))


def model_dir() -> Path:
    return Path(os.getenv("AQ_MODEL_DIR", "/opt/airflow/model"))


def data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def ensure_dirs() -> None:
    # Called from inside tasks, not at import. Airflow re-imports DAG modules
    # on every parse cycle.
    working_dir().mkdir(parents=True, exist_ok=True)
    model_dir().mkdir(parents=True, exist_ok=True)


RAW_CSV = "raw.csv"
VALIDATION_REPORT = "validation_report.json"
FEATURES_PKL = "features.pkl"
SPLITS_PKL = "splits.pkl"
SNAPSHOT_CSV = "air_quality_snapshot.csv"

PRODUCTION_MODEL = "production.pkl"
PRODUCTION_METRICS = "production_metrics.json"
CANDIDATE_METRICS = "candidate_metrics.json"
LATEST_PREDICTION = "latest_prediction.json"

LATITUDE = float(os.getenv("AQ_LATITUDE", "42.3601"))
LONGITUDE = float(os.getenv("AQ_LONGITUDE", "-71.0589"))
LOCATION_NAME = os.getenv("AQ_LOCATION_NAME", "Boston, MA")

TARGET = "pm2_5"
HORIZON_HOURS = int(os.getenv("AQ_HORIZON_HOURS", "1"))
TEST_FRACTION = float(os.getenv("AQ_TEST_FRACTION", "0.2"))
RANDOM_STATE = 42

POLLUTANT_COLUMNS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]
WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
]
