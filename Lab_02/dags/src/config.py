"""
Shared configuration for the air-quality pipeline.

Every path is read from an environment variable with a Docker-friendly default.
That indirection is what lets the unit tests point the whole pipeline at a
pytest `tmp_path` instead of writing into /opt/airflow.

Note that nothing here creates directories at import time -- Airflow re-imports
DAG modules on every parse cycle (in the scheduler AND in every worker), so
import-time side effects are re-run constantly. Use `ensure_dirs()` from inside
a task instead.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------- Filesystem ----------
def working_dir() -> Path:
    """Scratch space for intermediate artifacts shared between tasks."""
    return Path(os.getenv("AQ_WORKING_DIR", "/opt/airflow/working_data"))


def model_dir() -> Path:
    """Durable location for trained models and their metrics."""
    return Path(os.getenv("AQ_MODEL_DIR", "/opt/airflow/model"))


def data_dir() -> Path:
    """Bundled, version-controlled data (the offline fallback snapshot)."""
    return Path(__file__).resolve().parent.parent / "data"


def ensure_dirs() -> None:
    """Create the writable directories. Call this from inside a task."""
    working_dir().mkdir(parents=True, exist_ok=True)
    model_dir().mkdir(parents=True, exist_ok=True)


# ---------- Well-known artifact names ----------
RAW_CSV = "raw.csv"
VALIDATION_REPORT = "validation_report.json"
FEATURES_PKL = "features.pkl"
SPLITS_PKL = "splits.pkl"
SNAPSHOT_CSV = "air_quality_snapshot.csv"

PRODUCTION_MODEL = "production.pkl"
PRODUCTION_METRICS = "production_metrics.json"
CANDIDATE_METRICS = "candidate_metrics.json"
LATEST_PREDICTION = "latest_prediction.json"

# ---------- Location ----------
LATITUDE = float(os.getenv("AQ_LATITUDE", "42.3601"))     # Boston, MA
LONGITUDE = float(os.getenv("AQ_LONGITUDE", "-71.0589"))
LOCATION_NAME = os.getenv("AQ_LOCATION_NAME", "Boston, MA")

# ---------- Modelling ----------
TARGET = "pm2_5"
#: How many hours ahead we forecast. 1 == "next hour".
HORIZON_HOURS = int(os.getenv("AQ_HORIZON_HOURS", "1"))
#: Fraction of the (chronologically ordered) rows held out for testing.
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
