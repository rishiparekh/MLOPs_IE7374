from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Airflow puts the dags folder on sys.path at runtime; do the same so
# `from src import ...` resolves under pytest.
DAGS_DIR = Path(__file__).resolve().parent / "dags"
if str(DAGS_DIR) not in sys.path:
    sys.path.insert(0, str(DAGS_DIR))


@pytest.fixture
def pipeline_dirs(tmp_path, monkeypatch):
    working = tmp_path / "working_data"
    model = tmp_path / "model"
    monkeypatch.setenv("AQ_WORKING_DIR", str(working))
    monkeypatch.setenv("AQ_MODEL_DIR", str(model))
    working.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    return {"working": working, "model": model}


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    rows = 500
    rng = np.random.default_rng(0)
    times = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    hour_of_day = np.arange(rows) % 24

    # Daily cycle plus autocorrelated noise, so lag features carry signal.
    pm25 = 10 + 4 * np.sin(2 * np.pi * hour_of_day / 24)
    pm25 += np.cumsum(rng.normal(0, 0.25, rows)) * 0.3
    pm25 = np.clip(pm25, 0.1, None)

    return pd.DataFrame(
        {
            "time": times,
            "pm2_5": pm25,
            "pm10": pm25 * 1.6 + rng.normal(0, 0.5, rows),
            "carbon_monoxide": 180 + rng.normal(0, 12, rows),
            "nitrogen_dioxide": 14 + rng.normal(0, 3, rows),
            "sulphur_dioxide": 3 + rng.normal(0, 0.6, rows),
            "ozone": 55 + rng.normal(0, 8, rows),
            "temperature_2m": 8 + 6 * np.sin(2 * np.pi * hour_of_day / 24),
            "relative_humidity_2m": np.clip(62 + rng.normal(0, 8, rows), 0, 100),
            "wind_speed_10m": np.abs(rng.normal(12, 4, rows)),
            "surface_pressure": 1013 + rng.normal(0, 4, rows),
            "source": "synthetic",
        }
    )
