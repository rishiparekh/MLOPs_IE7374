"""
Task 3 & 4: feature engineering and a chronological train/test split.

This is where the air-quality problem earns its place over Lab 2's advertising
data. Time series introduces a failure mode that tabular classification does
not: **leakage**. Three guards against it live in this module.

1. Lag/rolling features are built only from strictly past observations. A
   rolling mean that includes the current row leaks the thing you are about to
   predict into its own feature.
2. The split is chronological, never `train_test_split(shuffle=True)`. Shuffling
   lets the model train on 3pm and test on 2pm of the same day, which inflates
   the score and tells you nothing about forecasting.
3. The scaler is fit on the training slice only, then applied to test. Fitting
   on everything leaks test-set distribution into training.
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)

#: Hours back to look for lag features.
LAG_HOURS = [1, 2, 3, 6, 24]
#: Window sizes (hours) for trailing rolling statistics.
ROLLING_WINDOWS = [3, 24]


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Pure feature construction -- no IO, so it is directly unit-testable.

    Returns a frame containing the engineered features plus a `target` column
    holding pm2_5 `HORIZON_HOURS` into the future.
    """
    frame = frame.sort_values("time").reset_index(drop=True).copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)

    engineered = pd.DataFrame({"time": frame["time"]})

    # Current-hour observations are legitimate inputs: at prediction time we
    # know *now*, we just do not know the next hour.
    for column in config.POLLUTANT_COLUMNS + config.WEATHER_COLUMNS:
        engineered[column] = pd.to_numeric(frame[column], errors="coerce")

    # Lags of the target.
    for lag in LAG_HOURS:
        engineered[f"{config.TARGET}_lag_{lag}h"] = engineered[config.TARGET].shift(lag)

    # Trailing statistics. `.shift(1)` BEFORE rolling is the leakage guard: it
    # makes every window end at the previous hour, excluding the current row.
    past_target = engineered[config.TARGET].shift(1)
    for window in ROLLING_WINDOWS:
        engineered[f"{config.TARGET}_roll_mean_{window}h"] = past_target.rolling(window).mean()
        engineered[f"{config.TARGET}_roll_std_{window}h"] = past_target.rolling(window).std()

    # Cyclical time encoding: hour 23 and hour 0 are adjacent, which a raw
    # integer column cannot express but sin/cos pairs can.
    hour = engineered["time"].dt.hour
    day_of_week = engineered["time"].dt.dayofweek
    engineered["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    engineered["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    engineered["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    engineered["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # The label: pm2_5 HORIZON_HOURS ahead.
    engineered["target"] = engineered[config.TARGET].shift(-config.HORIZON_HOURS)

    before = len(engineered)
    engineered = engineered.dropna().reset_index(drop=True)
    logger.info("Feature frame: %s rows (dropped %s incomplete)", len(engineered), before - len(engineered))
    return engineered


def feature_columns(engineered: pd.DataFrame) -> list[str]:
    """Everything except the timestamp and the label."""
    return [column for column in engineered.columns if column not in ("time", "target")]


def build_features(validated_path: str) -> str:
    """Airflow task entry point. Returns path to the pickled feature frame."""
    config.ensure_dirs()

    frame = pd.read_csv(validated_path, parse_dates=["time"])
    engineered = build_feature_frame(frame)

    out_path = config.working_dir() / config.FEATURES_PKL
    with open(out_path, "wb") as handle:
        pickle.dump(engineered, handle)

    logger.info("Built %s features -> %s", len(feature_columns(engineered)), out_path)
    return str(out_path)


def split_data(features_path: str) -> str:
    """
    Airflow task entry point: chronological split + train-only scaling.

    Replaces Lab 2's vestigial `separate_data_outputs()` passthrough with a task
    that actually does something.
    """
    config.ensure_dirs()

    with open(features_path, "rb") as handle:
        engineered: pd.DataFrame = pickle.load(handle)

    columns = feature_columns(engineered)
    split_index = int(len(engineered) * (1 - config.TEST_FRACTION))

    train = engineered.iloc[:split_index]
    test = engineered.iloc[split_index:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[columns].to_numpy())
    X_test = scaler.transform(test[columns].to_numpy())
    y_train = train["target"].to_numpy()
    y_test = test["target"].to_numpy()

    payload = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": columns,
        "scaler": scaler,
        "train_end": str(train["time"].iloc[-1]),
        "test_start": str(test["time"].iloc[0]),
        "test_timestamps": [str(value) for value in test["time"]],
    }

    out_path = config.working_dir() / config.SPLITS_PKL
    with open(out_path, "wb") as handle:
        pickle.dump(payload, handle)

    logger.info(
        "Chronological split: train=%s (to %s), test=%s (from %s)",
        len(y_train), payload["train_end"], len(y_test), payload["test_start"],
    )
    return str(out_path)
