"""
Tests for feature engineering and splitting.

The leakage tests are the important ones. A leaking pipeline still trains, still
produces a model, and still reports a great score -- there is no exception to
catch. Only an explicit assertion will tell you.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from src import config, features


def test_target_is_the_future_value(hourly_frame):
    engineered = features.build_feature_frame(hourly_frame)
    # Row i's target must equal pm2_5 at row i+HORIZON_HOURS of the same frame.
    assert engineered["target"].iloc[0] == pytest.approx(
        engineered["pm2_5"].iloc[config.HORIZON_HOURS]
    )


def test_lag_features_look_backwards(hourly_frame):
    engineered = features.build_feature_frame(hourly_frame)
    for lag in features.LAG_HOURS:
        column = f"pm2_5_lag_{lag}h"
        assert column in engineered.columns
        assert engineered[column].iloc[lag] == pytest.approx(engineered["pm2_5"].iloc[0])


def test_rolling_mean_excludes_the_current_row(hourly_frame):
    """
    The leakage guard. A 3h rolling mean at row i must average rows i-3..i-1.
    If it included row i, the feature would contain the value the model is
    partly being asked to extrapolate from -- and in a target-derived feature
    that is a direct path to an inflated score.
    """
    engineered = features.build_feature_frame(hourly_frame)
    index = 100
    expected = engineered["pm2_5"].iloc[index - 3 : index].mean()
    actual = engineered["pm2_5_roll_mean_3h"].iloc[index]
    assert actual == pytest.approx(expected)

    including_current = engineered["pm2_5"].iloc[index - 2 : index + 1].mean()
    assert actual != pytest.approx(including_current)


def test_cyclical_encoding_wraps_around(hourly_frame):
    engineered = features.build_feature_frame(hourly_frame)
    # Hour 0 and hour 23 must be near-neighbours in (sin, cos) space.
    by_hour = engineered.assign(hour=engineered["time"].dt.hour)
    hour_0 = by_hour[by_hour["hour"] == 0].iloc[0]
    hour_23 = by_hour[by_hour["hour"] == 23].iloc[0]
    distance = np.hypot(
        hour_0["hour_sin"] - hour_23["hour_sin"],
        hour_0["hour_cos"] - hour_23["hour_cos"],
    )
    assert distance < 0.3


def test_no_nulls_survive(hourly_frame):
    engineered = features.build_feature_frame(hourly_frame)
    assert not engineered.isna().any().any()
    # The warm-up rows needed by the longest lag/window must be gone.
    assert len(engineered) < len(hourly_frame)


def test_split_is_chronological(pipeline_dirs, hourly_frame):
    engineered = features.build_feature_frame(hourly_frame)
    features_path = pipeline_dirs["working"] / config.FEATURES_PKL
    with open(features_path, "wb") as handle:
        pickle.dump(engineered, handle)

    splits_path = features.split_data(str(features_path))
    with open(splits_path, "rb") as handle:
        splits = pickle.load(handle)

    # Every training timestamp precedes every test timestamp.
    assert splits["train_end"] < splits["test_start"]
    assert len(splits["y_train"]) > len(splits["y_test"])
    assert len(splits["y_train"]) + len(splits["y_test"]) == len(engineered)


def test_scaler_is_fit_on_training_data_only(pipeline_dirs, hourly_frame):
    """
    If the scaler had seen the test slice, the *training* matrix would be
    centred on the full-series mean rather than the training mean. Checking
    that the training columns are centred is therefore a proxy for "the test
    set did not influence the fit".
    """
    engineered = features.build_feature_frame(hourly_frame)
    features_path = pipeline_dirs["working"] / config.FEATURES_PKL
    with open(features_path, "wb") as handle:
        pickle.dump(engineered, handle)

    with open(features.split_data(str(features_path)), "rb") as handle:
        splits = pickle.load(handle)

    assert np.allclose(splits["X_train"].mean(axis=0), 0, atol=1e-8)
    assert np.allclose(splits["X_train"].std(axis=0), 1, atol=1e-8)
    # The test slice is transformed, not re-fit, so it is NOT exactly centred.
    assert not np.allclose(splits["X_test"].mean(axis=0), 0, atol=1e-8)


def test_feature_columns_exclude_time_and_target(hourly_frame):
    engineered = features.build_feature_frame(hourly_frame)
    columns = features.feature_columns(engineered)
    assert "time" not in columns
    assert "target" not in columns
    assert "pm2_5_lag_1h" in columns
