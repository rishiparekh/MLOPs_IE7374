"""Tests for training, evaluation, promotion and prediction."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from src import config, features, model_development as md


@pytest.fixture
def splits_path(pipeline_dirs, hourly_frame):
    """Run the real feature + split pipeline so tests exercise real artifacts."""
    engineered = features.build_feature_frame(hourly_frame)
    features_file = pipeline_dirs["working"] / config.FEATURES_PKL
    with open(features_file, "wb") as handle:
        pickle.dump(engineered, handle)
    return features.split_data(str(features_file))


# ---------- metrics ----------

def test_score_is_perfect_on_identical_arrays():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = md.score(values, values)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


# ---------- training ----------

@pytest.mark.parametrize("model_name", sorted(md.MODEL_REGISTRY))
def test_train_model_writes_a_loadable_artifact(splits_path, model_name):
    path = md.train_model(splits_path, model_name)
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    assert hasattr(model, "predict")


def test_train_model_rejects_unknown_name(splits_path):
    with pytest.raises(ValueError, match="Unknown model"):
        md.train_model(splits_path, "not_a_real_model")


# ---------- evaluation ----------

def test_evaluate_scores_every_candidate(pipeline_dirs, splits_path):
    for model_name in md.MODEL_REGISTRY:
        md.train_model(splits_path, model_name)

    metrics_path = md.evaluate_models(splits_path)
    metrics = json.loads(open(metrics_path).read())

    assert set(metrics["all_models"]) == set(md.MODEL_REGISTRY)
    assert metrics["best_model"] in md.MODEL_REGISTRY
    # The winner is the lowest-RMSE candidate, by definition.
    assert metrics["rmse"] == min(entry["rmse"] for entry in metrics["all_models"].values())
    assert metrics["n_test"] > 0
    assert "persistence_baseline" in metrics


def test_evaluate_raises_when_nothing_was_trained(pipeline_dirs, splits_path):
    with pytest.raises(RuntimeError, match="No candidate models"):
        md.evaluate_models(splits_path)


def test_persistence_baseline_is_in_original_units(pipeline_dirs, splits_path):
    """
    The baseline predicts "next hour == this hour". Because features are
    standardised, scoring it requires inverting the scaler first -- forgetting
    that produces a nonsense RMSE in the hundreds and a baseline nothing can
    beat.
    """
    md.train_model(splits_path, "ridge")
    metrics = json.loads(open(md.evaluate_models(splits_path)).read())

    with open(splits_path, "rb") as handle:
        splits = pickle.load(handle)

    # A sane baseline RMSE is on the order of the target's own scale.
    assert metrics["persistence_baseline"]["rmse"] < 5 * float(np.std(splits["y_test"]))


# ---------- promotion policy (implemented by you) ----------

def test_promotes_when_nothing_is_in_production():
    candidate = {"rmse": 2.0, "mae": 1.5, "r2": 0.8, "beats_baseline": True}
    promote, reason = md.should_promote(candidate, None)
    assert promote is True
    assert isinstance(reason, str) and reason


def test_promotes_a_clearly_better_model():
    candidate = {"rmse": 1.0, "mae": 0.8, "r2": 0.92, "beats_baseline": True}
    production = {"rmse": 2.0, "mae": 1.6, "r2": 0.78, "beats_baseline": True}
    promote, reason = md.should_promote(candidate, production)
    assert promote is True
    assert isinstance(reason, str) and reason


def test_rejects_a_clear_regression():
    candidate = {"rmse": 3.0, "mae": 2.4, "r2": 0.55, "beats_baseline": True}
    production = {"rmse": 1.0, "mae": 0.8, "r2": 0.92, "beats_baseline": True}
    promote, reason = md.should_promote(candidate, production)
    assert promote is False
    assert isinstance(reason, str) and reason


# ---------- branching + promotion mechanics ----------

def test_decide_promotion_returns_a_real_task_id(pipeline_dirs, splits_path):
    md.train_model(splits_path, "ridge")
    metrics_path = md.evaluate_models(splits_path)

    next_task = md.decide_promotion(metrics_path)

    assert next_task in (md.PROMOTE_TASK_ID, md.SKIP_TASK_ID)
    # The decision and its justification are persisted for the dashboard.
    metrics = json.loads(open(metrics_path).read())
    assert "promoted" in metrics
    assert metrics["decision_reason"]


def test_promote_model_installs_the_winner(pipeline_dirs, splits_path):
    md.train_model(splits_path, "ridge")
    metrics_path = md.evaluate_models(splits_path)

    md.promote_model(metrics_path)

    assert (pipeline_dirs["model"] / config.PRODUCTION_MODEL).exists()
    assert (pipeline_dirs["model"] / config.PRODUCTION_METRICS).exists()


def test_predict_latest_writes_a_forecast(pipeline_dirs, splits_path):
    md.train_model(splits_path, "ridge")
    md.promote_model(md.evaluate_models(splits_path))

    md.predict_latest(splits_path)

    payload = json.loads((pipeline_dirs["model"] / config.LATEST_PREDICTION).read_text())
    assert payload["target"] == config.TARGET
    assert isinstance(payload["predicted"], float)
    assert payload["absolute_error"] >= 0


def test_predict_latest_is_a_no_op_without_a_production_model(pipeline_dirs, splits_path):
    result = md.predict_latest(splits_path)
    assert result == "no-production-model"
