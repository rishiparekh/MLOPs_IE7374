"""
Tasks 5-8: train competing models, evaluate them, and decide what ships.

Lab 2 trained one LogisticRegression, printed a score, and promoted it
unconditionally. That last part is the real gap: a pipeline that overwrites
production with whatever it just trained will eventually overwrite a good model
with a bad one, on a schedule, unattended.

This module separates three concerns that Lab 2 conflated:

    train      -- fit a candidate, save it; knows nothing about production
    evaluate   -- score candidates on held-out data, pick the best candidate
    promote    -- compare best candidate against production, decide, then act
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config

logger = logging.getLogger(__name__)

#: Task ids the branch operator chooses between. Imported by the DAG so the
#: strings cannot drift apart.
PROMOTE_TASK_ID = "promote_model"
SKIP_TASK_ID = "skip_promotion"

#: Candidate architectures, trained in parallel as separate Airflow tasks.
MODEL_REGISTRY = {
    "ridge": lambda: Ridge(alpha=1.0),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    ),
}


def _load_splits(splits_path: str) -> dict:
    with open(splits_path, "rb") as handle:
        return pickle.load(handle)


def _candidate_path(model_name: str) -> Path:
    return config.working_dir() / f"candidate_{model_name}.pkl"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regression metrics. RMSE is the headline number; MAE and R2 are context."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_model(splits_path: str, model_name: str) -> str:
    """Airflow task entry point. Fits one candidate and returns its path."""
    config.ensure_dirs()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of {sorted(MODEL_REGISTRY)}"
        )

    splits = _load_splits(splits_path)
    model = MODEL_REGISTRY[model_name]()
    model.fit(splits["X_train"], splits["y_train"])

    out_path = _candidate_path(model_name)
    with open(out_path, "wb") as handle:
        pickle.dump(model, handle)

    logger.info("Trained %s on %s rows -> %s", model_name, len(splits["y_train"]), out_path)
    return str(out_path)


def evaluate_models(splits_path: str) -> str:
    """
    Airflow task entry point. Scores every trained candidate and records the
    winner in candidate_metrics.json.

    A persistence-baseline score is included for honesty: if a model cannot beat
    "next hour looks like this hour", it has learned nothing worth shipping.
    """
    config.ensure_dirs()

    splits = _load_splits(splits_path)
    X_test, y_test = splits["X_test"], splits["y_test"]

    results: dict[str, dict[str, float]] = {}
    for model_name in MODEL_REGISTRY:
        candidate = _candidate_path(model_name)
        if not candidate.exists():
            logger.warning("No trained artifact for %s; skipping", model_name)
            continue
        with open(candidate, "rb") as handle:
            model = pickle.load(handle)
        results[model_name] = score(y_test, model.predict(X_test))
        logger.info("%s -> %s", model_name, results[model_name])

    if not results:
        raise RuntimeError("No candidate models were trained; nothing to evaluate")

    # Persistence baseline: predict "next hour == this hour". Because the
    # features were standardised we cannot read pm2_5 straight off X_test, so
    # the baseline is scored against the lag-1 column in its original units.
    lag_1h_index = splits["feature_names"].index(f"{config.TARGET}_lag_1h")
    scaler = splits["scaler"]
    lag_1h_original = (
        X_test[:, lag_1h_index] * scaler.scale_[lag_1h_index] + scaler.mean_[lag_1h_index]
    )
    baseline = score(y_test, lag_1h_original)

    best_name = min(results, key=lambda name: results[name]["rmse"])
    metrics = {
        "best_model": best_name,
        **results[best_name],
        "all_models": results,
        "persistence_baseline": baseline,
        "beats_baseline": bool(results[best_name]["rmse"] < baseline["rmse"]),
        "n_train": int(len(splits["y_train"])),
        "n_test": int(len(y_test)),
        "n_features": int(len(splits["feature_names"])),
        "train_end": splits["train_end"],
        "test_start": splits["test_start"],
        "target": config.TARGET,
        "horizon_hours": config.HORIZON_HOURS,
    }

    out_path = config.working_dir() / config.CANDIDATE_METRICS
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Best candidate: %s (RMSE %.4f) -> %s", best_name, metrics["rmse"], out_path)
    return str(out_path)


# --------------------------------------------------------------------------
# Promotion policy
# --------------------------------------------------------------------------

#: Guidance for the policy below: RMSE differences smaller than this are mostly
#: resampling noise on a test set of a few hundred hours.
MIN_RELATIVE_IMPROVEMENT = 0.01  # 1%


def should_promote(candidate: dict, production: dict | None) -> tuple[bool, str]:
    """
    Decide whether `candidate` should replace the currently-deployed model.

    Args:
        candidate: metrics dict from `evaluate_models` -- has "rmse", "mae",
            "r2", and "beats_baseline" (bool).
        production: the same shape for the live model, or None if nothing has
            ever been promoted.

    Returns:
        (promote, reason) -- the reason string is surfaced on the Flask
        dashboard and in the notification email, so write it for a human.
    """
    if not production:
        return (True, "First deployed model: nothing in production to compare against.")

    # RMSE is an ERROR, so lower is better. `production - candidate` is
    # positive exactly when the candidate improves on the incumbent; dividing
    # by the incumbent turns it into a relative improvement.
    improvement = (production["rmse"] - candidate["rmse"]) / production["rmse"]

    if improvement > MIN_RELATIVE_IMPROVEMENT:
        return (
            True,
            f"RMSE improved by {improvement:.1%}, clearing the "
            f"{MIN_RELATIVE_IMPROVEMENT:.0%} threshold "
            f"({candidate['rmse']:.3f} new vs {production['rmse']:.3f} in production).",
        )

    if improvement < -MIN_RELATIVE_IMPROVEMENT:
        return (
            False,
            f"Regression: RMSE got {abs(improvement):.1%} worse "
            f"({candidate['rmse']:.3f} new vs {production['rmse']:.3f} in production). "
            f"Keeping the incumbent.",
        )

    return (
        False,
        f"RMSE change of {improvement:+.1%} is within the "
        f"{MIN_RELATIVE_IMPROVEMENT:.0%} noise threshold; not worth redeploying "
        f"(RMSE {candidate['rmse']:.3f} new vs {production['rmse']:.3f} in production).",
    )
    
    # raise NotImplementedError("should_promote() has not been implemented yet")


def decide_promotion(candidate_metrics_path: str) -> str:
    """
    Airflow BranchPythonOperator callable. Returns the id of the next task.

    Branching (rather than an `if` inside the promote task) means the decision
    is visible in the Airflow UI: you can see at a glance which runs shipped a
    model and which did not.
    """
    config.ensure_dirs()

    candidate = _read_json(Path(candidate_metrics_path)) or {}
    production = _read_json(config.model_dir() / config.PRODUCTION_METRICS)

    promote, reason = should_promote(candidate, production)

    decision = {**candidate, "promoted": promote, "decision_reason": reason}
    Path(candidate_metrics_path).write_text(json.dumps(decision, indent=2), encoding="utf-8")

    logger.info("Promotion decision: %s (%s)", "PROMOTE" if promote else "SKIP", reason)
    return PROMOTE_TASK_ID if promote else SKIP_TASK_ID


def promote_model(candidate_metrics_path: str) -> str:
    """Copy the winning candidate into the production slot and record its metrics."""
    config.ensure_dirs()

    metrics = _read_json(Path(candidate_metrics_path)) or {}
    best_name = metrics.get("best_model")
    if not best_name:
        raise RuntimeError("Candidate metrics did not name a best_model")

    source = _candidate_path(best_name)
    destination = config.model_dir() / config.PRODUCTION_MODEL
    shutil.copyfile(source, destination)

    metrics_path = config.model_dir() / config.PRODUCTION_METRICS
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(
        "Promoted %s -> %s (RMSE %.4f)",
        best_name, destination, metrics.get("rmse", float("nan")),
    )
    return str(destination)


def skip_promotion(candidate_metrics_path: str) -> str:
    """No-op branch: production keeps serving the incumbent model."""
    metrics = _read_json(Path(candidate_metrics_path)) or {}
    reason = metrics.get("decision_reason", "no reason recorded")
    logger.info("Kept incumbent production model. Reason: %s", reason)
    return reason


def predict_latest(splits_path: str) -> str:
    """
    Final task: load whatever is in production and forecast the most recent hour.

    This proves the artifact on disk is loadable and usable -- the same check
    Lab 2's `load_model()` performed, but writing a JSON the dashboard renders
    instead of printing to a log nobody reads.
    """
    config.ensure_dirs()

    model_path = config.model_dir() / config.PRODUCTION_MODEL
    if not model_path.exists():
        logger.warning("No production model yet; skipping prediction")
        return "no-production-model"

    splits = _load_splits(splits_path)
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)

    latest_features = splits["X_test"][-1].reshape(1, -1)
    prediction = float(model.predict(latest_features)[0])
    actual = float(splits["y_test"][-1])

    payload = {
        "location": config.LOCATION_NAME,
        "target": config.TARGET,
        "horizon_hours": config.HORIZON_HOURS,
        "as_of": splits["test_timestamps"][-1],
        "predicted": round(prediction, 3),
        "actual": round(actual, 3),
        "absolute_error": round(abs(prediction - actual), 3),
        "units": "ug/m3",
    }

    out_path = config.model_dir() / config.LATEST_PREDICTION
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Latest forecast: %.2f ug/m3 (actual %.2f)", prediction, actual)
    return str(out_path)
