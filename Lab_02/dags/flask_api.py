from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pendulum
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from flask import Flask, jsonify, redirect, render_template

from src import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://airflow-apiserver:8080")
AF_USER = os.getenv("AIRFLOW_USERNAME", "airflow")
AF_PASS = os.getenv("AIRFLOW_PASSWORD", "airflow")
TARGET_DAG_ID = os.getenv("TARGET_DAG_ID", "Airflow_Lab2_AirQuality")
FLASK_PORT = int(os.getenv("AQ_FLASK_PORT", "5555"))

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))

_CACHED_TOKEN: str | None = None
_TOKEN_EXPIRY: float = 0.0


def get_auth_token() -> str | None:
    global _CACHED_TOKEN, _TOKEN_EXPIRY

    if _CACHED_TOKEN and time.time() < _TOKEN_EXPIRY - 60:
        return _CACHED_TOKEN

    try:
        response = requests.post(
            f"{WEBSERVER}/auth/token",
            json={"username": AF_USER, "password": AF_PASS},
            timeout=10,
        )
    except requests.RequestException as exc:
        logger.error("Could not reach Airflow auth endpoint: %s", exc)
        return None

    if response.status_code in (200, 201):
        _CACHED_TOKEN = response.json().get("access_token")
        _TOKEN_EXPIRY = time.time() + 3600
        return _CACHED_TOKEN

    logger.error("Auth failed: %s %s", response.status_code, response.text)
    return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_pipeline_artifacts() -> dict:
    return {
        "metrics": _read_json(config.working_dir() / config.CANDIDATE_METRICS),
        "production": _read_json(config.model_dir() / config.PRODUCTION_METRICS),
        "prediction": _read_json(config.model_dir() / config.LATEST_PREDICTION),
        "validation": _read_json(config.working_dir() / config.VALIDATION_REPORT),
    }


def get_latest_run_info() -> tuple[bool, dict]:
    info = {
        "dag_id": TARGET_DAG_ID,
        "location": config.LOCATION_NAME,
        "state": "unknown",
        "run_id": "N/A",
        "logical_date": "N/A",
        "start_date": "N/A",
        "end_date": "N/A",
        "note": "",
        **get_pipeline_artifacts(),
    }

    token = get_auth_token()
    if not token:
        info["note"] = "Could not authenticate against the Airflow API."
        return False, info

    url = f"{WEBSERVER}/api/v2/dags/{TARGET_DAG_ID}/dagRuns?order_by=-run_after&limit=1"
    try:
        response = requests.get(
            url,
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except requests.RequestException as exc:
        info["note"] = f"Network issue reaching Airflow: {exc}"
        return False, info

    if response.status_code != 200:
        info["note"] = f"Airflow API returned {response.status_code}"
        return False, info

    runs = response.json().get("dag_runs", [])
    if not runs:
        info["note"] = "No runs yet. Trigger the pipeline DAG to populate this page."
        return False, info

    run = runs[0]
    state = run.get("state", "unknown")
    info.update(
        {
            "state": state,
            "run_id": run.get("dag_run_id", "N/A"),
            "logical_date": run.get("logical_date") or run.get("run_after", "N/A"),
            "start_date": run.get("start_date", "N/A"),
            "end_date": run.get("end_date", "N/A"),
        }
    )

    if state in ("queued", "running"):
        info["note"] = "Pipeline is still in progress."
        return False, info

    return state == "success", info


@app.route("/")
def index():
    ok, _ = get_latest_run_info()
    return redirect("/success" if ok else "/failure")


@app.route("/success")
def success():
    _, info = get_latest_run_info()
    return render_template("success.html", **info)


@app.route("/failure")
def failure():
    _, info = get_latest_run_info()
    return render_template("failure.html", **info)


@app.route("/metrics")
def metrics():
    ok, info = get_latest_run_info()
    return jsonify({"healthy": ok, **info})


@app.route("/health")
def health():
    return "ok", 200


def start_flask_app() -> None:
    # Blocks the worker slot for as long as the server runs.
    logger.info("Starting dashboard on 0.0.0.0:%s (Airflow at %s)", FLASK_PORT, WEBSERVER)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)


default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

dag = DAG(
    dag_id="Airflow_Lab2_AirQuality_Flask",
    default_args=default_args,
    description="Serves the air-quality pipeline status dashboard",
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=["mlops", "air-quality", "lab2", "dashboard"],
)

PythonOperator(
    task_id="start_dashboard",
    python_callable=start_flask_app,
    dag=dag,
)
