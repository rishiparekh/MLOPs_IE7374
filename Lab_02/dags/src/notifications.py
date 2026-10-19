from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

ALERT_EMAIL = os.getenv("AQ_ALERT_EMAIL", "")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _summary_html() -> str:
    metrics = _read_json(config.working_dir() / config.CANDIDATE_METRICS)
    prediction = _read_json(config.model_dir() / config.LATEST_PREDICTION)

    if not metrics:
        return "<p>No metrics were produced by this run.</p>"

    rows = [
        ("Best model", metrics.get("best_model", "n/a")),
        ("RMSE", f"{metrics.get('rmse', float('nan')):.3f}"),
        ("MAE", f"{metrics.get('mae', float('nan')):.3f}"),
        ("R squared", f"{metrics.get('r2', float('nan')):.3f}"),
        ("Beats persistence baseline", "yes" if metrics.get("beats_baseline") else "no"),
        ("Promoted to production", "yes" if metrics.get("promoted") else "no"),
        ("Decision", metrics.get("decision_reason", "n/a")),
    ]
    if prediction:
        rows.append(
            (
                f"Forecast (+{prediction.get('horizon_hours', 1)}h)",
                f"{prediction.get('predicted')} {prediction.get('units', '')}",
            )
        )

    cells = "".join(
        f"<tr><td style='padding:4px 12px 4px 0'><b>{label}</b></td>"
        f"<td style='padding:4px 0'>{value}</td></tr>"
        for label, value in rows
    )
    return f"<table style='border-collapse:collapse'>{cells}</table>"


def _send(subject: str, html: str, context: dict) -> None:
    if not ALERT_EMAIL:
        logger.warning("AQ_ALERT_EMAIL is unset; skipping notification %r", subject)
        return

    # Imported here so the module stays importable without Airflow.
    from airflow.providers.smtp.operators.smtp import EmailOperator

    EmailOperator(
        task_id="notify",
        to=ALERT_EMAIL,
        subject=subject,
        html_content=html,
    ).execute(context=context)


def notify_success(context: dict) -> None:
    run_id = context.get("run_id", "unknown")
    html = (
        f"<h3>Air quality pipeline succeeded</h3>"
        f"<p>Location: {config.LOCATION_NAME}<br>Run: {run_id}</p>"
        f"{_summary_html()}"
    )
    _send(f"[OK] Air quality pipeline - {config.LOCATION_NAME}", html, context)


def notify_failure(context: dict) -> None:
    task_instance = context.get("task_instance")
    task_id = getattr(task_instance, "task_id", "unknown")
    exception = context.get("exception", "no exception recorded")

    validation = _read_json(config.working_dir() / config.VALIDATION_REPORT)
    violations = validation.get("violations") or []
    violation_html = (
        "<ul>" + "".join(f"<li>{item}</li>" for item in violations) + "</ul>"
        if violations
        else "<p>No data-validation violations recorded.</p>"
    )

    html = (
        f"<h3>Air quality pipeline failed</h3>"
        f"<p>Task: <code>{task_id}</code><br>Run: {context.get('run_id', 'unknown')}</p>"
        f"<p><b>Error:</b> {exception}</p>"
        f"{violation_html}"
    )
    _send(f"[FAILED] Air quality pipeline - task {task_id}", html, context)
