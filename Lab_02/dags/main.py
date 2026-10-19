
from __future__ import annotations

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import (
    BranchPythonOperator,
    PythonOperator,
)
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

# Airflow moved TriggerRule between 3.x point releases; accept either location
# so this DAG parses on the same image Lab 2 ships with and on newer ones.
try:
    from airflow.task.trigger_rule import TriggerRule
except ImportError:  # pragma: no cover
    from airflow.utils.trigger_rule import TriggerRule

from src import config
from src.features import build_features, split_data
from src.ingest import ingest_data
from src.model_development import (
    PROMOTE_TASK_ID,
    SKIP_TASK_ID,
    decide_promotion,
    evaluate_models,
    predict_latest,
    promote_model,
    skip_promotion,
    train_model,
)
from src.notifications import notify_failure, notify_success
from src.validate import validate_data

# ---------- Default args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    # Retries are for transient faults (a flaky API call), which is exactly
    # what the ingest task can hit. Validation failures raise
    # AirflowFailException and bypass this on purpose.
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=2),
    "on_failure_callback": notify_failure,
}

# ---------- DAG ----------
dag = DAG(
    dag_id="Airflow_Lab2_AirQuality",
    default_args=default_args,
    description=f"Hourly PM2.5 forecasting for {config.LOCATION_NAME}",
    schedule="@daily",
    # catchup=False matters more here than in Lab 2: the source API only serves
    # a recent window, so backfilling 2024 would train on data we cannot fetch.
    catchup=False,
    # The pipeline writes to shared paths (production.pkl); concurrent runs
    # would race each other.
    max_active_runs=1,
    tags=["mlops", "air-quality", "lab2"],
    owner_links={"Rishi Parekh": "https://github.com/rishiparekh"},
)

# ---------- Tasks ----------
start_task = BashOperator(
    task_id="start_pipeline",
    bash_command=f"echo 'Starting air-quality pipeline for {config.LOCATION_NAME}'",
    owner="Rishi Parekh",
    dag=dag,
)

ingest_task = PythonOperator(
    task_id="ingest_data",
    python_callable=ingest_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    # `.output` is the XCom reference to the upstream task's return value.
    # It carries a *path string*; the data itself lives on the shared volume.
    op_args=[ingest_task.output],
    # One shot: bad data will still be bad in two minutes.
    retries=0,
    dag=dag,
)

build_features_task = PythonOperator(
    task_id="build_features",
    python_callable=build_features,
    op_args=[validate_task.output],
    dag=dag,
)

split_task = PythonOperator(
    task_id="split_data",
    python_callable=split_data,
    op_args=[build_features_task.output],
    dag=dag,
)

# Fan-out: both candidates train in parallel from the same split.
train_tasks = [
    PythonOperator(
        task_id=f"train_{model_name}",
        python_callable=train_model,
        op_args=[split_task.output, model_name],
        dag=dag,
    )
    for model_name in ("ridge", "random_forest")
]

evaluate_task = PythonOperator(
    task_id="evaluate_models",
    python_callable=evaluate_models,
    op_args=[split_task.output],
    dag=dag,
)

decide_task = BranchPythonOperator(
    task_id="decide_promotion",
    python_callable=decide_promotion,
    op_args=[evaluate_task.output],
    dag=dag,
)

promote_task = PythonOperator(
    task_id=PROMOTE_TASK_ID,
    python_callable=promote_model,
    op_args=[evaluate_task.output],
    dag=dag,
)

skip_task = PythonOperator(
    task_id=SKIP_TASK_ID,
    python_callable=skip_promotion,
    op_args=[evaluate_task.output],
    dag=dag,
)

# Join. A branch leaves the untaken side in state `skipped`; the default
# ALL_SUCCESS rule would therefore skip everything downstream forever.
# NONE_FAILED_MIN_ONE_SUCCESS is what lets the graph reconverge.
promotion_complete = EmptyOperator(
    task_id="promotion_complete",
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

predict_task = PythonOperator(
    task_id="predict_latest",
    python_callable=predict_latest,
    op_args=[split_task.output],
    on_success_callback=notify_success,
    dag=dag,
)

trigger_dashboard = TriggerDagRunOperator(
    task_id="trigger_dashboard",
    trigger_dag_id="Airflow_Lab2_AirQuality_Flask",
    conf={"message": "Air-quality pipeline finished"},
    reset_dag_run=False,
    # Fire-and-forget: the dashboard DAG runs a blocking web server, so waiting
    # for it to "complete" would hang this run forever.
    wait_for_completion=False,
    # Launch the dashboard even on failure -- a status page is most useful
    # precisely when the pipeline broke.
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# ---------- Dependencies ----------
start_task >> ingest_task >> validate_task >> build_features_task >> split_task
split_task >> train_tasks >> evaluate_task
evaluate_task >> decide_task >> [promote_task, skip_task] >> promotion_complete
promotion_complete >> predict_task >> trigger_dashboard
