# Airflow Lab 2 — Air Quality Forecasting

A reimplementation of [Airflow Lab 2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_2)
against a different application. The reference lab predicts advertising clicks
from a frozen CSV; this one forecasts **next-hour PM2.5 concentration for
Boston, MA** from live Open-Meteo data.

Every orchestration objective of the original is preserved. The application
change is what makes several of them *mean* something.

---

## Why this use case

The reference lab schedules itself `@daily` but reads a CSV that never changes,
so every run trains an identical model on identical data. Swapping in a live
API makes the schedule real: each run sees hours the previous run did not.

It also introduces failure modes the original cannot exhibit:

| Concern | Reference lab | Here |
|---|---|---|
| Upstream availability | n/a (local file) | API can be down → explicit fallback |
| Data quality | assumed | validated, with a failing gate |
| Leakage | n/a | time series; three explicit guards |
| Model regression | promoted unconditionally | compared against production first |

---

## Objective mapping

| # | Lab 2 objective | Where it lives here |
|---|---|---|
| 1 | Orchestration separate from logic | [`dags/main.py`](dags/main.py) vs [`dags/src/`](dags/src/) |
| 2 | ML pipeline as a task graph | 13 tasks, see graph below |
| 3 | Artifacts passed by reference | tasks return paths; data travels via `/opt/airflow/working_data` |
| 4 | Task wiring with `.output` | `op_args=[ingest_task.output]` throughout |
| 5 | Model persistence | `model/production.pkl` + `production_metrics.json` |
| 6 | Cross-DAG orchestration | `TriggerDagRunOperator` → dashboard DAG, `ALL_DONE` |
| 7 | Email alerting | `on_failure_callback` on every task, `on_success_callback` on the last |
| 8 | Flask status page via Airflow REST API | [`dags/flask_api.py`](dags/flask_api.py), `POST /auth/token` + `GET /api/v2/dags/.../dagRuns` |
| 9 | Reproducible environment | [`docker-compose.yaml`](docker-compose.yaml), [`setup.sh`](setup.sh), `.env` |

Added beyond the original: a data-quality gate, parallel model training, a
branching promotion decision, a persistence baseline, and a unit test suite.

---

## Pipeline

```
start_pipeline
   └─> ingest_data ──> validate_data ──> build_features ──> split_data
                                                                │
                                              ┌─────────────────┴─────────────────┐
                                              │                                   │
                                        train_ridge                      train_random_forest
                                              │                                   │
                                              └─────────────────┬─────────────────┘
                                                                │
                                                        evaluate_models
                                                                │
                                                        decide_promotion          ← branch
                                                          ╱           ╲
                                              promote_model           skip_promotion
                                                          ╲           ╱
                                                       promotion_complete         ← join
                                                                │
                                                         predict_latest
                                                                │
                                                        trigger_dashboard
```

**Fan-out** (two models in parallel), **branch** (promote or not), and **join**
(`NONE_FAILED_MIN_ONE_SUCCESS`) are the three control-flow patterns the
original's straight line does not exercise.

---

## Quickstart

```bash
cd Lab_02
./setup.sh
```

Then:

| What | Where |
|---|---|
| Airflow UI | <http://localhost:8080> — `airflow` / `airflow` |
| Dashboard | <http://localhost:5555> — after the pipeline has run |
| Dashboard JSON | <http://localhost:5555/metrics> |

1. Unpause **`Airflow_Lab2_AirQuality`** in the UI and trigger it.
2. On completion it triggers **`Airflow_Lab2_AirQuality_Flask`**, which serves
   the dashboard.

`./setup.sh --clean` tears down volumes first. Without the flag your DAG run
history survives, which you usually want while iterating.

### Local development (no Docker)

The pipeline logic runs without Airflow — that is the point of keeping it in
`dags/src/`:

```bash
cd Lab_02
python -m venv lab_02
./lab_02/Scripts/python.exe -m pip install -r requirements.txt -r requirements-dev.txt
./lab_02/Scripts/python.exe -m pytest tests -q
```

On macOS/Linux use `lab_02/bin/python` instead of `lab_02/Scripts/python.exe`.

> **Windows note:** the venv is nested at `Lab_02/lab_02` deliberately. On a
> case-insensitive filesystem a top-level `lab_02` collides with `Lab_02`, and
> the `.gitignore` containing `*` that `venv` writes into its own directory
> would then hide the entire lab from git.

---

## Layout

```
Lab_02/
├── dags/
│   ├── main.py                  # pipeline DAG
│   ├── flask_api.py             # dashboard DAG + Flask app
│   ├── src/
│   │   ├── config.py            # env-driven paths and constants
│   │   ├── ingest.py            # Open-Meteo pull + snapshot fallback
│   │   ├── validate.py          # data-quality gate
│   │   ├── features.py          # lags, rolling stats, chronological split
│   │   ├── model_development.py # train / evaluate / promote / predict
│   │   └── notifications.py     # success and failure emails
│   ├── data/
│   │   └── air_quality_snapshot.csv   # 60 days of real Boston data
│   └── templates/
│       ├── success.html
│       └── failure.html
├── tests/
├── conftest.py
├── docker-compose.yaml
├── setup.sh
└── requirements.txt
```

---

## Design notes

**Paths come from environment variables.** `config.working_dir()` reads
`AQ_WORKING_DIR` rather than hardcoding `/opt/airflow/working_data`, which is
what lets the tests point the whole pipeline at a pytest `tmp_path`.

**Nothing creates directories at import time.** Airflow re-imports DAG modules
on every parse cycle, in the scheduler and in every worker. The reference lab's
module-level `os.makedirs(...)` therefore runs constantly; here it is
`config.ensure_dirs()`, called from inside tasks.

**Ingestion degrades instead of failing.** If Open-Meteo is unreachable,
`ingest_data()` falls back to the committed 60-day snapshot and tags the rows
`source=snapshot`. A lab that only works when the network does is a lab that
fails during grading.

**Validation failures bypass retries.** `validate_data` raises
`AirflowFailException`, which fails the task without consuming its retry
budget — retrying bad data just fails again two minutes later. `ingest_data`
keeps `retries=2` because *its* failures are transient.

**Three leakage guards**, all covered by tests:
- rolling windows are `.shift(1)`-ed so they end at the previous hour;
- the split is chronological, never `train_test_split(shuffle=True)`;
- the scaler is fit on the training slice only.

**Promotion is a branch, not an `if`.** Using `BranchPythonOperator` puts the
decision in the Airflow UI — you can see at a glance which runs shipped a model.

**A persistence baseline is always scored.** If the model cannot beat "next
hour looks like this hour", it has learned nothing worth deploying.

---

## Email setup

Optional. Leave `AQ_ALERT_EMAIL` empty in `.env` and the callbacks log a
warning instead of failing.

To enable, generate a [Gmail app password](https://support.google.com/accounts/answer/185833)
and set in `.env`:

```
AQ_ALERT_EMAIL=you@example.com
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-16-char-app-password
```

These reach Airflow as `AIRFLOW__SMTP__*` environment variables via
`docker-compose.yaml`. Unlike the reference lab, no password is written into a
version-controlled `airflow.cfg`.

---

## Airflow 3 gotchas found while getting this running

Three things broke on `apache/airflow:3.0.3` that are worth knowing before you
debug your own DAG. All three are fixed in this repo.

**1. Tasks queue forever with no error in the scheduler log.**
On Airflow 3 a task subprocess no longer talks to the metadata database — it
calls back over the **Task Execution API**, which defaults to
`http://localhost:8080/execution/`. That default is only correct when tasks run
in the same container as the apiserver. Under `LocalExecutor` they run in the
*scheduler* container, where nothing listens on 8080, so every task is queued,
retries four times, is SIGKILLed, and the run hangs at `running` forever. The
scheduler log shows only `Adopting or resetting orphaned tasks` every five
minutes; the real traceback is buried in the worker subprocess output. Fix:

```yaml
AIRFLOW__CORE__EXECUTION_API_SERVER_URL: http://airflow-apiserver:8080/execution/
```

`max_active_runs=1` then amplifies this — the stuck run holds the only slot, so
every later run queues behind it.

**2. `from airflow.task.trigger_rule import TriggerRule` does not exist on 3.0.3.**
The reference lab uses that path. The correct one here is
`airflow.utils.trigger_rule`; [`dags/main.py`](dags/main.py) tries both so it
parses on either.

**3. `_PIP_ADDITIONAL_REQUIREMENTS` was pure cost.** The base image already
ships pandas 2.2.3, numpy 2.2.6, scikit-learn 1.7.0, requests 2.32.4, joblib
1.5.1, and the `fab`, `smtp` and `standard` providers. Listing them only forced
pip to *downgrade* four packages on every container start, risking conflicts
with Airflow's own pinned set. It has been removed; `requirements.txt` still
pins versions for the local venv and CI, where there is no base image to
inherit from.

Also: on Windows, `setup.sh` writes `AIRFLOW_UID=50000` rather than `id -u`.
Git Bash reports a Windows-derived id (e.g. 197609) that is meaningless to a
Linux container, and Docker Desktop manages bind-mount ownership in its own VM.

---

## Known limitations

- **The Flask task blocks a worker slot** for as long as the server runs. This
  is inherited from the reference lab and is fine for a lab, but a real
  deployment would run the API as its own container rather than as an Airflow
  task.
- **Open-Meteo serves a limited history window**, which is why `catchup=False`
  matters here more than in the reference lab — backfilling 2024 would request
  data the API will not return.
- **Airflow is not installed in the local venv.** It is a Linux-targeted
  install; the venv exists to run the pipeline logic and tests. Airflow itself
  runs in Docker.
- **Unpausing the DAG immediately creates a `@daily` scheduled run.** That is
  correct behaviour, but it takes the single `max_active_runs` slot, so a run
  you trigger by hand right afterwards will sit in `queued` until it finishes.

---

## Testing

```bash
./lab_02/Scripts/python.exe -m pytest tests -q
```

37 tests covering ingestion fallback, every validation rule, the three leakage
guards, training, evaluation, the promotion branch, and prediction.
