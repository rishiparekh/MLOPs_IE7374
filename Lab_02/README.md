# Airflow Lab 2 - Air Quality Forecasting

An Airflow pipeline that forecasts next-hour PM2.5 for Boston, MA from live
Open-Meteo data, trains two candidate models, and only deploys a new one if it
beats the model already in production. A Flask dashboard reports the status of
the last run.

## Setup

```bash
cd Lab_02
./setup.sh
```

Run it from Git Bash or WSL on Windows. It creates the host directories, writes
a `.env`, and starts the containers.

| | |
|---|---|
| Airflow UI | <http://localhost:8080> (`airflow` / `airflow`) |
| Dashboard | <http://localhost:5555> |
| Dashboard JSON | <http://localhost:5555/metrics> |

Then unpause **`Airflow_Lab2_AirQuality`** in the UI and trigger it. When it
finishes it triggers **`Airflow_Lab2_AirQuality_Flask`**, which serves the
dashboard.

`./setup.sh --clean` drops the volumes first. Without it your DAG run history
survives a restart.

Useful commands:

```bash
docker compose logs -f airflow-scheduler
docker compose down
```

### Email (optional)

Leave `AQ_ALERT_EMAIL` empty in `.env` and notifications are skipped. To enable
them, set a [Gmail app password](https://support.google.com/accounts/answer/185833):

```
AQ_ALERT_EMAIL=you@example.com
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
```

### Running the tests

The pipeline logic in `dags/src/` runs without Airflow:

```bash
python -m venv lab_02
./lab_02/Scripts/python.exe -m pip install -r requirements.txt -r requirements-dev.txt
./lab_02/Scripts/python.exe -m pytest tests -q
```

Use `lab_02/bin/python` on macOS or Linux. The venv is nested inside `Lab_02/`
on purpose: a top-level `lab_02` would collide with `Lab_02` on a
case-insensitive filesystem.

Airflow itself is not installed in the venv; it only runs in Docker.

## The data

Hourly observations for Boston (42.36, -71.06) from two free Open-Meteo
endpoints, merged on timestamp:

- **Air quality** - `pm2_5`, `pm10`, `carbon_monoxide`, `nitrogen_dioxide`,
  `sulphur_dioxide`, `ozone`
- **Weather** - `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`,
  `surface_pressure`

No API key is needed. Each run pulls the last 30 days, roughly 744 rows.

`dags/data/air_quality_snapshot.csv` holds a committed 61-day snapshot (1,464
rows) used automatically when the API is unreachable, so the pipeline and the
tests work offline. Rows are tagged `source=open-meteo` or `source=snapshot`.

PM2.5 in that snapshot averages 8.25 ug/m3 and ranges from 0.2 to 30.7, so this
is clean-air data with occasional mild spikes rather than a pollution event.

The target is PM2.5 one hour ahead. From the raw columns the pipeline builds 23
features: the current-hour readings, PM2.5 lags at 1, 2, 3, 6 and 24 hours,
trailing 3h and 24h rolling mean and standard deviation, and sin/cos encodings
of hour-of-day and day-of-week.

## How it works

```
start_pipeline
   └─> ingest_data ──> validate_data ──> build_features ──> split_data
                                                                │
                                              ┌─────────────────┴─────────────────┐
                                        train_ridge                      train_random_forest
                                              └─────────────────┬─────────────────┘
                                                                │
                                                        evaluate_models
                                                                │
                                                        decide_promotion
                                                          ╱           ╲
                                              promote_model           skip_promotion
                                                          ╲           ╱
                                                       promotion_complete
                                                                │
                                                         predict_latest
                                                                │
                                                        trigger_dashboard
```

| Task | What it does |
|---|---|
| `ingest_data` | Pulls both Open-Meteo endpoints, merges them, falls back to the snapshot on any failure. Writes `raw.csv`. |
| `validate_data` | Checks schema, row count, null fractions, value ranges and duplicate timestamps. Writes `validation_report.json` and fails the run if anything breaks. |
| `build_features` | Builds lag, rolling and calendar features and the target column. |
| `split_data` | Chronological 80/20 split, then fits a `StandardScaler` on the training slice only. |
| `train_ridge`, `train_random_forest` | Train in parallel on the same split and pickle a candidate each. |
| `evaluate_models` | Scores both on the held-out tail, picks the lower RMSE, and also scores a "next hour equals this hour" baseline. Writes `candidate_metrics.json`. |
| `decide_promotion` | Branches: promotes only if RMSE improves on production by more than 1%, otherwise keeps the incumbent. |
| `promote_model` | Copies the winner to `model/production.pkl` and records its metrics. |
| `predict_latest` | Loads the production model and forecasts the most recent hour into `latest_prediction.json`. |
| `trigger_dashboard` | Starts the Flask DAG. Runs even when something upstream failed. |

Tasks pass file paths through XCom rather than DataFrames; the data itself
lives on the `working_data` and `model` volumes.

The dashboard queries the Airflow REST API for the state of the last pipeline
run and renders it alongside the metrics and forecast those tasks wrote.

### Layout

```
Lab_02/
├── dags/
│   ├── main.py                  # pipeline DAG
│   ├── flask_api.py             # dashboard DAG + Flask app
│   ├── src/
│   │   ├── config.py            # paths and constants from env vars
│   │   ├── ingest.py
│   │   ├── validate.py
│   │   ├── features.py
│   │   ├── model_development.py
│   │   └── notifications.py
│   ├── data/air_quality_snapshot.csv
│   └── templates/               # success.html, failure.html
├── tests/
├── docker-compose.yaml
└── setup.sh
```

## Notes

- Unpausing the DAG immediately creates a scheduled run, and `max_active_runs`
  is 1, so a manual trigger right after will wait in `queued` until it
  finishes.
- The dashboard DAG holds a worker slot for as long as the Flask server runs.
- Open-Meteo only serves a recent window of history, which is why `catchup` is
  off.
