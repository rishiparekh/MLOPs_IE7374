#!/usr/bin/env bash
# Bootstrap the Airflow stack for the air-quality lab.
#
# On Windows run this from Git Bash or WSL. Unlike Lab 2's setup.sh this does
# NOT `docker compose down -v` by default -- blowing away the postgres volume
# every time also destroys your DAG run history, which you usually want to keep
# while iterating. Pass --clean when you genuinely want a fresh start.
set -euo pipefail

CLEAN=0
[[ "${1:-}" == "--clean" ]] && CLEAN=1

cd "$(dirname "$0")"

if [[ $CLEAN -eq 1 ]]; then
  echo "==> Tearing down existing stack and volumes"
  docker compose down -v || true
  rm -rf ./logs ./plugins ./working_data ./model
fi

echo "==> Creating host directories"
mkdir -p ./logs ./plugins ./config ./working_data ./model

echo "==> Writing .env"
# Airflow containers run as a non-root user; on Linux/macOS matching the host
# UID keeps the bind-mounted logs/ and model/ directories writable from both
# sides.
#
# On Windows this does NOT apply. Git Bash reports a Windows-derived id (e.g.
# 197609) that means nothing to a Linux container, and Docker Desktop handles
# bind-mount ownership in its own VM anyway. Use the image's built-in airflow
# user (50000) there instead.
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) HOST_UID=50000 ;;
  *)                        HOST_UID="$(id -u)" ;;
esac
echo "    AIRFLOW_UID=${HOST_UID}"

if [[ -f .env ]]; then
  echo "    .env already exists, leaving it alone (delete it to regenerate)"
else
  {
    echo "AIRFLOW_UID=${HOST_UID}"
    echo "AQ_LOCATION_NAME=Boston, MA"
    echo "AQ_LATITUDE=42.3601"
    echo "AQ_LONGITUDE=-71.0589"
    echo "AQ_HORIZON_HOURS=1"
    echo "# Leave AQ_ALERT_EMAIL empty to disable notification emails."
    echo "AQ_ALERT_EMAIL="
    echo "SMTP_USER="
    echo "SMTP_PASSWORD="
  } > .env
fi

echo "==> Starting Airflow"
docker compose up -d

cat <<'MSG'

Done. Next steps:

  Airflow UI    http://localhost:8080      (airflow / airflow)
  Dashboard     http://localhost:5555      (once the pipeline DAG has run)

  1. Unpause and trigger "Airflow_Lab2_AirQuality" in the UI.
  2. It finishes by triggering "Airflow_Lab2_AirQuality_Flask", which serves
     the dashboard. That DAG runs until you stop it -- expected, see README.

  Logs:  docker compose logs -f airflow-scheduler
  Stop:  docker compose down
MSG
