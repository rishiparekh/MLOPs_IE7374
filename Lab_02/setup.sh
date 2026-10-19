#!/usr/bin/env bash
# Starts the Airflow stack. Run from Git Bash or WSL on Windows.
# Pass --clean to drop volumes and start over.
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
# Git Bash reports a Windows id that means nothing to a Linux container, and
# Docker Desktop handles bind-mount ownership itself, so use the image's own
# airflow user there.
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
    echo "AQ_ALERT_EMAIL="
    echo "SMTP_USER="
    echo "SMTP_PASSWORD="
  } > .env
fi

echo "==> Starting Airflow"
docker compose up -d

cat <<'MSG'

Done.

  Airflow UI    http://localhost:8080      (airflow / airflow)
  Dashboard     http://localhost:5555      (once the pipeline DAG has run)

  1. Unpause and trigger "Airflow_Lab2_AirQuality" in the UI.
  2. It finishes by triggering "Airflow_Lab2_AirQuality_Flask", which serves
     the dashboard and keeps running until you stop it.

  Logs:  docker compose logs -f airflow-scheduler
  Stop:  docker compose down
MSG
