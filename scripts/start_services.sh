#!/usr/bin/env bash
# Shell script to start MLflow server and Airflow standalone concurrently (Linux/WSL2)
set -e

echo "=========================================================="
echo " Starting MLflow Tracking Server (Port 5000)..."
echo "=========================================================="
source ~/ml-env/bin/activate || true
nohup mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 > logs/mlflow.log 2>&1 &
echo "[+] MLflow running at http://localhost:5000 (PID: $!)"

echo "=========================================================="
echo " Starting Apache Airflow Standalone (Port 8080)..."
echo "=========================================================="
source ~/airflow-env/bin/activate || true
export AIRFLOW_HOME=~/airflow
mkdir -p logs
nohup airflow standalone > logs/airflow_standalone.log 2>&1 &
echo "[+] Airflow running at http://localhost:8080 (PID: $!)"

echo "All services initiated. Use 'tail -f logs/*.log' to monitor."

