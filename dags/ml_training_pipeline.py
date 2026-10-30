"""Apache Airflow DAG: NYC Taxi Fare End-to-End ML Pipeline.

Orchestrates the complete lifecycle:
  1. Data Validation (check Parquet file existence and schemas)
  2. Spark Data Cleaning & Feature Derivation
  3. scikit-learn Feature Pipeline & Joblib Artifact Serialization
  4. PyTorch Neural Network Training with Adam & MSE Loss
  5. Evaluation & MLflow Metric/Plot Artifact Logging

Note: When using the dual-virtualenv architecture, Airflow tasks execute
under the ML virtualenv using the configured ML_PYTHON_BIN.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# Path to python in ml-env (configurable via environment variable)
ML_PYTHON = os.environ.get("ML_PYTHON_BIN", "python")
CONFIG_PATH = os.environ.get("ML_CONFIG_PATH", "config/development.yaml")

default_args = {
    "owner": "nashtech_ai_competency",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="nyc_taxi_fare_ml_pipeline",
    default_args=default_args,
    description="End-to-end Enterprise ML Pipeline: Spark + scikit-learn + PyTorch + MLflow",
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["nashtech", "spark", "pytorch", "mlflow", "poc"],
) as dag:

    # Task 1: Validate Raw Data Files
    validate_raw_data = BashOperator(
        task_id="validate_raw_data",
        bash_command=f"{ML_PYTHON} src/utils.py --validate-data --config {CONFIG_PATH}",
    )

    # Task 2: Distributed Spark Ingestion & Outlier Filtering
    run_spark_processing = BashOperator(
        task_id="run_spark_processing",
        bash_command=f"{ML_PYTHON} src/ingestion/spark_processor.py --config {CONFIG_PATH}",
    )

    # Task 3: Feature Engineering & Preprocessor Persistence
    run_feature_engineering = BashOperator(
        task_id="run_feature_engineering",
        bash_command=f"{ML_PYTHON} src/preprocessing/features.py --config {CONFIG_PATH}",
    )

    # Task 4: PyTorch Model Training & Real-time MLflow Logging
    train_pytorch_model = BashOperator(
        task_id="train_pytorch_model",
        bash_command=f"{ML_PYTHON} src/training/train.py --config {CONFIG_PATH}",
    )

    # Task 5: Model Evaluation & Diagnostic Visualizations
    evaluate_and_log_model = BashOperator(
        task_id="evaluate_and_log_model",
        bash_command=f"{ML_PYTHON} src/evaluation/evaluate.py --config {CONFIG_PATH}",
    )

    # Pipeline Execution Dependency Chain
    (
        validate_raw_data
        >> run_spark_processing
        >> run_feature_engineering
        >> train_pytorch_model
        >> evaluate_and_log_model
    )

