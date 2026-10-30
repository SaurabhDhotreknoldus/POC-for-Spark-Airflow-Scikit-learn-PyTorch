"""Local turnkey pipeline execution runner.

Sequentially executes:
  1. Data Pre-flight Check (Generates synthetic sample if raw data is missing)
  2. Spark Ingestion & Outlier Filtering
  3. scikit-learn Feature Pipeline
  4. PyTorch Neural Network Training
  5. Model Evaluation, Metrics Summary, and MLflow Artifact Registration

Usage:
  python scripts/run_pipeline_local.py --config config/development.yaml
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.ingestion.spark_processor import configure_java_home
from src.utils import get_logger, load_config, validate_raw_data


def run_pipeline(config_path: str = "config/development.yaml"):
    logger = get_logger("PipelineRunner")
    java_home = configure_java_home(logger)
    if java_home:
        logger.info(f"Targeting Java Runtime: {java_home}")
    logger.info("=" * 65)
    logger.info("  STARTING ENTERPRISE ML PIPELINE EXECUTION (LOCAL MODE)")
    logger.info(f"  Configuration: {config_path}")
    logger.info("=" * 65)

    start_total = time.time()
    config = load_config(config_path)

    # 1. Pre-flight Data Check
    logger.info("\n[STEP 1/5] Checking Raw Data Files...")
    is_valid = validate_raw_data(config_path)
    if not is_valid:
        logger.warning("Raw Parquet files missing! Invoking synthetic data generator...")
        gen_script = os.path.join("scripts", "generate_sample_data.py")
        raw_dir = config.get("data", {}).get("raw_dir", "Data files")
        cmd = [sys.executable, gen_script, "--output-dir", raw_dir, "--records-per-file", "25000", "--months", "3"]
        subprocess.check_call(cmd)
        logger.info("Synthetic dataset generated successfully!")

    # 2. PySpark Data Ingestion & Cleaning
    logger.info("\n[STEP 2/5] Running PySpark Ingestion & Cleaning...")
    spark_script = os.path.join("src", "ingestion", "spark_processor.py")
    cmd = [sys.executable, spark_script, "--config", config_path]
    subprocess.check_call(cmd)

    # 3. scikit-learn Feature Engineering
    logger.info("\n[STEP 3/5] Running scikit-learn Feature Pipeline...")
    feat_script = os.path.join("src", "preprocessing", "features.py")
    cmd = [sys.executable, feat_script, "--config", config_path]
    subprocess.check_call(cmd)

    # 4. PyTorch Model Training
    logger.info("\n[STEP 4/5] Training PyTorch FareMLP Neural Network...")
    train_script = os.path.join("src", "training", "train.py")
    cmd = [sys.executable, train_script, "--config", config_path]
    subprocess.check_call(cmd)

    # 5. Model Evaluation & Visualization
    logger.info("\n[STEP 5/5] Evaluating Model & Logging MLflow Artifacts...")
    eval_script = os.path.join("src", "evaluation", "evaluate.py")
    cmd = [sys.executable, eval_script, "--config", config_path]
    subprocess.check_call(cmd)

    total_time = time.time() - start_total
    logger.info("\n" + "=" * 65)
    logger.info(f"  ALL PIPELINE STEPS COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    logger.info("=" * 65)
    logger.info("Artifacts produced:")
    logger.info("  - Cleaned Parquet : data/processed/taxi_cleaned.parquet")
    logger.info("  - Preprocessor    : models/preprocessor.joblib")
    logger.info("  - Trained Model   : models/fare_model.pt")
    logger.info("  - Actual vs Pred  : models/actual_vs_predicted.png")
    logger.info("  - Residuals Plot  : models/residuals_distribution.png")
    logger.info("  - MLflow Database : sqlite:///mlflow.db (View with: mlflow ui --port 5000)")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local ML Pipeline Runner")
    parser.add_argument("--config", type=str, default="config/development.yaml", help="Path to configuration YAML")
    args = parser.parse_args()

    run_pipeline(config_path=args.config)

