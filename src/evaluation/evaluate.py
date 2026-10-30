"""Evaluation and Diagnostic visualization for NYC Taxi Fare model.

Computes RMSE, MAE, R², generates actual-vs-predicted and residual plots,
and registers all evaluation metrics and plot artifacts to MLflow.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/pipeline execution
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.inference.predict import run_sample_traces
from src.tracking.mlflow_utils import MLflowTracker
from src.training.model import FareMLP
from src.utils import ensure_directory, get_logger, load_config


def evaluate_model(config_path: str = "config/development.yaml"):
    """Evaluates the trained FareMLP on test set and logs metrics/artifacts."""
    logger = get_logger("Evaluator")
    config = load_config(config_path)

    data_cfg = config["data"]
    mlflow_cfg = config.get("mlflow", {})
    npz_path = os.path.join("data", "training", "train_test_data.npz")
    model_path = data_cfg.get("model_path", "models/fare_model.pt")
    plots_dir = data_cfg.get("models_dir", "models")
    ensure_directory(plots_dir)

    # 1. Load Data
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Test features not found at {npz_path}.")

    data = np.load(npz_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    logger.info(f"Loaded test dataset: {len(X_test):,} samples.")

    # 2. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")

    logger.info(f"Loading model checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    input_dim = checkpoint["input_dim"]
    train_cfg = config.get("training", {})

    model = FareMLP(
        input_dim=input_dim,
        hidden1=train_cfg.get("hidden1", 128),
        hidden2=train_cfg.get("hidden2", 64),
        dropout=0.0,  # Evaluation mode disables dropout
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Model Inference
    logger.info("Running inference on holdout test set...")
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test)
        y_pred = model(x_tensor).squeeze(1).numpy()

    # 4. Compute Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    logger.info("=" * 50)
    logger.info("           MODEL PERFORMANCE METRICS")
    logger.info("=" * 50)
    logger.info(f"  Test RMSE : ${rmse:.2f}")
    logger.info(f"  Test MAE  : ${mae:.2f}")
    logger.info(f"  Test R²   : {r2:.4f} ({r2 * 100:.1f}% variance explained)")
    logger.info("=" * 50)

    # 5. Diagnostic Visualizations
    actual_pred_plot = os.path.join(plots_dir, "actual_vs_predicted.png")
    residuals_plot = os.path.join(plots_dir, "residuals_distribution.png")

    logger.info(f"Generating diagnostic plots in '{plots_dir}'...")

    # Plot 1: Actual vs Predicted Scatter
    plt.figure(figsize=(8, 6))
    sample_size = min(2000, len(y_test))
    idx = np.random.choice(len(y_test), sample_size, replace=False)
    plt.scatter(y_test[idx], y_pred[idx], alpha=0.35, color="#1f77b4", edgecolors="none")
    max_val = max(np.max(y_test[idx]), np.max(y_pred[idx]))
    plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect 1:1 Line")
    plt.title(f"NYC Taxi Fare: Actual vs Predicted (Sample of {sample_size:,})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Fare ($)", fontsize=11)
    plt.ylabel("Predicted Fare ($)", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(actual_pred_plot, dpi=150)
    plt.close()

    # Plot 2: Residuals Distribution
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, color="#2ca02c", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", lw=2, label="Zero Error")
    plt.title("Prediction Residuals Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    plt.xlabel("Residual ($)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(residuals_plot, dpi=150)
    plt.close()

    # 6. Log Evaluation to MLflow
    tracker = MLflowTracker(
        tracking_uri=mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"),
        experiment_name=mlflow_cfg.get("experiment_name", "nyc_taxi_fare_prediction"),
    )

    with tracker.start_run(run_name="model_evaluation") as active_run:
        tracker.log_metrics({"test_rmse": rmse, "test_mae": mae, "test_r2": r2})
        tracker.log_artifact(actual_pred_plot)
        tracker.log_artifact(residuals_plot)

        # 7. Generate and link diagnostic inference traces to the active evaluation run
        run_id = getattr(getattr(active_run, "info", None), "run_id", None)
        logger.info(f"Generating diagnostic inference traces for Run ID: {run_id}...")
        try:
            run_sample_traces(config_path=config_path, link_to_run_id=run_id)
            logger.info("Diagnostic inference traces successfully linked to evaluation run.")
        except Exception as e:
            logger.warning(f"Failed to generate traces during evaluation: {e}")

    logger.info("Evaluation complete! Metrics, artifacts, and inference traces logged successfully.")
    return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--config", type=str, default="config/development.yaml", help="Path to config YAML")
    args = parser.parse_args()

    evaluate_model(config_path=args.config)

