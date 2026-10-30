"""PyTorch training script for NYC Taxi Fare Prediction.

Loads engineered features from train_test_data.npz, instantiates FareMLP,
executes mini-batch gradient descent with Adam & MSE Loss, logs real-time metrics
and hyperparameters to MLflow, and saves trained model weights.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.preprocessing.features import TaxiFeaturePipeline
from src.tracking.mlflow_utils import MLflowTracker
from src.training.model import FareMLP
from src.utils import ensure_directory, get_logger, load_config


def train_model(config_path: str = "config/development.yaml"):
    """Trains FareMLP model and logs experiment details to MLflow."""
    logger = get_logger("Trainer")
    config = load_config(config_path)

    data_cfg = config["data"]
    train_cfg = config["training"]
    mlflow_cfg = config.get("mlflow", {})

    npz_path = os.path.join("data", "training", "train_test_data.npz")
    model_save_path = data_cfg.get("model_path", "models/fare_model.pt")
    ensure_directory(model_save_path)

    # 1. Load Preprocessed Data
    if not os.path.exists(npz_path):
        logger.info(f"{npz_path} not found. Running feature engineering pipeline first...")
        pipeline = TaxiFeaturePipeline(config_path)
        X_train, y_train, X_test, y_test = pipeline.run()
    else:
        logger.info(f"Loading cached features from {npz_path}...")
        data = np.load(npz_path)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]

    input_dim = X_train.shape[1]
    logger.info(f"Dataset loaded: X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    # 2. PyTorch Datasets & DataLoaders
    batch_size = train_cfg.get("batch_size", 128)
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model Architecture
    device_str = train_cfg.get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu")
    logger.info(f"Training device selected: {device}")

    model = FareMLP(
        input_dim=input_dim,
        hidden1=train_cfg.get("hidden1", 128),
        hidden2=train_cfg.get("hidden2", 64),
        dropout=train_cfg.get("dropout", 0.1),
    ).to(device)

    logger.info(f"Initialized FareMLP with {model.count_parameters():,} trainable parameters.")

    # 4. Optimization & Loss
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.00001)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5. MLflow Tracking Setup
    tracker = MLflowTracker(
        tracking_uri=mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db"),
        experiment_name=mlflow_cfg.get("experiment_name", "nyc_taxi_fare_prediction"),
    )

    epochs = train_cfg.get("epochs", 10)

    with tracker.start_run(run_name=f"FareMLP_epochs{epochs}_bs{batch_size}"):
        # Log Hyperparameters
        tracker.log_params(
            {
                "input_dim": input_dim,
                "hidden1": train_cfg.get("hidden1", 128),
                "hidden2": train_cfg.get("hidden2", 64),
                "dropout": train_cfg.get("dropout", 0.1),
                "batch_size": batch_size,
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "optimizer": "Adam",
                "loss_function": "MSELoss",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

        logger.info(f"Starting PyTorch training loop for {epochs} epochs...")
        start_time = time.time()
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            total_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_batches += 1

            train_loss = running_loss / max(1, total_batches)

            # Validation step
            model.eval()
            val_running_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for val_x, val_y in test_loader:
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                    val_preds = model(val_x)
                    val_loss = criterion(val_preds, val_y)
                    val_running_loss += val_loss.item()
                    val_batches += 1

            val_loss = val_running_loss / max(1, val_batches)
            epoch_time = time.time() - epoch_start

            # Log to MLflow
            tracker.log_metric("train_loss", train_loss, step=epoch)
            tracker.log_metric("val_loss", val_loss, step=epoch)

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] - Train MSE: {train_loss:.4f} | "
                f"Val MSE: {val_loss:.4f} (RMSE: ${np.sqrt(val_loss):.2f}) - {epoch_time:.2f}s"
            )

            # Track best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s! Best Validation MSE: {best_val_loss:.4f}")

        # 6. Save Model Checkpoint
        logger.info(f"Saving model weights to: {model_save_path}")
        torch.save(
            {
                "input_dim": input_dim,
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_val_loss": best_val_loss,
            },
            model_save_path,
        )

        # Log artifacts to MLflow
        tracker.log_artifact(model_save_path)
        preprocessor_path = data_cfg.get("preprocessor_path", "models/preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            tracker.log_artifact(preprocessor_path)

        # Log model with signature and example to ensure READY status in MLflow
        try:
            from mlflow.models.signature import infer_signature
            sample_t = torch.randn(1, input_dim)
            sample_np = sample_t.numpy()
            model.eval()
            with torch.no_grad():
                pred_np = model(sample_t).numpy()
            sig = infer_signature(sample_np, pred_np)
            tracker.log_pytorch_model(
                model,
                artifact_path="fare_model_registered",
                input_example=sample_np,
                signature=sig,
            )
        except Exception as e:
            logger.warning(f"Could not infer signature for model logging: {e}")
            tracker.log_pytorch_model(model, artifact_path="fare_model_registered")

    return model_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Training")
    parser.add_argument("--config", type=str, default="config/development.yaml", help="Path to config YAML")
    args = parser.parse_args()

    train_model(config_path=args.config)

