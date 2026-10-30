"""MLflow experiment tracking and artifact logging utilities.

Provides safe wrappers for experiment management, hyperparameter logging,
metrics logging, and model artifact registration.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import get_logger

logger = get_logger("MLflowTracker")

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow package is not installed. Experiment tracking will run in mock mode.")


class MLflowTracker:
    """Helper to configure and interact with MLflow tracking server."""

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "nyc_taxi_fare_prediction"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.is_active = MLFLOW_AVAILABLE

        if self.is_active:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"MLflow initialized: URI='{self.tracking_uri}', Experiment='{self.experiment_name}'")
            except Exception as e:
                logger.warning(f"Could not connect to MLflow URI '{self.tracking_uri}': {e}. Falling back to local ./mlruns")
                try:
                    mlflow.set_tracking_uri("./mlruns")
                    mlflow.set_experiment(self.experiment_name)
                except Exception as ex:
                    logger.warning(f"MLflow initialization failed: {ex}. Running without remote tracking.")
                    self.is_active = False

    def start_run(self, run_name: Optional[str] = None):
        """Starts a new MLflow run if MLflow is active."""
        if self.is_active:
            return mlflow.start_run(run_name=run_name)
        return DummyContext()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs hyperparameters to MLflow."""
        if self.is_active:
            try:
                # Flatten complex values to strings/scalars
                flat_params = {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()}
                mlflow.log_params(flat_params)
            except Exception as e:
                logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Logs a single metric."""
        if self.is_active:
            try:
                mlflow.log_metric(key, float(value), step=step)
            except Exception as e:
                logger.warning(f"Failed to log metric '{key}': {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs multiple metrics."""
        if self.is_active:
            try:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")

    def log_artifact(self, local_path: str) -> None:
        """Logs a file artifact (e.g. plot or joblib preprocessor)."""
        if self.is_active and Path(local_path).exists():
            try:
                mlflow.log_artifact(local_path)
                logger.info(f"Logged artifact to MLflow: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to log artifact '{local_path}': {e}")

    def log_pytorch_model(self, model, artifact_path: str = "pytorch_model", input_example=None, signature=None) -> None:
        """Logs a PyTorch model into MLflow model registry."""
        if self.is_active:
            try:
                kwargs = {}
                if input_example is not None:
                    kwargs["input_example"] = input_example
                if signature is not None:
                    kwargs["signature"] = signature
                mlflow.pytorch.log_model(model, artifact_path=artifact_path, **kwargs)
                logger.info(f"Registered PyTorch model artifact under '{artifact_path}'")
            except Exception as e:
                logger.warning(f"Failed to log PyTorch model to MLflow: {e}")


class DummyContext:
    """No-op context manager when MLflow is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

