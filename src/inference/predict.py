"""Inference and Observability engine with MLflow Tracing.

Provides real-time single and batch fare predictions with end-to-end
multi-span tracing (Preprocessing -> PyTorch Neural Network -> TLC Business Guardrails).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

# Add project root to path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.training.model import FareMLP
from src.utils import get_logger, load_config

logger = get_logger("InferenceEngine")

try:
    import mlflow
    from mlflow.entities import SpanType
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class TaxiFarePredictor:
    """Production predictor with preprocessor, PyTorch model, and MLflow Tracing."""

    def __init__(self, config_path: str = "config/development.yaml", model_id: Optional[str] = None):
        self.config = load_config(config_path)
        self.data_cfg = self.config["data"]
        self.mlflow_cfg = self.config.get("mlflow", {})
        self.tracking_uri = self.mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
        self.experiment_name = self.mlflow_cfg.get("experiment_name", "nyc_taxi_fare_prediction")

        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

        model_dir = self.data_cfg.get("models_dir", "models")
        self.preprocessor_path = self.data_cfg.get("preprocessor_path", os.path.join(model_dir, "preprocessor.joblib"))
        self.model_path = self.data_cfg.get("model_path", os.path.join(model_dir, "fare_model.pt"))

        # Resolve model_id for MLflow Model Traces association
        self.model_id = model_id
        if self.model_id is None and MLFLOW_AVAILABLE:
            try:
                client = MlflowClient()
                logged_models = client.search_logged_models(experiment_ids=[1, "1"])
                if logged_models:
                    self.model_id = logged_models[0].model_id
                    logger.info(f"Associated predictor with logged model ID: {self.model_id}")
            except Exception as e:
                logger.debug(f"Could not resolve model_id: {e}")

        # Load artifacts
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at '{self.preprocessor_path}'. Run features.py first.")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at '{self.model_path}'. Run train.py first.")

        logger.info(f"Loading preprocessor from {self.preprocessor_path}...")
        self.preprocessor = joblib.load(self.preprocessor_path)

        logger.info(f"Loading PyTorch checkpoint from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"), weights_only=False)
        train_cfg = self.config.get("training", {})

        self.model = FareMLP(
            input_dim=checkpoint["input_dim"],
            hidden1=train_cfg.get("hidden_dim1", 128),
            hidden2=train_cfg.get("hidden_dim2", 64),
            dropout=0.0,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("TaxiFarePredictor successfully initialized and ready for inference.")

    def predict(
        self,
        trip_distance: float,
        passenger_count: int = 1,
        pickup_hour: int = 12,
        pickup_day_of_week: int = 2,
        PULocationID: int = 161,
        DOLocationID: int = 236,
        RatecodeID: int = 1,
    ) -> Dict[str, Any]:
        """Runs end-to-end inference with MLflow multi-span tracing."""
        if MLFLOW_AVAILABLE:
            return self._traced_predict(
                trip_distance=trip_distance,
                passenger_count=passenger_count,
                pickup_hour=pickup_hour,
                pickup_day_of_week=pickup_day_of_week,
                PULocationID=PULocationID,
                DOLocationID=DOLocationID,
                RatecodeID=RatecodeID,
            )
        return self._untraced_predict(
            trip_distance, passenger_count, pickup_hour, pickup_day_of_week, PULocationID, DOLocationID, RatecodeID
        )

    def _untraced_predict(
        self, trip_distance, passenger_count, pickup_hour, pickup_day_of_week, PULocationID, DOLocationID, RatecodeID
    ):
        raw_df = pd.DataFrame([{
            "trip_distance": float(trip_distance),
            "passenger_count": int(passenger_count),
            "pickup_hour": int(pickup_hour),
            "pickup_day_of_week": int(pickup_day_of_week),
            "PULocationID": int(PULocationID),
            "DOLocationID": int(DOLocationID),
            "RatecodeID": int(RatecodeID),
        }])
        X_trans = self.preprocessor.transform(raw_df).astype(np.float32)
        with torch.no_grad():
            raw_pred = self.model(torch.from_numpy(X_trans)).item()
        final_fare = round(max(2.50, float(raw_pred)), 2)
        return {
            "trip_distance_miles": trip_distance,
            "predicted_fare_usd": final_fare,
            "raw_model_output": round(float(raw_pred), 2),
            "validation_passed": True,
        }

    def _traced_predict(
        self,
        trip_distance: float,
        passenger_count: int,
        pickup_hour: int,
        pickup_day_of_week: int,
        PULocationID: int,
        DOLocationID: int,
        RatecodeID: int,
    ) -> Dict[str, Any]:
        """Executes inference while recording fine-grained MLflow spans."""

        @mlflow.trace(name="NYC_Taxi_Fare_Prediction_Pipeline", span_type=SpanType.WORKFLOW)
        def _execute_pipeline(dist, pass_cnt, hour, dow, pu, do, rate_code):
            t0 = time.perf_counter()

            # Span 1: Feature Preprocessing
            with mlflow.start_span(name="Feature_Preprocessing_ColumnTransformer", span_type=SpanType.TOOL) as span:
                raw_inputs = {
                    "trip_distance": dist,
                    "passenger_count": pass_cnt,
                    "pickup_hour": hour,
                    "pickup_day_of_week": dow,
                    "PULocationID": pu,
                    "DOLocationID": do,
                    "RatecodeID": rate_code,
                }
                span.set_inputs(raw_inputs)
                raw_df = pd.DataFrame([raw_inputs])
                X_trans = self.preprocessor.transform(raw_df).astype(np.float32)
                span.set_outputs({
                    "features_dimension": int(X_trans.shape[1]),
                    "encoded_sparse_or_dense": "dense_float32",
                    "status": "features_ready",
                })

            # Span 2: PyTorch Model Inference
            with mlflow.start_span(name="PyTorch_FareMLP_Inference", span_type=SpanType.TASK) as span:
                span.set_inputs({
                    "tensor_shape": list(X_trans.shape),
                    "device": "cpu",
                    "architecture": "FareMLP(128->64->1)",
                })
                infer_t0 = time.perf_counter()
                with torch.no_grad():
                    raw_fare = self.model(torch.from_numpy(X_trans)).item()
                infer_latency_ms = round((time.perf_counter() - infer_t0) * 1000, 2)
                span.set_outputs({
                    "raw_fare_prediction_usd": round(float(raw_fare), 2),
                    "inference_latency_ms": infer_latency_ms,
                    "device_used": "cpu",
                })

            # Span 3: TLC Business Rules & Validation
            with mlflow.start_span(name="TLC_Business_Validation", span_type=SpanType.EVALUATOR) as span:
                span.set_inputs({"raw_predicted_fare": round(float(raw_fare), 2)})
                tlc_min_base_fare = 2.50
                tlc_max_fare_cap = 500.00
                enforced_fare = max(tlc_min_base_fare, min(tlc_max_fare_cap, float(raw_fare)))
                enforced_fare = round(enforced_fare, 2)
                is_valid = tlc_min_base_fare <= enforced_fare <= tlc_max_fare_cap
                span.set_outputs({
                    "enforced_fare_usd": enforced_fare,
                    "tlc_minimum_applied": float(raw_fare) < tlc_min_base_fare,
                    "is_compliant": is_valid,
                })

            # Update trace with model association, tags, and request metadata
            if self.model_id:
                try:
                    mlflow.update_current_trace(
                        model_id=self.model_id,
                        tags={
                            "model_name": "fare_model_registered",
                            "model_type": "PyTorch_FareMLP",
                            "framework": "PyTorch",
                            "pipeline": "NYC_Taxi_Fare_Prediction",
                            "environment": "Production_Serving",
                        },
                    )
                except Exception as ex:
                    logger.debug(f"Could not update trace with model_id: {ex}")

            total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "trip_distance_miles": dist,
                "passenger_count": pass_cnt,
                "pickup_hour": hour,
                "pickup_day_of_week": dow,
                "PULocationID": pu,
                "DOLocationID": do,
                "predicted_fare_usd": enforced_fare,
                "total_pipeline_latency_ms": total_latency_ms,
                "validation_passed": is_valid,
            }

        result = _execute_pipeline(
            trip_distance, passenger_count, pickup_hour, pickup_day_of_week, PULocationID, DOLocationID, RatecodeID
        )

        # Set rich tags and metadata on the active trace
        try:
            trace_id = mlflow.get_last_active_trace_id()
            if trace_id:
                client = MlflowClient()
                client.set_trace_tag(trace_id, "model_name", "fare_model_registered")
                client.set_trace_tag(trace_id, "model_type", "PyTorch_FareMLP")
                client.set_trace_tag(trace_id, "framework", "PyTorch")
                client.set_trace_tag(trace_id, "pipeline", "NYC_Taxi_Fare_Prediction")
                client.set_trace_tag(trace_id, "environment", "Production_Serving")
        except Exception as e:
            logger.debug(f"Could not set trace tag: {e}")

        return result


def run_sample_traces(
    config_path: str = "config/development.yaml",
    link_to_run_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Runs a suite of real-world NYC taxi ride scenarios to generate diagnostic traces."""
    predictor = TaxiFarePredictor(config_path=config_path, model_id=model_id)

    scenarios = [
        {
            "name": "JFK Airport to Midtown Manhattan (Rush Hour)",
            "trip_distance": 16.5,
            "passenger_count": 2,
            "pickup_hour": 18,
            "pickup_day_of_week": 4,
            "PULocationID": 132,  # JFK Airport
            "DOLocationID": 161,  # Midtown Center
            "RatecodeID": 2,      # JFK Flat rate
        },
        {
            "name": "Wall Street to Greenwich Village (Downtown Short Hop)",
            "trip_distance": 2.2,
            "passenger_count": 1,
            "pickup_hour": 14,
            "pickup_day_of_week": 2,
            "PULocationID": 209,  # Seaport
            "DOLocationID": 113,  # Greenwich Village North
            "RatecodeID": 1,      # Standard rate
        },
        {
            "name": "Upper East Side to LaGuardia Airport (Morning Commute)",
            "trip_distance": 8.7,
            "passenger_count": 1,
            "pickup_hour": 8,
            "pickup_day_of_week": 1,
            "PULocationID": 236,  # Upper East Side North
            "DOLocationID": 138,  # LaGuardia Airport
            "RatecodeID": 1,      # Standard rate
        },
        {
            "name": "Downtown Brooklyn to Lower Manhattan (Bridge Crossing)",
            "trip_distance": 3.8,
            "passenger_count": 3,
            "pickup_hour": 20,
            "pickup_day_of_week": 5,
            "PULocationID": 65,   # Downtown Brooklyn
            "DOLocationID": 87,   # Financial District North
            "RatecodeID": 1,      # Standard rate
        },
        {
            "name": "Times Square to Columbia University (Night Leisure)",
            "trip_distance": 5.4,
            "passenger_count": 2,
            "pickup_hour": 23,
            "pickup_day_of_week": 6,
            "PULocationID": 230,  # Times Square
            "DOLocationID": 41,   # Central Harlem / Morningside
            "RatecodeID": 1,      # Standard rate
        },
    ]

    results = []
    generated_trace_ids = []

    print("\n" + "=" * 70)
    print("   EXECUTING REAL-TIME TAXI FARE INFERENCE WITH MLFLOW TRACING")
    print("=" * 70)

    for i, s in enumerate(scenarios, 1):
        scenario_name = s.pop("name")
        res = predictor.predict(**s)
        results.append(res)
        trace_id = mlflow.get_last_active_trace_id() if MLFLOW_AVAILABLE else None
        if trace_id:
            generated_trace_ids.append(trace_id)

        print(f"[{i}/5] {scenario_name}")
        print(f"      Distance: {s['trip_distance']} mi | Pickup: {s['pickup_hour']}:00 | PU Zone: {s['PULocationID']} -> DO Zone: {s['DOLocationID']}")
        print(f"      -> Predicted Fare: ${res['predicted_fare_usd']:.2f} | Latency: {res.get('total_pipeline_latency_ms', 0):.1f}ms | Trace ID: {trace_id}")

    if MLFLOW_AVAILABLE:
        mlflow.flush_trace_async_logging()

        # Link generated traces to a target run if specified
        if link_to_run_id and generated_trace_ids:
            try:
                client = MlflowClient()
                client.link_traces_to_run(trace_ids=generated_trace_ids, run_id=link_to_run_id)
                print(f"\nSuccessfully linked {len(generated_trace_ids)} traces to Run ID: {link_to_run_id}")
            except Exception as e:
                logger.warning(f"Failed to link traces to run {link_to_run_id}: {e}")

    print("=" * 70)
    print("All inference traces logged and flushed to MLflow!")
    print("Open MLflow UI -> Traces tab at: http://localhost:5000/#/experiments/1/traces")
    print("=" * 70 + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Fare Model Inference & Tracing")
    parser.add_argument("--config", type=str, default="config/development.yaml", help="Path to config YAML")
    parser.add_argument("--samples", action="store_true", help="Run 5 real-world scenarios to populate traces")
    parser.add_argument("--link-run-id", type=str, default=None, help="MLflow Run ID to associate traces with")
    parser.add_argument("--model-id", type=str, default=None, help="MLflow Logged Model ID (m-...) to associate traces with")
    parser.add_argument("--distance", type=float, default=3.5, help="Trip distance in miles")
    parser.add_argument("--passengers", type=int, default=1, help="Passenger count")
    parser.add_argument("--hour", type=int, default=18, help="Pickup hour (0-23)")
    parser.add_argument("--day", type=int, default=4, help="Day of week (0=Mon..6=Sun)")
    parser.add_argument("--pu", type=int, default=161, help="Pickup Location ID")
    parser.add_argument("--do", type=int, default=236, help="Dropoff Location ID")
    parser.add_argument("--ratecode", type=int, default=1, help="Rate code ID")
    args = parser.parse_args()

    if args.samples:
        run_sample_traces(config_path=args.config, link_to_run_id=args.link_run_id, model_id=args.model_id)
    else:
        predictor = TaxiFarePredictor(config_path=args.config, model_id=args.model_id)
        res = predictor.predict(
            trip_distance=args.distance,
            passenger_count=args.passengers,
            pickup_hour=args.hour,
            pickup_day_of_week=args.day,
            PULocationID=args.pu,
            DOLocationID=args.do,
            RatecodeID=args.ratecode,
        )
        if MLFLOW_AVAILABLE:
            mlflow.flush_trace_async_logging()
            trace_id = mlflow.get_last_active_trace_id()
            if args.link_run_id and trace_id:
                client = MlflowClient()
                client.link_traces_to_run(trace_ids=[trace_id], run_id=args.link_run_id)
        print("\nPrediction Result:")
        for k, v in res.items():
            print(f"  {k}: {v}")

