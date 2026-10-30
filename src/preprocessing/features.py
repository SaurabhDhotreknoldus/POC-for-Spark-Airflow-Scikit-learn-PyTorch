"""Feature engineering and preprocessing pipeline using scikit-learn.

Applies ColumnTransformer (StandardScaler for numerics, OneHotEncoder for categoricals),
prevents data leakage via strict Train-only fitting, and persists preprocessor.joblib and train_test_data.npz.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Add project root to path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import ensure_directory, get_logger, load_config


class TaxiFeaturePipeline:
    """End-to-end scikit-learn preprocessing pipeline for NYC Taxi Fare prediction."""

    def __init__(self, config_path: str = "config/development.yaml"):
        self.logger = get_logger("FeaturePipeline")
        self.config = load_config(config_path)
        self.data_cfg = self.config["data"]
        self.feature_cfg = self.config["features"]
        self.num_cols = self.feature_cfg["numeric"]
        self.cat_cols = self.feature_cfg["categorical"]
        self.target_col = self.feature_cfg["target"]

    def build_transformer(self) -> ColumnTransformer:
        """Constructs an industry-standard ColumnTransformer."""
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.num_cols),
                ("cat", categorical_pipeline, self.cat_cols),
            ],
            remainder="drop",
        )
        return transformer

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads cleaned parquet data, splits train/test, fits transformer, and serializes artifacts.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        start_time = time.time()
        input_file = self.data_cfg["processed_file"]
        model_dir = self.data_cfg.get("models_dir", "models")
        preprocessor_file = self.data_cfg.get("preprocessor_path", os.path.join(model_dir, "preprocessor.joblib"))
        output_npz = os.path.join("data", "training", "train_test_data.npz")

        ensure_directory(preprocessor_file)
        ensure_directory(output_npz)

        self.logger.info(f"Loading cleaned data from: {input_file}")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Cleaned dataset not found at {input_file}. Run spark_processor.py first.")

        df = pd.read_parquet(input_file)
        self.logger.info(f"Loaded {len(df):,} records for feature transformation.")

        # Validate columns
        all_required = self.num_cols + self.cat_cols + [self.target_col]
        missing = [c for c in all_required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in dataset: {missing}")

        X = df[self.num_cols + self.cat_cols]
        y = df[self.target_col].values.astype(np.float32)

        # 80/20 Train/Test Split
        test_size = self.feature_cfg.get("test_size", 0.2)
        seed = self.data_cfg.get("random_seed", 42)
        self.logger.info(f"Splitting dataset (test_size={test_size}, random_seed={seed})...")
        X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        self.logger.info(f"Train split size: {len(X_train_df):,} | Test split size: {len(X_test_df):,}")

        # Build & Fit Transformer on TRAIN ONLY (Strict Data Leakage Prevention)
        self.logger.info("Fitting ColumnTransformer on X_train...")
        transformer = self.build_transformer()
        X_train = transformer.fit_transform(X_train_df).astype(np.float32)

        self.logger.info("Transforming X_test using fitted preprocessor...")
        X_test = transformer.transform(X_test_df).astype(np.float32)

        feature_dim = X_train.shape[1]
        self.logger.info(f"Transformed Feature Matrix dimensions: {X_train.shape} (D = {feature_dim} features)")

        # Save fitted transformer
        self.logger.info(f"Saving preprocessor to: {preprocessor_file}")
        joblib.dump(transformer, preprocessor_file)

        # Save numpy data bundle for PyTorch DataLoader
        self.logger.info(f"Persisting processed numpy arrays to: {output_npz}")
        np.savez_compressed(
            output_npz,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        elapsed = time.time() - start_time
        self.logger.info(f"Feature engineering pipeline completed successfully in {elapsed:.2f}s!")
        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--config", type=str, default="config/development.yaml", help="Path to config YAML")
    args = parser.parse_args()

    pipeline = TaxiFeaturePipeline(config_path=args.config)
    pipeline.run()

