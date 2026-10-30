"""Unit tests for scikit-learn feature transformation pipeline."""

import os
import joblib
import numpy as np
import pandas as pd
import pytest
from src.preprocessing.features import TaxiFeaturePipeline


def test_build_transformer():
    """Tests construction of the ColumnTransformer with numeric and categorical branches."""
    pipeline = TaxiFeaturePipeline("config/development.yaml")
    transformer = pipeline.build_transformer()
    assert transformer is not None

    transformers_dict = dict(transformer.transformers)
    assert "num" in transformers_dict
    assert "cat" in transformers_dict


def test_transformer_fit_transform(sample_taxi_dataframe, temp_workspace):
    """Verifies that the transformer fits on train and transforms test cleanly."""
    pipeline = TaxiFeaturePipeline("config/development.yaml")
    transformer = pipeline.build_transformer()

    df = sample_taxi_dataframe.copy()
    # Add derived temporal features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek + 1

    train_df = df.iloc[:150]
    test_df = df.iloc[150:]

    X_train = transformer.fit_transform(train_df[pipeline.num_cols + pipeline.cat_cols])
    X_test = transformer.transform(test_df[pipeline.num_cols + pipeline.cat_cols])

    assert X_train.shape[0] == 150
    assert X_test.shape[0] == 50
    assert X_train.shape[1] == X_test.shape[1]
    assert not np.isnan(X_train).any(), "Transformed train array must not contain NaNs"
    assert not np.isnan(X_test).any(), "Transformed test array must not contain NaNs"


def test_transformer_serialization(sample_taxi_dataframe, temp_workspace):
    """Verifies that fitted ColumnTransformer can be saved with joblib and reloaded for inference."""
    pipeline = TaxiFeaturePipeline("config/development.yaml")
    transformer = pipeline.build_transformer()

    df = sample_taxi_dataframe.copy()
    df["pickup_hour"] = 14
    df["pickup_day_of_week"] = 3

    transformer.fit(df[pipeline.num_cols + pipeline.cat_cols])

    joblib_path = os.path.join(temp_workspace, "test_preprocessor.joblib")
    joblib.dump(transformer, joblib_path)
    assert os.path.exists(joblib_path)

    loaded_transformer = joblib.load(joblib_path)
    out1 = transformer.transform(df[pipeline.num_cols + pipeline.cat_cols])
    out2 = loaded_transformer.transform(df[pipeline.num_cols + pipeline.cat_cols])
    np.testing.assert_array_almost_equal(out1, out2)

