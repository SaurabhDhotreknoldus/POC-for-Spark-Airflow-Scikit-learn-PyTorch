"""Pytest configuration and shared test fixtures."""

import os
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def sample_taxi_dataframe():
    """Generates a small in-memory DataFrame matching NYC TLC schema."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "VendorID": np.random.choice([1, 2], size=n),
        "tpep_pickup_datetime": pd.date_range("2026-01-01 08:00:00", periods=n, freq="15min"),
        "tpep_dropoff_datetime": pd.date_range("2026-01-01 08:20:00", periods=n, freq="15min"),
        "passenger_count": np.random.choice([1, 2, 3, 4], size=n).astype("float64"),
        "trip_distance": np.random.uniform(0.5, 15.0, size=n),
        "RatecodeID": np.ones(n, dtype="float64"),
        "store_and_fwd_flag": ["N"] * n,
        "PULocationID": np.random.randint(1, 260, size=n).astype("int64"),
        "DOLocationID": np.random.randint(1, 260, size=n).astype("int64"),
        "payment_type": np.ones(n, dtype="int64"),
        "fare_amount": np.random.uniform(5.0, 60.0, size=n),
        "extra": np.zeros(n),
        "mta_tax": np.full(n, 0.5),
        "tip_amount": np.random.uniform(1.0, 10.0, size=n),
        "tolls_amount": np.zeros(n),
        "improvement_surcharge": np.ones(n),
        "total_amount": np.random.uniform(7.0, 75.0, size=n),
        "congestion_surcharge": np.full(n, 2.5),
        "Airport_fee": np.zeros(n),
    })
    return df


@pytest.fixture
def temp_workspace():
    """Creates a temporary workspace directory for test artifacts and tears it down."""
    temp_dir = tempfile.mkdtemp(prefix="test_ml_pipeline_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

