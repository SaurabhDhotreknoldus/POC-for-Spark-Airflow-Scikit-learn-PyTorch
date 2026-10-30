"""Unit tests for PySpark data ingestion, validation, and outlier filtering."""

import os
import pytest
from src.utils import load_config

# Check if pyspark is available in current test environment
try:
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, dayofweek, hour
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="pyspark is not installed in current environment")
def test_spark_session_creation():
    """Tests that a local SparkSession can be instantiated and stopped cleanly."""
    spark = (
        SparkSession.builder.appName("PyTestSpark")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    assert spark is not None
    assert spark.version.startswith("3.")
    spark.stop()


@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="pyspark is not installed in current environment")
def test_spark_outlier_filtering(sample_taxi_dataframe, temp_workspace):
    """Verifies that Spark correctly discards negative fares and extreme distances."""
    spark = (
        SparkSession.builder.appName("PyTestSparkFilter")
        .master("local[1]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    try:
        # Introduce known outliers into pandas df
        df = sample_taxi_dataframe.copy()
        df.loc[0, "fare_amount"] = -10.0  # Should be dropped
        df.loc[1, "trip_distance"] = 0.0  # Should be dropped
        df.loc[2, "fare_amount"] = 9999.0 # Should be dropped
        df.loc[3, "passenger_count"] = 0.0 # Should be dropped

        spark_df = spark.createDataFrame(df)
        raw_count = spark_df.count()

        config = load_config("config/development.yaml")
        filters = config["filters"]

        filtered = spark_df.filter(
            (col("fare_amount") >= filters["min_fare"])
            & (col("fare_amount") <= filters["max_fare"])
            & (col("trip_distance") >= filters["min_distance"])
            & (col("trip_distance") <= filters["max_distance"])
            & (col("passenger_count") >= filters["min_passengers"])
            & (col("passenger_count") <= filters["max_passengers"])
        )

        clean_count = filtered.count()
        assert clean_count < raw_count
        assert clean_count == raw_count - 4

        # Verify no negative fares remain
        min_fare = filtered.select("fare_amount").rdd.min()[0]
        assert min_fare >= filters["min_fare"]
    finally:
        spark.stop()

