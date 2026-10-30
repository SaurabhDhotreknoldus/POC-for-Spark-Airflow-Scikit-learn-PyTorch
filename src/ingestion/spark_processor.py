"""PySpark-based data ingestion and cleaning for NYC Yellow Taxi trip records.

Performs distributed scanning, schema validation, outlier filtering, temporal feature derivation,
and writes cleaned staging Parquet dataset for downstream scikit-learn & PyTorch consumers.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import ensure_directory, get_logger, load_config


def configure_java_home(logger=None) -> str:
    """Auto-detects and configures Java 17/11 for PySpark compatibility on Windows.

    Spark 3.5.x requires Java 8, 11, or 17. Newer versions (Java 23+)
    removed Subject.getSubject, causing UnsupportedOperationException in Hadoop/Spark.
    """
    import glob

    current_home = os.environ.get("JAVA_HOME", "")
    # If already set to a compatible version, keep it
    if current_home and any(v in current_home.lower() for v in ["jdk-17", "jdk-11", "jdk-8", "jdk1.8"]):
        return current_home

    candidate_patterns = [
        r"C:\Program Files\Eclipse Adoptium\jdk-17*",
        r"C:\Program Files\Microsoft\jdk-17*",
        r"C:\Program Files\Java\jdk-17*",
        r"C:\Program Files\Eclipse Adoptium\jdk-11*",
        r"C:\Program Files\Microsoft\jdk-11*",
        r"C:\Program Files\Java\jdk-11*",
        r"C:\Program Files\Eclipse Adoptium\jdk-8*",
    ]
    for pattern in candidate_patterns:
        matches = glob.glob(pattern)
        for m in sorted(matches, reverse=True):
            java_exe = os.path.join(m, "bin", "java.exe")
            if os.path.exists(java_exe):
                os.environ["JAVA_HOME"] = m
                bin_dir = os.path.join(m, "bin")
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                if logger:
                    logger.info(f"Auto-configured PySpark compatible JAVA_HOME: '{m}'")
                return m

    return current_home


def is_windows_hadoop_ready() -> bool:
    """Checks if native Windows Hadoop binaries (winutils.exe and hadoop.dll) are available."""
    if os.name != "nt":
        return True  # Linux / macOS / WSL2 use glibc and libhadoop.so natively

    # Check HADOOP_HOME or standard C:\hadoop
    hadoop_home = os.environ.get("HADOOP_HOME", r"C:\hadoop")
    winutils = os.path.join(hadoop_home, "bin", "winutils.exe")
    hadoop_dll = os.path.join(hadoop_home, "bin", "hadoop.dll")
    if os.path.exists(winutils) and os.path.exists(hadoop_dll):
        os.environ["HADOOP_HOME"] = hadoop_home
        bin_dir = os.path.join(hadoop_home, "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        return True

    import shutil
    if shutil.which("winutils.exe"):
        return True

    return False


def get_spark_session(app_name: str = "NYC_Taxi_Cleaning", master: str = "local[*]", memory: str = "2g"):
    """Initializes and returns an optimized local SparkSession."""
    logger = get_logger("SparkSessionBuilder")
    configured_java = configure_java_home(logger)
    if configured_java:
        logger.info(f"Using JAVA_HOME: {configured_java}")

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError(
            "pyspark is not installed. Please install it via 'pip install pyspark' or activate your ml-env."
        )

    # Standard Java 17+ reflective access flags for Apache Spark
    java_options = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "-Djdk.reflect.useDirectMethodHandle=false"
    )

    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", memory)
        .config("spark.driver.extraJavaOptions", java_options)
        .config("spark.executor.extraJavaOptions", java_options)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


class SparkDataProcessor:
    """Handles scalable ingestion, outlier removal, and feature derivation using Apache Spark."""

    def __init__(self, config_path: str = "config/development.yaml"):
        self.logger = get_logger("SparkProcessor")
        self.config = load_config(config_path)
        self.data_cfg = self.config["data"]
        self.filter_cfg = self.config["filters"]
        self.spark_cfg = self.config.get("spark", {})

    def _process_with_pyarrow(self, input_files: list, output_path: str, start_time: float) -> str:
        """Fallback processing engine using PyArrow/Pandas for Windows environments
        where native Hadoop binaries (hadoop.dll/winutils.exe) are not installed.
        """
        import pandas as pd
        import pyarrow.parquet as pq

        self.logger.warning(
            "Native Hadoop Windows library (hadoop.dll / winutils.exe) is unavailable.\n"
            "   -> Using high-performance embedded PyArrow engine to complete ingestion seamlessly on Windows..."
        )

        sample_target = self.data_cfg.get("sample_rows")
        required_cols = [
            "tpep_pickup_datetime",
            "passenger_count",
            "trip_distance",
            "RatecodeID",
            "PULocationID",
            "DOLocationID",
            "fare_amount",
        ]

        dfs = []
        raw_count = 0
        self.logger.info(f"Scanning {len(input_files)} Parquet file(s) with PyArrow...")
        for f in input_files:
            table = pq.read_table(f, columns=required_cols)
            df_part = table.to_pandas()
            raw_count += len(df_part)
            dfs.append(df_part)
            if sample_target and raw_count >= sample_target * 4:
                break

        df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Loaded raw records: {len(df):,}")

        # Drop NULLs
        df = df.dropna(subset=required_cols)

        # Derive Temporal Features
        dt = pd.to_datetime(df["tpep_pickup_datetime"])
        df["pickup_hour"] = dt.dt.hour.astype(int)
        df["pickup_day_of_week"] = (dt.dt.dayofweek + 1).astype(int)

        # Outlier & Business Rules Filtering
        min_fare = self.filter_cfg.get("min_fare", 2.5)
        max_fare = self.filter_cfg.get("max_fare", 250.0)
        min_dist = self.filter_cfg.get("min_distance", 0.1)
        max_dist = self.filter_cfg.get("max_distance", 60.0)
        min_pass = self.filter_cfg.get("min_passengers", 1)
        max_pass = self.filter_cfg.get("max_passengers", 6)

        self.logger.info(
            f"Applying filters: fare [${min_fare}-${max_fare}], distance [{min_dist}-{max_dist} mi], "
            f"passengers [{min_pass}-{max_pass}]..."
        )

        filtered_df = df[
            (df["fare_amount"] >= min_fare)
            & (df["fare_amount"] <= max_fare)
            & (df["trip_distance"] >= min_dist)
            & (df["trip_distance"] <= max_dist)
            & (df["passenger_count"] >= min_pass)
            & (df["passenger_count"] <= max_pass)
            & (df["PULocationID"] > 0)
            & (df["DOLocationID"] > 0)
        ].copy()

        clean_count = len(filtered_df)
        self.logger.info(f"Records after quality filters: {clean_count:,} (dropped {len(df) - clean_count:,})")

        # Controlled sampling
        if sample_target and clean_count > sample_target:
            seed = self.data_cfg.get("random_seed", 42)
            self.logger.info(f"Sampling down to ~{sample_target:,} rows...")
            filtered_df = filtered_df.sample(n=sample_target, random_state=seed)

        output_df = filtered_df.drop(columns=["tpep_pickup_datetime"])
        output_df.to_parquet(output_path, engine="pyarrow", index=False)

        elapsed = time.time() - start_time
        self.logger.info(
            f"Data processing complete in {elapsed:.2f}s! Final cleaned records written: {len(output_df):,}"
        )
        return output_path

    def process(self) -> str:
        """Executes the full Spark cleaning pipeline with automatic Windows fallback.

        Returns:
            Path to the output cleaned Parquet file/directory.
        """
        start_time = time.time()
        raw_dir = self.data_cfg.get("raw_dir", "Data_Files")
        raw_pattern = self.data_cfg.get("raw_file_pattern", "*.parquet")

        import glob
        candidate_dirs = [raw_dir, "Data_Files", "Data files", os.path.join("data", "raw")]
        active_dir = raw_dir
        for candidate in candidate_dirs:
            if os.path.exists(candidate) and glob.glob(os.path.join(candidate, raw_pattern)):
                active_dir = candidate
                break

        matched_files = glob.glob(os.path.join(active_dir, raw_pattern))
        if not matched_files:
            raise FileNotFoundError(f"No files matching '{raw_pattern}' in '{active_dir}'.")

        input_path = os.path.join(active_dir, raw_pattern)
        output_path = self.data_cfg["processed_file"]
        ensure_directory(output_path)

        # Check if running on Windows without native Hadoop binaries
        if not is_windows_hadoop_ready():
            self.logger.warning(
                "Windows host detected without native Hadoop binaries (winutils.exe / hadoop.dll).\n"
                "   -> Bypassing Windows JVM NativeIO deadlock.\n"
                "   -> Executing ingestion via high-performance embedded PyArrow engine..."
            )
            return self._process_with_pyarrow(matched_files, output_path, start_time)

        # Attempt PySpark execution first
        spark = None
        try:
            self.logger.info(f"Initializing PySpark session (master={self.spark_cfg.get('master', 'local[*]')})...")
            spark = get_spark_session(
                app_name=self.spark_cfg.get("app_name", "NYCTaxi_Ingestion"),
                master=self.spark_cfg.get("master", "local[*]"),
                memory=self.spark_cfg.get("driver_memory", "2g"),
            )

            from pyspark.sql.functions import col, dayofweek, hour

            self.logger.info(f"Reading raw Parquet data from: {input_path}")
            df = spark.read.parquet(input_path)
            raw_count = df.count()
            self.logger.info(f"Loaded raw records: {raw_count:,}")

            if raw_count == 0:
                raise ValueError(f"No records found in raw input at {input_path}")

            # 1. Column Selection & Validation
            required_cols = [
                "tpep_pickup_datetime",
                "passenger_count",
                "trip_distance",
                "RatecodeID",
                "PULocationID",
                "DOLocationID",
                "fare_amount",
            ]
            for c in required_cols:
                if c not in df.columns:
                    raise KeyError(f"Required column '{c}' is missing from raw dataset.")

            df = df.select(required_cols)

            # 2. Drop records with NULL critical values
            df = df.dropna(subset=required_cols)

            # 3. Derive Temporal Features
            self.logger.info("Deriving temporal features: pickup_hour (0-23) and pickup_day_of_week (1-7)...")
            df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
            df = df.withColumn("pickup_day_of_week", dayofweek(col("tpep_pickup_datetime")))

            # 4. Outlier & Business Rules Filtering
            min_fare = self.filter_cfg.get("min_fare", 2.5)
            max_fare = self.filter_cfg.get("max_fare", 250.0)
            min_dist = self.filter_cfg.get("min_distance", 0.1)
            max_dist = self.filter_cfg.get("max_distance", 60.0)
            min_pass = self.filter_cfg.get("min_passengers", 1)
            max_pass = self.filter_cfg.get("max_passengers", 6)

            self.logger.info(
                f"Applying filters: fare [${min_fare}-${max_fare}], distance [{min_dist}-{max_dist} mi], "
                f"passengers [{min_pass}-{max_pass}]..."
            )

            filtered_df = df.filter(
                (col("fare_amount") >= min_fare)
                & (col("fare_amount") <= max_fare)
                & (col("trip_distance") >= min_dist)
                & (col("trip_distance") <= max_dist)
                & (col("passenger_count") >= min_pass)
                & (col("passenger_count") <= max_pass)
                & (col("PULocationID") > 0)
                & (col("DOLocationID") > 0)
            )

            # 5. Controlled sampling if requested in config (Enterprise Best Practice)
            sample_target = self.data_cfg.get("sample_rows")
            clean_count = filtered_df.count()
            self.logger.info(f"Records after quality filters: {clean_count:,} (dropped {raw_count - clean_count:,})")

            if sample_target and clean_count > sample_target:
                fraction = float(sample_target) / float(clean_count)
                seed = self.data_cfg.get("random_seed", 42)
                self.logger.info(f"Sampling down to ~{sample_target:,} rows (fraction={fraction:.4f})...")
                filtered_df = filtered_df.sample(withReplacement=False, fraction=fraction, seed=seed)

            # Drop timestamp column as hour/day are now derived
            output_df = filtered_df.drop("tpep_pickup_datetime")

            # 6. Write to clean parquet staging
            self.logger.info(f"Persisting cleaned dataset to: {output_path}")
            output_df.coalesce(1).write.mode("overwrite").parquet(output_path)

            final_count = spark.read.parquet(output_path).count()
            elapsed = time.time() - start_time
            self.logger.info(
                f"Spark Processing complete in {elapsed:.2f}s! Final cleaned records written: {final_count:,}"
            )
            return output_path

        except Exception as e:
            self.logger.warning(f"PySpark operation encountered Windows NativeIO/JVM exception: {e}")
            if spark is not None:
                try:
                    spark.stop()
                except Exception:
                    pass
            # Trigger seamless fallback
            return self._process_with_pyarrow(matched_files, output_path, start_time)

        finally:
            if spark is not None:
                try:
                    spark.stop()
                    self.logger.info("SparkSession stopped.")
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark Taxi Data Ingestion")
    parser.add_argument("--config", type=str, default="config/development.yaml", help="Path to configuration YAML")
    args = parser.parse_args()

    processor = SparkDataProcessor(config_path=args.config)
    processor.process()

