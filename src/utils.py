"""Utility functions: configuration loading, logging, and environment validation."""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def get_logger(name: str = "MLPipeline", log_level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized console logger.

    Args:
        name: Logger name.
        log_level: Logging severity level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def load_config(config_path: str = "config/development.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file into a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration settings.

    Raises:
        FileNotFoundError: If configuration file does not exist.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def ensure_directory(path: str) -> None:
    """Ensures that the directory for a given path exists."""
    p = Path(path)
    if p.suffix:  # It's a file path
        p.parent.mkdir(parents=True, exist_ok=True)
    else:  # It's a directory
        p.mkdir(parents=True, exist_ok=True)


def validate_raw_data(config_path: str = "config/development.yaml") -> bool:
    """Validates the presence and readability of raw NYC Taxi Parquet data files.

    Args:
        config_path: Path to configuration file.

    Returns:
        True if valid data files exist; False otherwise.
    """
    logger = get_logger("DataValidator")
    config = load_config(config_path)
    raw_dir = config.get("data", {}).get("raw_dir", "Data_Files")
    pattern = config.get("data", {}).get("raw_file_pattern", "*.parquet")
    
    # Check configured raw_dir, or fallback to alternates (e.g. Data_Files / Data files / data/raw)
    candidate_dirs = [raw_dir, "Data_Files", "Data files", os.path.join("data", "raw")]
    active_dir = None
    for candidate in candidate_dirs:
        if os.path.exists(candidate) and glob.glob(os.path.join(candidate, pattern)):
            active_dir = candidate
            break

    if active_dir is None:
        active_dir = raw_dir

    search_path = os.path.join(active_dir, pattern)
    files = glob.glob(search_path)
    
    if not files:
        logger.error(f"Validation FAILED: No files found matching '{search_path}'.")
        return False

    valid_files = 0
    total_bytes = 0
    for f in files:
        size = os.path.getsize(f)
        if size > 0:
            valid_files += 1
            total_bytes += size
        else:
            logger.warning(f"Empty file detected: {f}")

    mb = total_bytes / (1024 * 1024)
    logger.info(f"Validation SUCCESS: Found {valid_files} valid Parquet file(s) ({mb:.2f} MB total).")
    return valid_files > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Utilities & Validation")
    parser.add_argument("--config", type=str, default="config/development.yaml", help="Path to config YAML")
    parser.add_argument("--validate-data", action="store_true", help="Validate raw data files")
    args = parser.parse_args()

    if args.validate_data:
        is_valid = validate_raw_data(args.config)
        sys.exit(0 if is_valid else 1)

