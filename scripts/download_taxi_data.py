"""Script to download real NYC TLC Yellow Taxi Parquet files from official cloud repository."""

import argparse
import os
import urllib.request
from pathlib import Path


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def download_taxi_data(year: int = 2024, months: list[int] = [1, 2], output_dir: str = "Data files") -> None:
    """Downloads official monthly Parquet files from NYC TLC public CloudFront distribution.

    Args:
        year: Year to download (e.g., 2024).
        months: List of month integers (1 to 12).
        output_dir: Target local directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for m in months:
        filename = f"yellow_tripdata_{year}-{m:02d}.parquet"
        target_file = out_path / filename
        if target_file.exists():
            print(f"File {filename} already exists, skipping.")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"Downloading {url} -> {target_file}...")
        try:
            urllib.request.urlretrieve(url, target_file)
            size_mb = os.path.getsize(target_file) / (1024 * 1024)
            print(f"Successfully downloaded {filename} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"Failed to download {url}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NYC Taxi Parquet data.")
    parser.add_argument("--year", type=int, default=2024, help="Year of trip data")
    parser.add_argument("--months", nargs="+", type=int, default=[1, 2], help="List of months")
    parser.add_argument("--output-dir", type=str, default="Data files", help="Destination folder")
    args = parser.parse_args()

    download_taxi_data(year=args.year, months=args.months, output_dir=args.output_dir)

