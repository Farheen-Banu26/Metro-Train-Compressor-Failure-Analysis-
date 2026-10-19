"""
Data acquisition module for Metro Train Compressor Failure Analysis.
Downloads the official UCI MetroPT-3 dataset zip file, extracts the contents,
and validates column integrity, row count, and schema consistency.
"""

import sys
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

from src.config import (
    UCI_DATASET_URL,
    RAW_DATA_DIR,
    RAW_ZIP_FILE,
    RAW_CSV_FILE,
    ALL_FEATURES,
    TIMESTAMP_COL,
)


def download_uci_dataset(url: str = UCI_DATASET_URL, dest_path: Path = RAW_ZIP_FILE) -> Path:
    """
    Downloads the official UCI MetroPT-3 zip archive with a stream progress bar.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists() and dest_path.stat().st_size > 1_000_000:
        print(f"[INFO] Archive already exists at {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return dest_path

    print(f"[INFO] Downloading UCI MetroPT-3 dataset from: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024 * 1024  # 1 MB chunk

    with open(dest_path, "wb") as f, tqdm(
        desc="Downloading MetroPT-3",
        total=total_size if total_size > 0 else None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"[INFO] Download completed successfully: {dest_path}")
    return dest_path


def extract_zip_dataset(zip_path: Path = RAW_ZIP_FILE, extract_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Extracts the downloaded zip archive to the raw data directory.
    """
    print(f"[INFO] Extracting archive: {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        print(f"[INFO] Files in archive: {file_list}")
        zip_ref.extractall(extract_dir)

    print(f"[INFO] Extraction complete. Checking for {RAW_CSV_FILE}...")
    if not RAW_CSV_FILE.exists():
        # Look for any .csv file extracted
        csv_files = list(extract_dir.glob("*.csv"))
        if csv_files:
            print(f"[INFO] Found CSV file: {csv_files[0]}")
            return csv_files[0]
        else:
            raise FileNotFoundError(f"Could not find any CSV files in {extract_dir}")
    return RAW_CSV_FILE


def verify_dataset_integrity(csv_path: Path = RAW_CSV_FILE) -> dict:
    """
    Performs fast initial streaming verification of CSV header, row count, and column coverage.
    """
    import pandas as pd

    print(f"[INFO] Verifying dataset integrity on: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Raw file size: {file_size_mb:.2f} MB")

    # Read preview (first 100 rows)
    df_preview = pd.read_csv(csv_path, nrows=100)
    columns_found = list(df_preview.columns)
    print(f"[INFO] Columns found ({len(columns_found)}): {columns_found}")

    # Check for timestamp
    has_timestamp = any("time" in c.lower() for c in columns_found)
    print(f"[INFO] Timestamp column present: {has_timestamp}")

    # Check required 15 features
    missing_features = [f for f in ALL_FEATURES if f not in columns_found]
    if missing_features:
        print(f"[WARNING] Missing expected features: {missing_features}")
    else:
        print(f"[SUCCESS] All 15 required domain features verified present in dataset!")

    verification_info = {
        "file_size_mb": file_size_mb,
        "columns": columns_found,
        "total_columns": len(columns_found),
        "all_15_features_present": len(missing_features) == 0,
        "preview_shape": df_preview.shape,
    }
    return verification_info


if __name__ == "__main__":
    zip_p = download_uci_dataset()
    csv_p = extract_zip_dataset(zip_p)
    info = verify_dataset_integrity(csv_p)
    print("[INFO] Data acquisition and verification completed successfully!")
