"""
Data preprocessing, cleaning, chronological partitioning, and feature scaling module.
Ensures zero data leakage by fitting transformers strictly on the healthy training partition.
Generates 10-second resampled operational datasets and sliding window tensors.
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from src.config import (
    RAW_CSV_FILE,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    ANALOG_FEATURES,
    DIGITAL_FEATURES,
    ALL_FEATURES,
    TIMESTAMP_COL,
    RESAMPLE_FREQ,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    FAILURE_EVENTS,
    WINDOW_SIZE,
    STRIDE_TRAIN,
    STRIDE_TEST,
)


def load_raw_dataset(csv_path: Path = RAW_CSV_FILE) -> pd.DataFrame:
    """
    Loads raw CSV dataset, parses timestamps, and sorts chronologically.
    """
    print(f"[PREPROCESS] Loading raw telemetry from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Standardize column names if needed
    df.columns = [c.strip() for c in df.columns]

    # Convert timestamp
    if TIMESTAMP_COL in df.columns:
        print("[PREPROCESS] Parsing datetime timestamps...")
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
        df = df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)
    else:
        raise KeyError(f"Timestamp column '{TIMESTAMP_COL}' not found in {df.columns}")

    return df


def audit_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data hygiene: audits missing values, duplicates, and physical limits.
    """
    print(f"[PREPROCESS] Starting data audit on {len(df):,} records...")

    # 1. Missing Values
    null_counts = df[ALL_FEATURES].isnull().sum()
    total_nulls = null_counts.sum()
    print(f"[PREPROCESS] Total missing values across features: {total_nulls}")
    if total_nulls > 0:
        print("[PREPROCESS] Missing values breakdown:")
        print(null_counts[null_counts > 0])
        print("[PREPROCESS] Performing forward-fill followed by backward-fill for continuous telemetry...")
        df[ALL_FEATURES] = df[ALL_FEATURES].ffill().bfill()

    # 2. Duplicate Timestamps
    dup_count = df.duplicated(subset=[TIMESTAMP_COL]).sum()
    print(f"[PREPROCESS] Duplicate timestamps detected: {dup_count}")
    if dup_count > 0:
        print("[PREPROCESS] Dropping duplicate timestamps keeping first occurrence...")
        df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="first")

    # 3. Physical Bound Checks
    print("[PREPROCESS] Auditing physical boundaries on analog sensors...")
    # Bar pressures shouldn't be drastically negative (< -0.5 bar sensor drift)
    for p_col in ["TP2", "TP3", "Reservoirs", "H1", "DV_pressure"]:
        if p_col in df.columns:
            neg_mask = df[p_col] < -1.0
            if neg_mask.sum() > 0:
                print(f"[PREPROCESS] Warning: {neg_mask.sum()} readings in {p_col} < -1.0 bar clipped to 0.0.")
                df.loc[neg_mask, p_col] = 0.0

    # Motor current cannot be physically negative
    if "Motor_current" in df.columns:
        df["Motor_current"] = df["Motor_current"].clip(lower=0.0)

    # Digital flags should strictly be 0 or 1
    for d_col in DIGITAL_FEATURES:
        if d_col in df.columns:
            df[d_col] = (df[d_col] > 0.5).astype(float)

    print("[PREPROCESS] Data hygiene and physical boundary audit completed.")
    return df


def resample_telemetry(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Downsamples 1Hz telemetry to 10s intervals using domain aggregations.
    Preserves macro thermodynamic cycles while reducing noise and size.
    """
    print(f"[PREPROCESS] Resampling telemetry to {freq} frequency...")
    df_ts = df.set_index(TIMESTAMP_COL)

    # Aggregate: mean for analog signals, max for digital flags (if active during interval)
    agg_rules = {}
    for col in ANALOG_FEATURES:
        if col in df_ts.columns:
            agg_rules[col] = "mean"
    for col in DIGITAL_FEATURES:
        if col in df_ts.columns:
            agg_rules[col] = "max"

    df_resampled = df_ts.resample(freq).agg(agg_rules).dropna().reset_index()
    print(f"[PREPROCESS] Resampled shape: {df_resampled.shape} (reduced from {len(df):,} rows)")
    return df_resampled


def label_ground_truth_failure_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean 'is_failure_period' and 'failure_id' based strictly on official operator maintenance logs.
    NOTE: Used SOLELY for evaluation and benchmarking, never for training the autoencoder!
    """
    df["is_failure_period"] = 0
    df["failure_id"] = 0

    for event in FAILURE_EVENTS:
        start = pd.to_datetime(event["start"])
        end = pd.to_datetime(event["end"])
        mask = (df[TIMESTAMP_COL] >= start) & (df[TIMESTAMP_COL] <= end)
        df.loc[mask, "is_failure_period"] = 1
        df.loc[mask, "failure_id"] = event["id"]
        print(f"[PREPROCESS] Event #{event['id']} ({event['name']}): tagged {mask.sum():,} records between {start} and {end}")

    return df


def create_chronological_splits(df: pd.DataFrame):
    """
    Splits data chronologically to strictly avoid any data leakage.
    Train: Baseline healthy operation (Feb 28 - Mar 31)
    Val:   Healthy validation tuning (Apr 01 - Apr 15)
    Test:  Degradation, all 4 failure incidents, and post-maintenance (Apr 16 - Aug 31)
    """
    print("[PREPROCESS] Partitioning dataset chronologically...")
    train_mask = (df[TIMESTAMP_COL] >= pd.to_datetime(TRAIN_START)) & (df[TIMESTAMP_COL] <= pd.to_datetime(TRAIN_END))
    val_mask   = (df[TIMESTAMP_COL] >= pd.to_datetime(VAL_START))   & (df[TIMESTAMP_COL] <= pd.to_datetime(VAL_END))
    test_mask  = (df[TIMESTAMP_COL] >= pd.to_datetime(TEST_START))  & (df[TIMESTAMP_COL] <= pd.to_datetime(TEST_END))

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val   = df[val_mask].copy().reset_index(drop=True)
    df_test  = df[test_mask].copy().reset_index(drop=True)

    print(f"[PREPROCESS] Train partition: {df_train.shape[0]:,} records ({df_train[TIMESTAMP_COL].min()} -> {df_train[TIMESTAMP_COL].max()})")
    print(f"[PREPROCESS] Val partition:   {df_val.shape[0]:,} records ({df_val[TIMESTAMP_COL].min()} -> {df_val[TIMESTAMP_COL].max()})")
    print(f"[PREPROCESS] Test partition:  {df_test.shape[0]:,} records ({df_test[TIMESTAMP_COL].min()} -> {df_test[TIMESTAMP_COL].max()})")

    # Fit Scaler STRICTLY on Train partition (StandardScaler prevents near-zero IQR explosions on resting sensors)
    scaler = StandardScaler()
    print("[PREPROCESS] Fitting StandardScaler strictly on the healthy training partition...")
    train_scaled = scaler.fit_transform(df_train[ALL_FEATURES])
    val_scaled   = scaler.transform(df_val[ALL_FEATURES])
    test_scaled  = scaler.transform(df_test[ALL_FEATURES])

    # Add scaled columns
    for idx, col in enumerate(ALL_FEATURES):
        df_train[f"{col}_scaled"] = train_scaled[:, idx]
        df_val[f"{col}_scaled"]   = val_scaled[:, idx]
        df_test[f"{col}_scaled"]  = test_scaled[:, idx]

    # Save fitted scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[PREPROCESS] Scaler saved to {scaler_path}")

    # Save processed splits to parquet
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
    df_val.to_parquet(PROCESSED_DATA_DIR / "val.parquet", index=False)
    df_test.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)
    df.to_parquet(PROCESSED_DATA_DIR / "telemetry_resampled.parquet", index=False)
    print(f"[PREPROCESS] Processed Parquet files written to {PROCESSED_DATA_DIR}")

    return df_train, df_val, df_test, scaler


def create_sliding_windows(
    data_array: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE_TRAIN
) -> np.ndarray:
    """
    Converts 2D scaled feature matrix (N, num_features) into 3D window tensors (num_windows, window_size, num_features).
    """
    n_samples, n_features = data_array.shape
    if n_samples < window_size:
        raise ValueError(f"Sample length {n_samples} is smaller than window size {window_size}")

    windows = []
    for start in range(0, n_samples - window_size + 1, stride):
        windows.append(data_array[start : start + window_size])

    return np.array(windows, dtype=np.float32)


def run_preprocessing_pipeline():
    """
    Complete end-to-end preprocessing pipeline execution.
    """
    df_raw = load_raw_dataset()
    df_clean = audit_and_clean_data(df_raw)
    df_resampled = resample_telemetry(df_clean, freq=RESAMPLE_FREQ)
    df_tagged = label_ground_truth_failure_intervals(df_resampled)
    df_train, df_val, df_test, scaler = create_chronological_splits(df_tagged)
    print("[PREPROCESS] Pipeline successfully completed!")
    return df_train, df_val, df_test


if __name__ == "__main__":
    run_preprocessing_pipeline()
