"""
Inference module for Metro Train Compressor Failure Analysis.
Provides model loading, input validation, sliding window generation,
reconstruction error scoring, and anomaly threshold comparison.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.config import (
    ALL_FEATURES,
    ANALOG_FEATURES,
    DIGITAL_FEATURES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TIMESTAMP_COL,
    WINDOW_SIZE,
)
from src.model import LSTMAutoencoder
from src.preprocessing import create_sliding_windows


# Cache loaded artifacts to avoid reloading from disk on every Streamlit rerun
_CACHED_ARTIFACTS: Optional[Dict] = None


def load_inference_artifacts(device: str = "cpu") -> Dict:
    """
    Loads the trained LSTM autoencoder weights, fitted StandardScaler,
    and anomaly threshold from disk.
    """
    global _CACHED_ARTIFACTS
    if _CACHED_ARTIFACTS is not None and _CACHED_ARTIFACTS["device"] == device:
        return _CACHED_ARTIFACTS

    model_path = MODELS_DIR / "best_lstm_autoencoder.pt"
    scaler_path = MODELS_DIR / "scaler.pkl"
    metrics_path = MODELS_DIR / "evaluation_metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Fitted scaler not found at {scaler_path}. Run preprocessing first.")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found at {metrics_path}. Run evaluation first.")

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load threshold from evaluation metrics
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    threshold = float(metrics["threshold"])

    # Load PyTorch model
    model = LSTMAutoencoder(input_dim=len(ALL_FEATURES), seq_len=WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _CACHED_ARTIFACTS = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "threshold_method": metrics.get("threshold_method", "99th percentile of validation reconstruction error"),
        "features": list(ALL_FEATURES),
        "window_size": WINDOW_SIZE,
        "device": device,
    }
    return _CACHED_ARTIFACTS


def validate_input_dataframe(
    df: pd.DataFrame,
    min_rows: int = WINDOW_SIZE
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates uploaded or manual input data.
    Checks:
    1. Presence of all 15 required domain features.
    2. Minimum required sequence length (min_rows, default 60).
    3. Non-null and numeric validity.
    4. Proper timestamp handling (parses existing timestamp or synthesizes 10s steps).
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, "Input data is empty or not a valid DataFrame.", None

    if len(df) < min_rows:
        return (
            False,
            f"Insufficient sequence length: Received {len(df)} rows, but the model requires at least "
            f"{min_rows} consecutive time steps (10 minutes of operational telemetry at 10-second resolution).",
            None
        )

    # Check for missing feature columns
    missing_cols = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_cols:
        return (
            False,
            f"Missing required sensor columns: {missing_cols}. "
            f"The input must contain all 15 features: {ALL_FEATURES}.",
            None
        )

    df_clean = df.copy()

    # Handle timestamp column
    if TIMESTAMP_COL in df_clean.columns:
        try:
            df_clean[TIMESTAMP_COL] = pd.to_datetime(df_clean[TIMESTAMP_COL])
            df_clean = df_clean.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)
        except Exception as e:
            return False, f"Failed to parse timestamp column: {str(e)}", None
    else:
        # Generate synthetic regularized 10-second timestamps
        base_time = pd.Timestamp.now(tz="UTC").floor("min")
        df_clean[TIMESTAMP_COL] = [base_time + pd.Timedelta(seconds=10 * i) for i in range(len(df_clean))]

    # Validate numeric types and missing values in features
    for col in ALL_FEATURES:
        try:
            df_clean[col] = pd.to_numeric(df_clean[col])
        except Exception:
            return False, f"Column '{col}' contains non-numeric values that cannot be parsed as floats.", None

        # Check for NaNs
        null_count = df_clean[col].isna().sum()
        if null_count > 0:
            if null_count > len(df_clean) * 0.1:
                return (
                    False,
                    f"Column '{col}' has {null_count} missing values ({null_count / len(df_clean):.1%}), "
                    "which exceeds the 10% interpolation limit.",
                    None
                )
            # Forward/backward fill minor null gaps
            df_clean[col] = df_clean[col].interpolate(method="linear").bfill().ffill()

    # Reorder columns: timestamp first, then exact ALL_FEATURES order
    ordered_cols = [TIMESTAMP_COL] + list(ALL_FEATURES)
    df_clean = df_clean[ordered_cols].reset_index(drop=True)

    return True, "Validation successful.", df_clean


def run_inference(
    df_valid: pd.DataFrame,
    artifacts: Dict,
    stride: int = 1,
    batch_size: int = 256
) -> Tuple[pd.DataFrame, Dict]:
    """
    Performs genuine inference on a validated input dataframe.
    Scales features with fitted StandardScaler, slices into sliding windows (length 60),
    computes reconstruction error with the trained LSTM Autoencoder, and classifies
    each window against the validation threshold tau.
    """
    model: LSTMAutoencoder = artifacts["model"]
    scaler = artifacts["scaler"]
    threshold = artifacts["threshold"]
    device = artifacts["device"]

    model.eval()

    # Feature scaling (strictly transform with fitted scaler)
    scaled_array = scaler.transform(df_valid[ALL_FEATURES])

    # Create sliding windows of shape (num_windows, 60, 15)
    windows = create_sliding_windows(scaled_array, window_size=WINDOW_SIZE, stride=stride)
    num_windows = len(windows)

    # Compute window reconstruction error
    window_errors: List[float] = []
    channel_errors_list: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch = torch.tensor(windows[i : i + batch_size], dtype=torch.float32).to(device)
            err = model.compute_reconstruction_error(batch)
            window_errors.extend(err.cpu().numpy().tolist())

            # Per-channel MSE for sensor attribution: (batch_size, num_features)
            recon = model(batch)
            ch_err = torch.mean((batch - recon) ** 2, dim=1).cpu().numpy()
            channel_errors_list.append(ch_err)

    channel_errors_all = np.vstack(channel_errors_list)

    # Compile results per window
    results = []
    for idx in range(num_windows):
        start_row = idx * stride
        end_row = start_row + WINDOW_SIZE - 1

        start_ts = df_valid[TIMESTAMP_COL].iloc[start_row]
        end_ts = df_valid[TIMESTAMP_COL].iloc[end_row]

        err = float(window_errors[idx])
        diff = err - threshold
        is_anomaly = bool(err > threshold)
        status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"

        # Sensor that contributed the largest MSE to this window's reconstruction error
        top_sensor_idx = int(np.argmax(channel_errors_all[idx]))
        top_sensor = ALL_FEATURES[top_sensor_idx]
        top_sensor_err = float(channel_errors_all[idx, top_sensor_idx])

        results.append({
            "window_id": idx + 1,
            "start_time": str(start_ts),
            "end_time": str(end_ts),
            "reconstruction_error": round(err, 6),
            "threshold": round(threshold, 6),
            "diff_from_threshold": round(diff, 6),
            "status": status,
            "is_anomaly": is_anomaly,
            "top_contributor": top_sensor,
            "top_contributor_mse": round(top_sensor_err, 6),
        })

    results_df = pd.DataFrame(results)

    # Calculate overall summary
    total_w = len(results_df)
    anom_w = int(results_df["is_anomaly"].sum())
    anom_pct = round((anom_w / total_w) * 100, 2) if total_w > 0 else 0.0

    summary = {
        "total_windows": total_w,
        "anomalous_windows": anom_w,
        "normal_windows": total_w - anom_w,
        "anomaly_percentage": anom_pct,
        "mean_reconstruction_error": round(float(np.mean(window_errors)), 6),
        "max_reconstruction_error": round(float(np.max(window_errors)), 6),
        "min_reconstruction_error": round(float(np.min(window_errors)), 6),
        "threshold": round(threshold, 6),
        "threshold_method": artifacts.get("threshold_method", ""),
        "overall_status": "ANOMALY DETECTED" if anom_w > 0 else "NORMAL",
        "primary_contributing_sensor": results_df["top_contributor"].mode()[0] if not results_df.empty else "N/A",
    }

    return results_df, summary


def get_historical_sample(sample_key: str) -> pd.DataFrame:
    """
    Extracts a real, genuine sequence from the processed test telemetry.
    Clearly labeled with exact timestamp ranges and physical operating context.
    """
    test_parquet = PROCESSED_DATA_DIR / "test.parquet"
    if not test_parquet.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_parquet}")

    df_test = pd.read_parquet(test_parquet)

    samples = {
        "normal_baseline": {
            "name": "Normal Baseline (Healthy Pumping Cycle)",
            "start": "2020-04-16 04:40:20",
            "end": "2020-04-16 04:50:10",
            "length": 60,
            "expected": "NORMAL",
            "desc": "Regular periodic compressor cycle with healthy reservoir pressure (8.5–10 bar) and low resting motor current."
        },
        "normal_multicycle": {
            "name": "Normal Multi-Cycle (20-Minute Stream, 120 steps)",
            "start": "2020-04-16 05:00:00",
            "end": "2020-04-16 05:19:50",
            "length": 120,
            "expected": "NORMAL",
            "desc": "Two full consecutive normal compressor pumping/idling cycles with zero air leaks."
        },
        "event_1_leak": {
            "name": "Failure Event #1 (Active Air Leak)",
            "start": "2020-04-18 14:08:20",
            "end": "2020-04-18 14:18:10",
            "length": 60,
            "expected": "ANOMALY DETECTED",
            "desc": "Documented air leak on April 18 causing continuous motor strain and rapid pressure loss."
        },
        "event_2_leak": {
            "name": "Failure Event #2 (Active Air Leak Before Depot Overhaul)",
            "start": "2020-05-30 01:06:40",
            "end": "2020-05-30 01:16:30",
            "length": 60,
            "expected": "ANOMALY DETECTED",
            "desc": "Active air leak preceding May 30 depot overhaul, exhibiting sustained high error."
        },
        "event_3_breakdown": {
            "name": "Failure Event #3 (Catastrophic Pneumatic Line Breakdown)",
            "start": "2020-06-06 09:25:00",
            "end": "2020-06-06 09:34:50",
            "length": 60,
            "expected": "ANOMALY DETECTED",
            "desc": "Severe pneumatic line breakdown with extreme motor duty cycle and pressure venting."
        },
        "event_4_pressure_drop": {
            "name": "Failure Event #4 (Reservoir Pressure Drop)",
            "start": "2020-07-15 17:35:20",
            "end": "2020-07-15 17:45:10",
            "length": 60,
            "expected": "ANOMALY DETECTED",
            "desc": "Air leak episode with accelerated governor switching and high crankcase temperature."
        },
    }

    if sample_key not in samples:
        raise ValueError(f"Unknown sample key '{sample_key}'. Available: {list(samples.keys())}")

    meta = samples[sample_key]
    mask = (df_test[TIMESTAMP_COL] >= meta["start"]) & (df_test[TIMESTAMP_COL] <= meta["end"])
    slice_df = df_test.loc[mask, [TIMESTAMP_COL] + list(ALL_FEATURES)].copy().reset_index(drop=True)

    # Truncate to exact required length if slightly over
    if len(slice_df) > meta["length"]:
        slice_df = slice_df.iloc[: meta["length"]]

    return slice_df


def get_csv_template(sample_type: str = "nominal") -> str:
    """
    Generates a downloadable CSV template with the exact 15 required domain features.
    If sample_type == 'nominal', pre-fills 60 rows with genuine baseline values.
    If sample_type == 'blank', provides 60 timestamped rows with default nominal values.
    """
    sample_df = get_historical_sample("normal_baseline")
    if sample_type == "blank":
        # Neutral resting baseline values
        blank_df = sample_df.copy()
        blank_df["TP2"] = 0.0
        blank_df["TP3"] = 9.0
        blank_df["H1"] = 7.5
        blank_df["DV_pressure"] = 0.0
        blank_df["Reservoirs"] = 9.0
        blank_df["Motor_current"] = 0.0
        blank_df["Oil_temperature"] = 60.0
        blank_df["COMP"] = 1
        blank_df["DV_eletric"] = 0
        blank_df["Towers"] = 1
        blank_df["MPG"] = 1
        blank_df["LPS"] = 0
        blank_df["Pressure_switch"] = 1
        blank_df["Oil_level"] = 1
        blank_df["Caudal_impulses"] = 1
        return blank_df.to_csv(index=False)
    else:
        return sample_df.to_csv(index=False)
