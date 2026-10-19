"""
Detailed audit of alarm timing relative to documented failure event windows.
Differentiates active anomaly detection during an ongoing failure from prior precursor lead time.
"""

import json
import torch
import pandas as pd
import numpy as np

from src.config import FAILURE_EVENTS, WINDOW_SIZE, ALL_FEATURES
from src.model import LSTMAutoencoder
from src.preprocessing import create_sliding_windows
from src.evaluate import compute_windowed_anomaly_scores


def audit_timing():
    df_test = pd.read_parquet("data/processed/test.parquet")
    stride_eval = 6
    test_ts = pd.to_datetime(df_test["timestamp"].values[WINDOW_SIZE - 1 : : stride_eval])

    with open("models/evaluation_metrics.json") as f:
        metrics = json.load(f)
    tau = metrics["threshold"]

    scaled_cols = [f"{c}_scaled" for c in ALL_FEATURES]
    test_scaled = df_test[scaled_cols].values
    test_windows = create_sliding_windows(test_scaled, window_size=WINDOW_SIZE, stride=stride_eval)
    
    model = LSTMAutoencoder(input_dim=len(ALL_FEATURES), seq_len=WINDOW_SIZE)
    model.load_state_dict(torch.load("models/best_lstm_autoencoder.pt", map_location="cpu"))
    test_errors = compute_windowed_anomaly_scores(model, test_windows)
    test_preds = (test_errors > tau).astype(int)

    print("==============================================================")
    print("      EVENT ALARM TIMING & LEAD-TIME CLARIFICATION AUDIT      ")
    print("==============================================================")
    print(f"Total test windows evaluated: {len(test_preds):,}")
    print(f"Anomaly threshold tau: {tau:.6f}")

    for ev in FAILURE_EVENTS:
        s = pd.to_datetime(ev["start"])
        e = pd.to_datetime(ev["end"])
        
        # 12 hours before failure start
        pre12_s = s - pd.Timedelta(hours=12)
        pre12_mask = (test_ts >= pre12_s) & (test_ts < s)
        pre12_preds = test_preds[pre12_mask]
        
        # 24 hours before failure start
        pre24_s = s - pd.Timedelta(hours=24)
        pre24_mask = (test_ts >= pre24_s) & (test_ts < s)
        pre24_preds = test_preds[pre24_mask]

        # During documented failure window
        dur_mask = (test_ts >= s) & (test_ts <= e)
        dur_preds = test_preds[dur_mask]
        dur_errors = test_errors[dur_mask]

        active_ts_during = test_ts[dur_mask][dur_preds == 1]
        first_alarm = active_ts_during[0] if len(active_ts_during) > 0 else None
        
        # Detection delay from start of event
        if first_alarm is not None:
            detection_delay_sec = (first_alarm - s).total_seconds()
            detection_delay_min = detection_delay_sec / 60.0
        else:
            detection_delay_min = None

        print(f"\n--- Event #{ev['id']} ({ev['name']}) ---")
        print(f"  Documented Failure Period: {s} -> {e}")
        print(f"  Duration: {(e - s).total_seconds()/3600:.2f} hours")
        print(f"  Maintenance Action: {ev['maintenance']}")
        print(f"  First Alarm During Failure: {first_alarm} (Detection delay: {detection_delay_min:.1f} minutes from start)")
        print(f"  Alarm Duty Cycle During Failure: {dur_preds.mean()*100:.2f}% ({dur_preds.sum()}/{len(dur_preds)} windows)")
        print(f"  Peak Reconstruction Error: {dur_errors.max():.4f} ({dur_errors.max()/tau:.2f}x threshold)")
        print(f"  Precursor Alarms (24h to start): {pre24_preds.mean()*100:.2f}% duty cycle ({pre24_preds.sum()}/{len(pre24_preds)} windows)")
        print(f"  Precursor Alarms (12h to start): {pre12_preds.mean()*100:.2f}% duty cycle ({pre12_preds.sum()}/{len(pre12_preds)} windows)")


if __name__ == "__main__":
    audit_timing()
