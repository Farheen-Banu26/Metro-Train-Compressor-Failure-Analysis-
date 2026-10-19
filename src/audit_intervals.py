"""
Audit script for calculating exact samples in each failure interval.
"""

import pandas as pd
from src.config import FAILURE_EVENTS, WINDOW_SIZE


def audit_intervals():
    print("Loading raw CSV...")
    df_raw = pd.read_csv("data/raw/MetroPT3(AirCompressor).csv", usecols=["timestamp"])
    df_raw["ts"] = pd.to_datetime(df_raw["timestamp"])

    print("Loading resampled parquet...")
    df_proc = pd.read_parquet("data/processed/telemetry_resampled.parquet")
    df_proc["ts"] = pd.to_datetime(df_proc["timestamp"])

    print("Loading test parquet...")
    df_test = pd.read_parquet("data/processed/test.parquet")
    df_test["ts"] = pd.to_datetime(df_test["timestamp"])

    print("\n========================================================")
    print("      FAILURE INTERVAL ROW COUNT AUDIT DETAILS          ")
    print("========================================================")

    for ev in FAILURE_EVENTS:
        s = pd.to_datetime(ev["start"])
        e = pd.to_datetime(ev["end"])
        duration_sec = (e - s).total_seconds()
        theoretical_10s = int(duration_sec / 10) + 1

        raw_cnt = int(((df_raw["ts"] >= s) & (df_raw["ts"] <= e)).sum())
        proc_cnt = int(((df_proc["ts"] >= s) & (df_proc["ts"] <= e)).sum())
        tagged_cnt = int((df_proc["failure_id"] == ev["id"]).sum())
        test_cnt = int(((df_test["ts"] >= s) & (df_test["ts"] <= e)).sum())

        print(f"\n--- Event #{ev['id']} ({ev['name']}) ---")
        print(f"  Window: {s} -> {e} ({duration_sec/3600:.2f} hours)")
        print(f"  Theoretical continuous 10s bins: {theoretical_10s:,}")
        print(f"  Actual raw records:              {raw_cnt:,}")
        print(f"  Resampled 10s records:           {proc_cnt:,}")
        print(f"  Tagged records in test split:    {test_cnt:,}")
        print(f"  Tagged count vs Theoretical:     {proc_cnt / theoretical_10s * 100:.2f}% completeness")

    # Now let's audit sliding windows at evaluation stride
    stride_eval = 6  # 1-minute resolution
    test_labels = df_test["is_failure_period"].values[WINDOW_SIZE - 1 : : stride_eval]
    test_timestamps = df_test["ts"].values[WINDOW_SIZE - 1 : : stride_eval]

    print("\n========================================================")
    print(f" EVALUATION WINDOWS AUDIT (stride={stride_eval}, W={WINDOW_SIZE})")
    print("========================================================")
    print(f"Total test sequence windows: {len(test_labels):,}")
    print(f"Total positive (failure) windows: {int(test_labels.sum()):,} ({test_labels.mean()*100:.2f}%)")
    print(f"Total negative (normal) windows:  {int((test_labels == 0).sum()):,} ({(test_labels == 0).mean()*100:.2f}%)")

    for ev in FAILURE_EVENTS:
        s = pd.to_datetime(ev["start"])
        e = pd.to_datetime(ev["end"])
        mask = (pd.to_datetime(test_timestamps) >= s) & (pd.to_datetime(test_timestamps) <= e)
        ev_window_cnt = int(mask.sum())
        duration_min = (e - s).total_seconds() / 60
        theoretical_eval_windows = int(duration_min)
        print(f"  Event #{ev['id']}: {ev_window_cnt:,} eval windows (Theoretical 1-min windows: ~{theoretical_eval_windows:,})")


if __name__ == "__main__":
    audit_intervals()
