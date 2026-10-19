"""
Comprehensive audit script for inspecting models, history, metrics, figures, and data partitions.
"""

import json
import pickle
from pathlib import Path
from PIL import Image
import torch
import pandas as pd
import numpy as np

from src.config import (
    MODELS_DIR,
    FIGURES_EDA_DIR,
    FIGURES_EVAL_DIR,
    PROCESSED_DATA_DIR,
    ALL_FEATURES,
    TOTAL_FEATURE_COUNT,
    TIMESTAMP_COL,
)


def run_audit():
    print("==================================================")
    print("          PROJECT COMPONENT AUDIT REPORT          ")
    print("==================================================")

    # 1. Model Checkpoint
    pt_path = MODELS_DIR / "best_lstm_autoencoder.pt"
    if not pt_path.exists():
        print(f"[FAIL] Checkpoint missing: {pt_path}")
        return False
    weights = torch.load(pt_path, map_location="cpu")
    print(f"[PASS] PyTorch Checkpoint verified: {pt_path} ({pt_path.stat().st_size:,} bytes, {len(weights)} parameter tensors)")

    # 2. Training History
    history_path = MODELS_DIR / "training_history.json"
    if not history_path.exists():
        print(f"[FAIL] History missing: {history_path}")
        return False
    with open(history_path, "r") as f:
        hist = json.load(f)
    print(f"[PASS] Training History verified: {len(hist['epochs'])} epochs completed")
    print(f"       Initial: Train MSE = {hist['train_loss'][0]:.6f}, Val MSE = {hist['val_loss'][0]:.6f}")
    print(f"       Final:   Train MSE = {hist['train_loss'][-1]:.6f}, Val MSE = {hist['val_loss'][-1]:.6f}")

    # 3. Scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[PASS] Fitted Scaler verified: {scaler.__class__.__name__} (features: {scaler.n_features_in_})")

    # 4. Evaluation Metrics
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    print(f"[PASS] Evaluation Metrics verified:")
    print(f"       Decision Threshold (tau): {metrics['threshold']:.6f}")
    print(f"       ROC-AUC: {metrics['test_metrics']['roc_auc']:.4f}")
    print(f"       PR-AUC:  {metrics['test_metrics']['pr_auc']:.4f}")
    print(f"       Recall:  {metrics['test_metrics']['recall']*100:.2f}%")
    print(f"       Precision: {metrics['test_metrics']['precision']*100:.2f}%")
    print(f"       F1-Score:  {metrics['test_metrics']['f1_score']:.4f}")
    print("       Event Detection:")
    for ev in metrics["event_level_evaluation"]:
        print(f"         * Event #{ev['event_id']} ({ev['name']}): detected={ev['detected']}, duty_cycle={ev['alarm_duty_cycle']}%, peak_error={ev['max_reconstruction_error']:.4f}")

    # 5. Data Partitions & Data Leakage Verification
    train_df = pd.read_parquet(PROCESSED_DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")
    print("\n[PARTITIONS & DATA LEAKAGE AUDIT]")
    print(f"       Train: {len(train_df):,} rows | {train_df[TIMESTAMP_COL].min()} -> {train_df[TIMESTAMP_COL].max()} | Failure records: {train_df['is_failure_period'].sum()}")
    print(f"       Val:   {len(val_df):,} rows | {val_df[TIMESTAMP_COL].min()} -> {val_df[TIMESTAMP_COL].max()} | Failure records: {val_df['is_failure_period'].sum()}")
    print(f"       Test:  {len(test_df):,} rows | {test_df[TIMESTAMP_COL].min()} -> {test_df[TIMESTAMP_COL].max()} | Failure records: {test_df['is_failure_period'].sum():,}")

    assert train_df['is_failure_period'].sum() == 0, "DATA LEAKAGE: Failure data found in training set!"
    assert val_df['is_failure_period'].sum() == 0, "DATA LEAKAGE: Failure data found in validation set!"
    assert pd.to_datetime(train_df[TIMESTAMP_COL].max()) < pd.to_datetime(val_df[TIMESTAMP_COL].min()), "Chronological boundary violation: Train overlaps with Val!"
    assert pd.to_datetime(val_df[TIMESTAMP_COL].max()) < pd.to_datetime(test_df[TIMESTAMP_COL].min()), "Chronological boundary violation: Val overlaps with Test!"
    print("[PASS] ZERO DATA LEAKAGE CONFIRMED: Strict chronological boundaries and clean partitions.")

    # 6. Figures Verification (Opening with Pillow to verify image validity)
    print("\n[FIGURES AUDIT]")
    eda_figs = sorted(list(FIGURES_EDA_DIR.glob("*.png")))
    print(f"Found {len(eda_figs)} EDA figures in {FIGURES_EDA_DIR}:")
    for f in eda_figs:
        img = Image.open(f)
        img.verify()  # Verifies file integrity
        print(f" [PASS] {f.name}: Dimensions {img.size}, {f.stat().st_size:,} bytes")

    eval_figs = sorted(list(FIGURES_EVAL_DIR.glob("*.png")))
    print(f"\nFound {len(eval_figs)} Evaluation figures in {FIGURES_EVAL_DIR}:")
    for f in eval_figs:
        img = Image.open(f)
        img.verify()  # Verifies file integrity
        print(f" [PASS] {f.name}: Dimensions {img.size}, {f.stat().st_size:,} bytes")

    print("\n==================================================")
    print("            ALL AUDIT CHECKS PASSED               ")
    print("==================================================")
    return True


if __name__ == "__main__":
    run_audit()
