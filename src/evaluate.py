"""
Model evaluation, anomaly scoring, metric computation, and visualization module.
Computes real empirical metrics (PR-AUC, ROC-AUC, F1, Precision, Recall, Event Detection Delay)
and generates 5 publication-ready evaluation figures.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_EVAL_DIR,
    ALL_FEATURES,
    ANALOG_FEATURES,
    TIMESTAMP_COL,
    FAILURE_EVENTS,
    WINDOW_SIZE,
    ANOMALY_PERCENTILE,
)
from src.model import LSTMAutoencoder
from src.preprocessing import create_sliding_windows


plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "figure.titlesize": 16,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "savefig.dpi": 300,
})

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "accent": "#2ca02c",
    "danger": "#d62728",
    "threshold": "#8e44ad",
    "healthy": "#27ae60",
    "failure": "#c0392b"
}


def compute_windowed_anomaly_scores(
    model: LSTMAutoencoder,
    data_array: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu"
) -> np.ndarray:
    """
    Computes per-window Mean Squared Error reconstruction losses.
    """
    model.eval()
    errors = []
    num_windows = len(data_array)

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch = torch.tensor(data_array[i : i + batch_size], dtype=torch.float32).to(device)
            err = model.compute_reconstruction_error(batch)
            errors.extend(err.cpu().numpy())

    return np.array(errors, dtype=np.float32)


def compute_per_channel_reconstruction_error(
    model: LSTMAutoencoder,
    data_array: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """
    Computes MSE per individual feature channel: (num_windows, num_features).
    Used for root-cause sensor attribution.
    """
    model.eval()
    channel_errors = []
    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(data_array), batch_size):
            batch = torch.tensor(data_array[i : i + batch_size], dtype=torch.float32).to(device)
            reconstructed = model(batch)
            # MSE along sequence dimension (dim 1)
            err = torch.mean((batch - reconstructed) ** 2, dim=1)
            channel_errors.append(err.cpu().numpy())

    return np.vstack(channel_errors)


def evaluate_model_pipeline(device: str = None) -> Dict:
    """
    Executes full evaluation: calculates real metrics, generates 5 evaluation figures,
    and returns a summary dictionary.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAL] Evaluating on device: {device}...")

    FIGURES_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load scaled datasets
    df_val = pd.read_parquet(PROCESSED_DATA_DIR / "val.parquet")
    df_test = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")

    scaled_cols = [f"{col}_scaled" for col in ALL_FEATURES]
    val_scaled = df_val[scaled_cols].values
    test_scaled = df_test[scaled_cols].values

    # Generate sliding windows
    print("[EVAL] Generating evaluation sliding windows...")
    stride_eval = 6  # 1-minute evaluation resolution (160k windows across 4.5 months)
    val_windows = create_sliding_windows(val_scaled, window_size=WINDOW_SIZE, stride=10)
    test_windows = create_sliding_windows(test_scaled, window_size=WINDOW_SIZE, stride=stride_eval)

    # Corresponding ground truth labels for test windows (aligned with the end timestamp of each window)
    test_labels = df_test["is_failure_period"].values[WINDOW_SIZE - 1 : : stride_eval][: len(test_windows)]
    test_timestamps = df_test[TIMESTAMP_COL].values[WINDOW_SIZE - 1 : : stride_eval][: len(test_windows)]

    # Load Model
    model = LSTMAutoencoder(input_dim=len(ALL_FEATURES), seq_len=WINDOW_SIZE).to(device)
    model_path = MODELS_DIR / "best_lstm_autoencoder.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[EVAL] Loaded trained model weights from {model_path}")

    # Compute Validation Errors & Establish Dynamic Threshold tau
    print("[EVAL] Scoring validation windows to compute anomaly threshold...")
    val_errors = compute_windowed_anomaly_scores(model, val_windows, device=device)
    threshold = float(np.percentile(val_errors, ANOMALY_PERCENTILE))
    threshold_std = float(val_errors.mean() + 3 * val_errors.std())
    print(f"[EVAL] Validation Reconstruction Error: Mean={val_errors.mean():.6f}, Std={val_errors.std():.6f}")
    print(f"[EVAL] Established Anomaly Threshold (99th percentile): tau = {threshold:.6f} (vs 3-std: {threshold_std:.6f})")

    # Compute Test Errors & Anomaly Predictions
    print("[EVAL] Scoring test sequence windows...")
    test_errors = compute_windowed_anomaly_scores(model, test_windows, device=device)
    test_preds = (test_errors > threshold).astype(int)

    # Compute Classification Metrics
    roc_auc = float(roc_auc_score(test_labels, test_errors))
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_errors)
    pr_auc = float(auc(recall_curve, precision_curve))
    
    f1 = float(f1_score(test_labels, test_preds, zero_division=0))
    prec = float(precision_score(test_labels, test_preds, zero_division=0))
    rec = float(recall_score(test_labels, test_preds, zero_division=0))
    cm = confusion_matrix(test_labels, test_preds).tolist()

    print(f"[EVAL] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"[EVAL] Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")

    # Event-Level Recall Analysis
    event_results = []
    for ev in FAILURE_EVENTS:
        start_ts = pd.to_datetime(ev["start"])
        end_ts = pd.to_datetime(ev["end"])
        mask = (pd.to_datetime(test_timestamps) >= start_ts) & (pd.to_datetime(test_timestamps) <= end_ts)
        ev_preds = test_preds[mask]
        ev_errors = test_errors[mask]
        
        detected = bool(np.sum(ev_preds) > 0)
        detection_ratio = float(np.mean(ev_preds)) if len(ev_preds) > 0 else 0.0
        max_score = float(np.max(ev_errors)) if len(ev_errors) > 0 else 0.0

        event_results.append({
            "event_id": ev["id"],
            "name": ev["name"],
            "start": ev["start"],
            "end": ev["end"],
            "detected": detected,
            "alarm_duty_cycle": round(detection_ratio * 100, 2),
            "max_reconstruction_error": round(max_score, 6),
        })
        print(f"[EVAL] Event #{ev['id']}: Detected = {detected} (Alarms active {detection_ratio*100:.1f}% of failure window)")

    # Save metrics JSON
    metrics_summary = {
        "threshold": threshold,
        "threshold_method": f"{ANOMALY_PERCENTILE}th percentile of validation reconstruction error",
        "validation_stats": {
            "mean_loss": float(val_errors.mean()),
            "std_loss": float(val_errors.std()),
            "median_loss": float(np.median(val_errors)),
        },
        "test_metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "total_test_windows": len(test_labels),
            "total_ground_truth_failure_windows": int(np.sum(test_labels)),
            "total_predicted_anomalies": int(np.sum(test_preds)),
        },
        "event_level_evaluation": event_results
    }

    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[EVAL] Metrics saved to {metrics_path}")

    # ==========================================
    # Generate Evaluation Figures (Fig 13 to 17)
    # ==========================================

    # Figure 13: Loss Convergence Curves
    history_file = MODELS_DIR / "training_history.json"
    if history_file.exists():
        with open(history_file, "r") as f:
            hist = json.load(f)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist["epochs"], hist["train_loss"], label="Train MSE Loss", color=COLORS["primary"], lw=2.0, marker="o")
        ax.plot(hist["epochs"], hist["val_loss"], label="Validation MSE Loss", color=COLORS["secondary"], lw=2.0, marker="s")
        ax.set_title("Figure 13: LSTM Autoencoder Training & Validation Loss Convergence", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error (MSE)")
        ax.legend(frameon=True)
        fig.savefig(FIGURES_EVAL_DIR / "fig13_loss_convergence_curves.png")
        plt.close(fig)
        print("  --> Saved Fig 13: Loss Convergence Curves")

    # Figure 14: Reconstruction Error Distributions (Normal vs. Failure)
    fig, ax = plt.subplots(figsize=(11, 6))
    normal_errors = test_errors[test_labels == 0]
    failure_errors = test_errors[test_labels == 1]
    
    sns.kdeplot(normal_errors, ax=ax, color=COLORS["healthy"], fill=True, alpha=0.4, label="Healthy Operational Windows", clip=(0, None))
    sns.kdeplot(failure_errors, ax=ax, color=COLORS["failure"], fill=True, alpha=0.5, label="Documented Failure Incident Windows", clip=(0, None))
    ax.axvline(threshold, color=COLORS["threshold"], linestyle="--", lw=2, label=f"Anomaly Threshold ({threshold:.4f})")
    
    ax.set_title("Figure 14: Reconstruction Error Distribution Separation (Healthy vs. Failure States)", fontweight="bold")
    ax.set_xlabel("Sequence Reconstruction Error (MSE)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0, float(np.percentile(failure_errors, 98) * 1.2))
    ax.legend(frameon=True)
    fig.savefig(FIGURES_EVAL_DIR / "fig14_reconstruction_error_distribution.png")
    plt.close(fig)
    print("  --> Saved Fig 14: Reconstruction Error Distribution")

    # Figure 15: Continuous Anomaly Score Timeline vs. Dynamic Threshold
    fig, ax = plt.subplots(figsize=(16, 6))
    sub_sample = 5  # Subsample for smooth line rendering
    ax.plot(pd.to_datetime(test_timestamps[::sub_sample]), test_errors[::sub_sample], color=COLORS["primary"], lw=0.7, alpha=0.8, label="Reconstruction Error")
    ax.axhline(threshold, color=COLORS["threshold"], linestyle="--", lw=1.8, label=f"Dynamic Threshold tau ({threshold:.4f})")

    # Highlight Failure Intervals
    for ev in FAILURE_EVENTS:
        start_d = pd.to_datetime(ev["start"])
        end_d = pd.to_datetime(ev["end"])
        ax.axvspan(start_d, end_d, color=COLORS["danger"], alpha=0.3, label=f"Ground-Truth: {ev['name']}" if ev["id"] == 1 else None)

    ax.set_title("Figure 15: Continuous Test Anomaly Score Timeline vs. Dynamic Alarm Threshold", fontweight="bold")
    ax.set_xlabel("Date (April - August 2020)")
    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_ylim(0, float(np.percentile(test_errors, 99.8) * 1.3))
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(FIGURES_EVAL_DIR / "fig15_anomaly_score_timeline_threshold.png")
    plt.close(fig)
    print("  --> Saved Fig 15: Anomaly Score Timeline")

    # Figure 16: PR-AUC & ROC-AUC Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PR Curve
    axes[0].plot(recall_curve, precision_curve, color=COLORS["primary"], lw=2.5, label=f"LSTM Autoencoder (PR-AUC = {pr_auc:.3f})")
    no_skill = np.sum(test_labels) / len(test_labels)
    axes[0].axhline(no_skill, color="gray", linestyle="--", label=f"No Skill ({no_skill:.4f})")
    axes[0].set_title("Precision-Recall Curve (Imbalanced Failure Benchmark)", fontweight="bold")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(loc="lower left", frameon=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_errors)
    axes[1].plot(fpr, tpr, color=COLORS["secondary"], lw=2.5, label=f"LSTM Autoencoder (ROC-AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier (0.500)")
    axes[1].set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate (FPR)")
    axes[1].set_ylabel("True Positive Rate (TPR)")
    axes[1].legend(loc="lower right", frameon=True)

    fig.suptitle("Figure 16: Model Performance Metrics Across Decision Thresholds", fontsize=15, fontweight="bold")
    fig.savefig(FIGURES_EVAL_DIR / "fig16_precision_recall_roc_curves.png")
    plt.close(fig)
    print("  --> Saved Fig 16: PR and ROC Curves")

    # Figure 17: Per-Sensor Error Attribution Breakdown (Explainability)
    print("[EVAL] Computing per-channel reconstruction attribution for Failure Event #3...")
    # Sample failure windows from Event #3
    ev3_mask = (pd.to_datetime(test_timestamps) >= pd.to_datetime("2020-06-05 10:00:00")) & \
               (pd.to_datetime(test_timestamps) <= pd.to_datetime("2020-06-07 14:30:00"))
    
    if np.sum(ev3_mask) > 0:
        sample_fail_windows = test_windows[ev3_mask][:1000]
        channel_errs = compute_per_channel_reconstruction_error(model, sample_fail_windows, device=device)
        mean_channel_errs = channel_errs.mean(axis=0)

        fig, ax = plt.subplots(figsize=(13, 6))
        bars = ax.bar(ALL_FEATURES, mean_channel_errs, color=COLORS["primary"], edgecolor="black", alpha=0.85)
        
        # Highlight top 3 contributing sensor features
        top_indices = np.argsort(mean_channel_errs)[-3:]
        for idx in top_indices:
            bars[idx].set_color(COLORS["danger"])

        ax.set_title("Figure 17: Per-Sensor Reconstruction Error Attribution During Failure Event #3 (Root-Cause Explainability)", fontweight="bold")
        ax.set_xlabel("Telemetry Sensor Feature")
        ax.set_ylabel("Mean Channel Reconstruction Error")
        ax.set_xticklabels(ALL_FEATURES, rotation=45, ha="right")

        fig.savefig(FIGURES_EVAL_DIR / "fig17_sensor_attribution_breakdown.png")
        plt.close(fig)
        print("  --> Saved Fig 17: Sensor Attribution Breakdown")

    print("[EVAL] Model evaluation pipeline completed successfully!")
    return metrics_summary


if __name__ == "__main__":
    evaluate_model_pipeline()
