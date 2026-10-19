"""
Training runner script for the Deep LSTM Autoencoder.
Loads processed training and validation splits, generates sliding window tensors,
and executes the training loop with checkpointing.
"""

import sys
import numpy as np
import pandas as pd
import torch

from src.config import (
    PROCESSED_DATA_DIR,
    ALL_FEATURES,
    WINDOW_SIZE,
    STRIDE_TRAIN,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
)
from src.preprocessing import create_sliding_windows
from src.trainer import train_lstm_autoencoder


def run_training():
    print("[RUNNER] Loading processed training and validation datasets...")
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Processed datasets not found. Run preprocessing first.")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    scaled_cols = [f"{c}_scaled" for c in ALL_FEATURES]
    train_scaled = df_train[scaled_cols].values
    val_scaled = df_val[scaled_cols].values

    print(f"[RUNNER] Train shape: {train_scaled.shape} | Val shape: {val_scaled.shape}")
    print(f"[RUNNER] Generating sliding window tensors (W={WINDOW_SIZE}, stride={STRIDE_TRAIN})...")

    train_windows = create_sliding_windows(train_scaled, window_size=WINDOW_SIZE, stride=STRIDE_TRAIN)
    val_windows = create_sliding_windows(val_scaled, window_size=WINDOW_SIZE, stride=STRIDE_TRAIN)

    print(f"[RUNNER] Created {len(train_windows):,} training windows and {len(val_windows):,} validation windows.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RUNNER] Hardware device: {device}")

    model, history = train_lstm_autoencoder(
        train_windows=train_windows,
        val_windows=val_windows,
        input_dim=len(ALL_FEATURES),
        seq_len=WINDOW_SIZE,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        device=device
    )

    print("[RUNNER] Model training completed successfully!")
    return model, history


if __name__ == "__main__":
    run_training()
