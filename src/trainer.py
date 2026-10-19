"""
Model training and validation pipeline for the LSTM Autoencoder.
Implements early stopping, learning rate scheduling, validation loss tracking,
and saves the best model checkpoint and training history.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config import (
    MODELS_DIR,
    WINDOW_SIZE,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    LATENT_DIM,
    ENCODER_DIMS,
    DECODER_DIMS,
    DROPOUT_RATE,
    RANDOM_SEED,
)
from src.model import LSTMAutoencoder


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_lstm_autoencoder(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    input_dim: int = 15,
    seq_len: int = WINDOW_SIZE,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    device: str = None
) -> Tuple[LSTMAutoencoder, Dict[str, List[float]]]:
    """
    Trains the LSTM Autoencoder strictly on healthy baseline data.
    """
    set_seed()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] Using computation device: {device}")

    # Build DataLoaders
    train_tensor = torch.tensor(train_windows, dtype=torch.float32)
    val_tensor = torch.tensor(val_windows, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[TRAIN] Training samples: {len(train_tensor):,} windows | Validation: {len(val_tensor):,} windows")

    # Initialize Model
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dims=(ENCODER_DIMS[0], ENCODER_DIMS[1]),
        latent_dim=LATENT_DIM,
        seq_len=seq_len,
        dropout=DROPOUT_RATE
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "epochs": []}
    best_val_loss = float("inf")
    patience_counter = 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_lstm_autoencoder.pt"

    start_time = time.time()
    print("[TRAIN] Starting LSTM Autoencoder training loop...")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item() * len(x)

        epoch_train_loss = running_train_loss / len(train_tensor)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                reconstructed = model(x)
                loss = criterion(reconstructed, x)
                running_val_loss += loss.item() * len(x)

        epoch_val_loss = running_val_loss / len(val_tensor)
        scheduler.step(epoch_val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        print(f"[Epoch {epoch:02d}/{epochs:02d}] Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}", flush=True)

        # Checkpoint Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint (Val MSE: {best_val_loss:.6f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[TRAIN] Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).", flush=True)
                break

    total_time = time.time() - start_time
    print(f"[TRAIN] Training completed in {total_time:.2f} seconds. Best Val MSE: {best_val_loss:.6f}", flush=True)

    # Load best checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Save history
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[TRAIN] Training history saved to {history_path}")

    return model, history
