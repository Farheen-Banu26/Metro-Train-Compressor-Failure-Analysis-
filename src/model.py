"""
Deep LSTM Autoencoder neural network architecture in PyTorch.
Learns a compact latent representation of normal APU multivariate temporal dynamics.
Used for unsupervised/semi-supervised temporal anomaly detection in predictive maintenance.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMEncoder(nn.Module):
    """
    Encoder network: compresses multivariate time series windows into a low-dimensional bottleneck.
    """
    def __init__(
        self,
        input_dim: int = 15,
        hidden_dims: Tuple[int, int] = (64, 32),
        latent_dim: int = 16,
        dropout: float = 0.2
    ):
        super(LSTMEncoder, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            batch_first=True,
            num_layers=1
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[1],
            batch_first=True,
            num_layers=1
        )
        self.fc_latent = nn.Linear(hidden_dims[1], latent_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, (hn, _) = self.lstm2(out)
        # Use last hidden state for bottleneck
        latent = self.act(self.fc_latent(hn[-1]))  # (batch_size, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    """
    Decoder network: reconstructs the original multivariate time series window from the latent vector.
    """
    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dims: Tuple[int, int] = (32, 64),
        output_dim: int = 15,
        seq_len: int = 60,
        dropout: float = 0.2
    ):
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.fc_expand = nn.Linear(latent_dim, hidden_dims[0])
        self.lstm1 = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[0],
            batch_first=True,
            num_layers=1
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dims[0],
            hidden_size=hidden_dims[1],
            batch_first=True,
            num_layers=1
        )
        self.fc_out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Expand latent vector: (batch_size, hidden_dims[0])
        expanded = self.fc_expand(latent)
        # Repeat across sequence length: (batch_size, seq_len, hidden_dims[0])
        rep = expanded.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm1(rep)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        reconstruction = self.fc_out(out)  # (batch_size, seq_len, output_dim)
        return reconstruction


class LSTMAutoencoder(nn.Module):
    """
    Complete Deep LSTM Autoencoder for time-series anomaly detection.
    """
    def __init__(
        self,
        input_dim: int = 15,
        hidden_dims: Tuple[int, int] = (64, 32),
        latent_dim: int = 16,
        seq_len: int = 60,
        dropout: float = 0.2
    ):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout
        )
        self.decoder = LSTMDecoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=input_dim,
            seq_len=seq_len,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes Mean Squared Error per sequence window.
        Returns tensor of shape (batch_size,)
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            # MSE across sequence length and feature dimensions
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error
