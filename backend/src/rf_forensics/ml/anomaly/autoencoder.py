"""
RF Signal Anomaly Detection - Autoencoder Model

Convolutional autoencoder for detecting anomalous RF signals based on
reconstruction error. Signals that don't fit learned patterns have higher
reconstruction error, yielding higher anomaly scores.

Per BACKEND_CONTRACT.md, detection.anomaly_score should be 0.0-1.0 where >0.5 = anomalous.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SignalAutoencoder(nn.Module):
    """
    Autoencoder for RF signal anomaly detection.

    Input: Detection feature vector (normalized)
    Output: Reconstruction + latent representation

    Architecture optimized for real-time inference on GPU.
    Uses LayerNorm for stable training with varying input distributions.
    """

    def __init__(self, input_dim: int = 12, latent_dim: int = 4):
        """
        Initialize autoencoder.

        Args:
            input_dim: Dimension of input feature vector.
            latent_dim: Dimension of latent (compressed) representation.
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: compress detection features to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, latent_dim),
        )

        # Decoder: reconstruct from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LayerNorm(16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(32, input_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (reconstructed, latent) tensors.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output space."""
        return self.decoder(latent)

    def compute_anomaly_score(
        self, x: torch.Tensor, threshold: float = 0.1, scale: float = 10.0
    ) -> torch.Tensor:
        """
        Compute anomaly score as normalized reconstruction error.

        Uses sigmoid to map MSE to 0-1 range. Default parameters tuned so:
        - MSE ~0.1 -> score ~0.5 (borderline)
        - MSE <0.05 -> score <0.3 (normal)
        - MSE >0.2 -> score >0.7 (anomalous)

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            threshold: MSE threshold for score=0.5 (default 0.1).
            scale: Scaling factor for sigmoid steepness (default 10.0).

        Returns:
            Anomaly scores tensor of shape (batch_size,), values in [0, 1].
        """
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            # Per-sample MSE
            mse = torch.mean((x - reconstructed) ** 2, dim=-1)
            # Sigmoid normalization to [0, 1]
            score = torch.sigmoid((mse - threshold) * scale)
        return score

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute raw reconstruction error (MSE) without normalization.

        Useful for threshold tuning and analysis.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            MSE tensor of shape (batch_size,).
        """
        with torch.no_grad():
            reconstructed, _ = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=-1)
        return mse


class IQAutoencoder(nn.Module):
    """
    Convolutional autoencoder for raw IQ signal anomaly detection.

    For use when IQ segments are available. More powerful than feature-based
    but requires more computation.
    """

    def __init__(self, segment_len: int = 1024, latent_dim: int = 32):
        """
        Initialize IQ autoencoder.

        Args:
            segment_len: Length of IQ segment (number of complex samples).
            latent_dim: Dimension of latent representation.
        """
        super().__init__()

        self.segment_len = segment_len
        self.latent_dim = latent_dim

        # Input: (batch, 2, segment_len) - real and imaginary as channels
        # Encoder with 1D convolutions
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )

        # Calculate flattened size
        self._flat_size = 64 * (segment_len // 8)

        # Latent space projection
        self.fc_encode = nn.Linear(self._flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self._flat_size)

        # Decoder with transposed convolutions
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, segment_len // 8)),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 2, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Complex IQ tensor of shape (batch, segment_len) or
               real tensor of shape (batch, 2, segment_len).

        Returns:
            Tuple of (reconstructed, latent).
        """
        # Handle complex input
        if x.is_complex():
            x = torch.stack([x.real, x.imag], dim=1)

        # Encode
        encoded = self.encoder(x)
        latent = self.fc_encode(encoded)

        # Decode
        decoded = self.fc_decode(latent)
        reconstructed = self.decoder(decoded)

        return reconstructed, latent

    def compute_anomaly_score(
        self, x: torch.Tensor, threshold: float = 0.05, scale: float = 20.0
    ) -> torch.Tensor:
        """Compute anomaly score from IQ reconstruction error."""
        with torch.no_grad():
            if x.is_complex():
                x_real = torch.stack([x.real, x.imag], dim=1)
            else:
                x_real = x

            reconstructed, _ = self.forward(x_real)
            mse = torch.mean((x_real - reconstructed) ** 2, dim=(1, 2))
            score = torch.sigmoid((mse - threshold) * scale)
        return score
