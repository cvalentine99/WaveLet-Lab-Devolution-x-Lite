"""
Anomaly Detection Inference Engine

GPU-accelerated inference wrapper for anomaly scoring of RF detections.
Integrates with the pipeline orchestrator to score each detection.

Features:
- Batch inference for efficiency
- Online learning for adaptation to new RF environments
- Automatic normalization stat tracking
- Graceful fallback when model unavailable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .autoencoder import SignalAutoencoder

if TYPE_CHECKING:
    from detection.peaks import Detection

logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""

    model_path: str = "models/anomaly_ae.pt"
    device: str = "cuda"
    input_dim: int = 12
    latent_dim: int = 4

    # Anomaly scoring thresholds
    score_threshold: float = 0.1  # MSE for score=0.5
    score_scale: float = 10.0  # Sigmoid steepness

    # Online learning
    online_learning: bool = False
    learning_rate: float = 1e-4
    update_interval: int = 100  # Update after N samples

    # Feature normalization
    normalize_features: bool = True
    warmup_samples: int = 50  # Samples before reliable scoring


class AnomalyDetector:
    """
    Inference wrapper for autoencoder-based anomaly detection.

    Scores detections based on reconstruction error - signals that don't
    match learned patterns have higher anomaly scores.
    """

    def __init__(self, config: AnomalyConfig | None = None):
        """
        Initialize anomaly detector.

        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or AnomalyConfig()

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, anomaly detection disabled")
            self._model = None
            self._device = None
            return

        # Set device
        if self.config.device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        logger.info(f"Anomaly detector using device: {self._device}")

        # Initialize model
        self._model = SignalAutoencoder(
            input_dim=self.config.input_dim, latent_dim=self.config.latent_dim
        ).to(self._device)
        self._model.eval()

        # Try to load pretrained weights
        self._load_model()

        # Feature normalization statistics (updated online)
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._sample_count = 0

        # Online learning state
        if self.config.online_learning:
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.config.learning_rate
            )
            self._update_buffer: list[np.ndarray] = []

    def _load_model(self) -> bool:
        """
        Load pretrained model weights if available.

        Returns:
            True if weights loaded, False if using random init.
        """
        model_path = Path(self.config.model_path)

        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self._device)
                self._model.load_state_dict(state_dict)
                logger.info(f"Loaded anomaly model from {model_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load anomaly model: {e}")

        logger.info("Using randomly initialized anomaly model (will learn online)")
        return False

    def save_model(self, path: str | None = None) -> None:
        """Save model weights to disk."""
        if self._model is None:
            return

        save_path = Path(path or self.config.model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), save_path)
        logger.info(f"Saved anomaly model to {save_path}")

    @property
    def available(self) -> bool:
        """Check if anomaly detection is available."""
        return self._model is not None

    def score_detection(self, detection: Detection) -> float:
        """
        Compute anomaly score for a single detection.

        Args:
            detection: Detection object to score.

        Returns:
            Anomaly score in [0, 1], or 0.0 if model unavailable.
        """
        if not self.available:
            return 0.0

        features = self._extract_features(detection)
        features_norm = self._normalize(features)

        tensor = torch.tensor(features_norm, dtype=torch.float32, device=self._device).unsqueeze(0)

        score = self._model.compute_anomaly_score(
            tensor, threshold=self.config.score_threshold, scale=self.config.score_scale
        )

        return float(score.item())

    def score_batch(self, detections: list[Detection]) -> list[float]:
        """
        Compute anomaly scores for a batch of detections.

        More efficient than calling score_detection repeatedly.

        Args:
            detections: List of Detection objects.

        Returns:
            List of anomaly scores in [0, 1].
        """
        if not self.available or not detections:
            return [0.0] * len(detections)

        # Extract and normalize features
        features = np.array([self._extract_features(d) for d in detections])

        # Update normalization stats
        self._update_stats(features)

        # Check if we have enough samples for reliable scoring
        if self._sample_count < self.config.warmup_samples:
            # During warmup, return moderate scores
            return [0.3] * len(detections)

        features_norm = self._normalize(features)

        # Run inference
        tensor = torch.tensor(features_norm, dtype=torch.float32, device=self._device)

        scores = self._model.compute_anomaly_score(
            tensor, threshold=self.config.score_threshold, scale=self.config.score_scale
        )

        scores_list = scores.cpu().tolist()

        # Online learning update if enabled
        if self.config.online_learning:
            self._buffer_for_update(features_norm, scores_list)

        return scores_list

    def _extract_features(self, det: Detection) -> np.ndarray:
        """
        Extract feature vector from detection for anomaly scoring.

        Feature vector is designed to capture signal characteristics
        that might indicate anomalous behavior.

        Args:
            det: Detection object.

        Returns:
            Feature array of shape (input_dim,).
        """
        # Normalize features to roughly similar scales
        features = np.array(
            [
                # Frequency features (normalized to GHz range)
                det.center_freq_hz / 1e9,
                # Bandwidth features (normalized to MHz)
                det.bandwidth_hz / 1e6,
                # Shape ratios (dimensionless, ~0-2 range)
                det.bandwidth_3db_hz / max(det.bandwidth_hz, 1.0),
                det.bandwidth_6db_hz / max(det.bandwidth_hz, 1.0),
                # Power features (normalized from typical dBm range)
                (det.peak_power_db + 120) / 100,  # -120 to -20 dBm -> 0 to 1
                # SNR (normalized, typical 0-60 dB range)
                det.snr_db / 60,
                # Classification confidence (already 0-1)
                det.modulation_confidence,
                # Temporal features
                getattr(det, "duty_cycle", 0.5) or 0.5,
                # Cluster context (normalized)
                (getattr(det, "cluster_id", -1) + 1) / 100
                if getattr(det, "cluster_id", -1) >= 0
                else 0,
                # Track history
                min(getattr(det, "frames_seen", 1) or 1, 100) / 100,
                # Symbol rate (normalized to Msym/s)
                (getattr(det, "symbol_rate", 0) or 0) / 1e6,
                # Reserved for future features
                0.0,
            ],
            dtype=np.float32,
        )

        return features

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using running statistics.

        Args:
            features: Raw features, shape (input_dim,) or (batch, input_dim).

        Returns:
            Normalized features with zero mean, unit variance.
        """
        if not self.config.normalize_features:
            return features

        if self._feature_mean is None or self._feature_std is None:
            # No stats yet, return as-is
            return features

        # Z-score normalization
        std_safe = np.maximum(self._feature_std, 1e-6)  # Avoid division by zero
        return (features - self._feature_mean) / std_safe

    def _update_stats(self, features: np.ndarray) -> None:
        """
        Update running mean and std for feature normalization.

        Uses Welford's online algorithm for numerical stability.

        Args:
            features: Batch of features, shape (batch, input_dim).
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        batch_size = len(features)

        if self._feature_mean is None:
            # Initialize
            self._feature_mean = np.mean(features, axis=0)
            self._feature_std = np.std(features, axis=0) + 1e-6
            self._sample_count = batch_size
        else:
            # Incremental update
            for feat in features:
                self._sample_count += 1
                delta = feat - self._feature_mean
                self._feature_mean += delta / self._sample_count

                # Update variance (Welford's algorithm)
                if self._sample_count > 1:
                    delta2 = feat - self._feature_mean
                    # Running M2 would be needed for exact variance
                    # Using EMA approximation for simplicity
                    alpha = 0.01  # Smoothing factor
                    self._feature_std = (1 - alpha) * self._feature_std + alpha * np.abs(delta)

    def _buffer_for_update(self, features_norm: np.ndarray, scores: list[float]) -> None:
        """
        Buffer samples for online learning update.

        Only buffers low-anomaly samples to avoid learning anomalies.

        Args:
            features_norm: Normalized features.
            scores: Corresponding anomaly scores.
        """
        # Only learn from normal samples (low anomaly score)
        for feat, score in zip(features_norm, scores):
            if score < 0.3:  # Only use clearly normal samples
                self._update_buffer.append(feat)

        # Update model when buffer is full
        if len(self._update_buffer) >= self.config.update_interval:
            self._online_update()

    def _online_update(self) -> None:
        """Perform online learning update on buffered samples."""
        if not self._update_buffer or self._model is None:
            return

        # Convert buffer to tensor
        batch = torch.tensor(
            np.array(self._update_buffer), dtype=torch.float32, device=self._device
        )

        # Single update step
        self._model.train()
        self._optimizer.zero_grad()

        reconstructed, _ = self._model(batch)
        loss = torch.nn.functional.mse_loss(reconstructed, batch)

        loss.backward()
        self._optimizer.step()

        self._model.eval()

        logger.debug(f"Online update: loss={loss.item():.4f}, samples={len(batch)}")

        # Clear buffer
        self._update_buffer = []

    def get_stats(self) -> dict:
        """Get current detector statistics."""
        return {
            "available": self.available,
            "device": str(self._device) if self._device else None,
            "sample_count": self._sample_count,
            "warmup_complete": self._sample_count >= self.config.warmup_samples,
            "feature_mean": self._feature_mean.tolist() if self._feature_mean is not None else None,
            "online_buffer_size": len(self._update_buffer)
            if hasattr(self, "_update_buffer")
            else 0,
        }


class FallbackAnomalyDetector:
    """
    Statistical fallback for anomaly detection when PyTorch unavailable.

    Uses simple statistical methods (z-score, IQR) instead of neural network.
    """

    def __init__(self, z_threshold: float = 3.0):
        """
        Initialize fallback detector.

        Args:
            z_threshold: Z-score threshold for anomaly (default 3.0 = 3 sigma).
        """
        self.z_threshold = z_threshold
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._sample_count = 0
        self._warmup = 50

    @property
    def available(self) -> bool:
        return True

    def score_batch(self, detections: list[Detection]) -> list[float]:
        """Score detections using statistical method."""
        if not detections:
            return []

        # Extract key features
        features = np.array(
            [
                [
                    d.center_freq_hz / 1e9,
                    d.bandwidth_hz / 1e6,
                    d.snr_db,
                    d.peak_power_db,
                ]
                for d in detections
            ]
        )

        # Update stats
        self._update_stats(features)

        if self._sample_count < self._warmup:
            return [0.3] * len(detections)

        # Compute z-scores
        std_safe = np.maximum(self._feature_std, 1e-6)
        z_scores = np.abs((features - self._feature_mean) / std_safe)

        # Max z-score across features
        max_z = np.max(z_scores, axis=1)

        # Convert to 0-1 score (sigmoid on z-score)
        scores = 1 / (1 + np.exp(-(max_z - self.z_threshold)))

        return scores.tolist()

    def score_detection(self, detection: Detection) -> float:
        """Score single detection."""
        return self.score_batch([detection])[0]

    def _update_stats(self, features: np.ndarray) -> None:
        """Update running statistics."""
        if self._feature_mean is None:
            self._feature_mean = np.mean(features, axis=0)
            self._feature_std = np.std(features, axis=0) + 1e-6
            self._sample_count = len(features)
        else:
            # EMA update
            alpha = 0.1
            batch_mean = np.mean(features, axis=0)
            batch_std = np.std(features, axis=0)
            self._feature_mean = (1 - alpha) * self._feature_mean + alpha * batch_mean
            self._feature_std = (1 - alpha) * self._feature_std + alpha * batch_std
            self._sample_count += len(features)


def create_anomaly_detector(
    prefer_neural: bool = True, config: AnomalyConfig | None = None
) -> AnomalyDetector:
    """
    Factory function to create appropriate anomaly detector.

    Args:
        prefer_neural: Prefer neural network if available.
        config: Configuration for neural detector.

    Returns:
        AnomalyDetector or FallbackAnomalyDetector instance.
    """
    if prefer_neural and TORCH_AVAILABLE:
        detector = AnomalyDetector(config)
        if detector.available:
            return detector

    logger.info("Using statistical fallback for anomaly detection")
    return FallbackAnomalyDetector()
