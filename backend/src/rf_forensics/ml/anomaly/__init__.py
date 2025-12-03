"""
RF Signal Anomaly Detection Module

GPU-accelerated autoencoder-based anomaly detection for RF forensics.
Detects unusual signals that don't match learned patterns.

Per BACKEND_CONTRACT.md:
- detection.anomaly_score: 0.0-1.0
- Values > 0.5 indicate anomalous signals
"""

from .autoencoder import IQAutoencoder, SignalAutoencoder
from .inference import (
    AnomalyConfig,
    AnomalyDetector,
    FallbackAnomalyDetector,
    create_anomaly_detector,
)

__all__ = [
    "SignalAutoencoder",
    "IQAutoencoder",
    "AnomalyConfig",
    "AnomalyDetector",
    "FallbackAnomalyDetector",
    "create_anomaly_detector",
]
