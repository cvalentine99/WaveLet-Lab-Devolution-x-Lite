"""
GPU RF Forensics Engine - ML Module

Machine learning components: clustering, feature extraction, anomaly detection.
"""

from rf_forensics.ml.clustering import ClusterInfo, EmitterClusterer
from rf_forensics.ml.features import FeatureExtractor, FeatureNormalizer

# Anomaly detection (optional - graceful fallback)
try:
    from rf_forensics.ml.anomaly import (
        AnomalyConfig,
        AnomalyDetector,
        SignalAutoencoder,
        create_anomaly_detector,
    )

    ANOMALY_AVAILABLE = True
except ImportError:
    ANOMALY_AVAILABLE = False
    AnomalyDetector = None
    AnomalyConfig = None
    SignalAutoencoder = None
    create_anomaly_detector = None

__all__ = [
    "EmitterClusterer",
    "ClusterInfo",
    "FeatureExtractor",
    "FeatureNormalizer",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyConfig",
    "SignalAutoencoder",
    "create_anomaly_detector",
    "ANOMALY_AVAILABLE",
]
