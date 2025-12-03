"""
GPU RF Forensics Engine - Feature Extraction

Feature extraction pipeline for signal clustering and classification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from rf_forensics.detection.peaks import Detection
from rf_forensics.forensics.cumulants import CumulantCalculator
from rf_forensics.forensics.symbol_rate import SymbolRateEstimator


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    include_spectral: bool = True
    include_temporal: bool = True
    include_modulation: bool = True
    include_cyclostationary: bool = False  # Computationally expensive


@dataclass
class FeatureVector:
    """Extracted feature vector for a signal."""

    values: np.ndarray
    names: list[str]
    detection_id: int = 0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {name: float(val) for name, val in zip(self.names, self.values)}


class FeatureExtractor:
    """
    Feature extraction pipeline for RF signal analysis.

    Extracts features in multiple categories:
    - Spectral: center frequency, bandwidth, peak power, spectral shape
    - Temporal: burst duration, duty cycle, repetition rate
    - Modulation: cumulant values, symbol rate, constellation spread
    - Cyclostationary: dominant cyclic frequencies (optional)
    """

    def __init__(self, feature_config: FeatureConfig | None = None, sample_rate: float = 1e6):
        """
        Initialize feature extractor.

        Args:
            feature_config: Configuration for which features to extract.
            sample_rate: Sample rate in Hz.
        """
        self._config = feature_config or FeatureConfig()
        self._sample_rate = sample_rate

        # Initialize sub-components
        self._cumulant_calc = CumulantCalculator()
        self._symbol_estimator = SymbolRateEstimator()

        # Feature names
        self._feature_names = self._build_feature_names()

    def _build_feature_names(self) -> list[str]:
        """Build list of feature names based on config."""
        names = []

        if self._config.include_spectral:
            names.extend(
                [
                    "center_freq_normalized",
                    "bandwidth_normalized",
                    "peak_power_db",
                    "snr_db",
                    "spectral_flatness",
                    "spectral_centroid",
                    "spectral_spread",
                ]
            )

        if self._config.include_temporal:
            names.extend(
                [
                    "burst_duration_ms",
                    "duty_cycle",
                    "repetition_rate_hz",
                ]
            )

        if self._config.include_modulation:
            names.extend(
                [
                    "c40_real",
                    "c40_imag",
                    "c42_real",
                    "c42_imag",
                    "symbol_rate_normalized",
                    "kurtosis",
                ]
            )

        if self._config.include_cyclostationary:
            names.extend(
                [
                    "dominant_cyclic_freq",
                    "cyclic_magnitude",
                ]
            )

        return names

    def extract(
        self, detection: Detection, signal: cp.ndarray, psd: cp.ndarray | None = None
    ) -> FeatureVector:
        """
        Extract features for a single detection.

        Args:
            detection: Detection metadata.
            signal: Complex signal segment.
            psd: Optional PSD array.

        Returns:
            FeatureVector with extracted features.
        """
        signal = cp.asarray(signal)
        features = []

        # Spectral features
        if self._config.include_spectral:
            spectral = self._extract_spectral(detection, signal, psd)
            features.extend(spectral)

        # Temporal features
        if self._config.include_temporal:
            temporal = self._extract_temporal(detection, signal)
            features.extend(temporal)

        # Modulation features
        if self._config.include_modulation:
            modulation = self._extract_modulation(signal)
            features.extend(modulation)

        # Cyclostationary features
        if self._config.include_cyclostationary:
            cyclo = self._extract_cyclostationary(signal)
            features.extend(cyclo)

        return FeatureVector(
            values=np.array(features, dtype=np.float32),
            names=self._feature_names,
            detection_id=detection.detection_id,
        )

    def _extract_spectral(
        self, detection: Detection, signal: cp.ndarray, psd: cp.ndarray | None
    ) -> list[float]:
        """Extract spectral features."""
        features = []

        # Normalized center frequency
        features.append(detection.center_freq_hz / self._sample_rate)

        # Normalized bandwidth
        features.append(detection.bandwidth_hz / self._sample_rate)

        # Peak power and SNR
        features.append(detection.peak_power_db)
        features.append(detection.snr_db)

        # Spectral flatness and shape
        if psd is not None:
            psd = cp.asarray(psd)
            psd_linear = 10 ** (psd / 10)

            # Spectral flatness (geometric mean / arithmetic mean)
            geo_mean = float(cp.exp(cp.mean(cp.log(psd_linear + 1e-12))))
            arith_mean = float(cp.mean(psd_linear))
            flatness = geo_mean / (arith_mean + 1e-12)

            # Spectral centroid (normalized)
            freqs = cp.arange(len(psd)) / len(psd)
            centroid = float(cp.sum(freqs * psd_linear) / (cp.sum(psd_linear) + 1e-12))

            # Spectral spread
            spread = float(
                cp.sqrt(cp.sum((freqs - centroid) ** 2 * psd_linear) / (cp.sum(psd_linear) + 1e-12))
            )

            features.extend([flatness, centroid, spread])
        else:
            features.extend([0.5, 0.5, 0.1])  # Default values

        return features

    def _extract_temporal(self, detection: Detection, signal: cp.ndarray) -> list[float]:
        """Extract temporal features."""
        features = []

        # Burst duration (estimate from signal length)
        duration_ms = len(signal) / self._sample_rate * 1000
        features.append(duration_ms)

        # Duty cycle (estimate from signal energy)
        energy = cp.abs(signal) ** 2
        threshold = float(cp.mean(energy)) * 0.5
        above_threshold = cp.sum(energy > threshold) / len(energy)
        features.append(float(above_threshold))

        # Repetition rate (placeholder - would need longer observation)
        features.append(0.0)

        return features

    def _extract_modulation(self, signal: cp.ndarray) -> list[float]:
        """Extract modulation-related features."""
        features = []

        # Cumulants
        cumulants = self._cumulant_calc.compute_cumulants(signal)
        features.extend(
            [
                cumulants.c40.real,
                cumulants.c40.imag,
                cumulants.c42.real,
                cumulants.c42.imag,
            ]
        )

        # Symbol rate (normalized)
        result = self._symbol_estimator.estimate(signal, self._sample_rate)
        features.append(result.rate_hz / self._sample_rate)

        # Kurtosis
        signal = cp.asarray(signal)
        normalized = signal / (cp.std(signal) + 1e-12)
        kurtosis = float(
            cp.mean(cp.abs(normalized) ** 4) / (cp.mean(cp.abs(normalized) ** 2) ** 2 + 1e-12)
        )
        features.append(kurtosis)

        return features

    def _extract_cyclostationary(self, signal: cp.ndarray) -> list[float]:
        """Extract cyclostationary features (expensive)."""
        # Placeholder - full implementation would use FAMAnalyzer
        return [0.0, 0.0]

    def batch_extract(
        self,
        detections: list[Detection],
        signals: list[cp.ndarray],
        psds: list[cp.ndarray] | None = None,
    ) -> cp.ndarray:
        """
        Extract features for multiple detections.

        Args:
            detections: List of detections.
            signals: List of signal segments.
            psds: Optional list of PSDs.

        Returns:
            2D array of shape (num_detections, num_features).
        """
        num_features = len(self._feature_names)
        features = np.zeros((len(detections), num_features), dtype=np.float32)

        for i, (det, sig) in enumerate(zip(detections, signals)):
            psd = psds[i] if psds else None
            fv = self.extract(det, sig, psd)
            features[i] = fv.values

        return cp.asarray(features) if CUPY_AVAILABLE else features

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def num_features(self) -> int:
        return len(self._feature_names)


class FeatureNormalizer:
    """
    Feature normalization using StandardScaler-like approach.

    Normalizes features to zero mean and unit variance.
    """

    def __init__(self):
        self._mean = None
        self._std = None
        self._fitted = False

    def fit(self, features: cp.ndarray) -> None:
        """
        Fit normalizer to feature data.

        Args:
            features: 2D array of shape (num_samples, num_features).
        """
        features = cp.asarray(features)
        self._mean = cp.mean(features, axis=0)
        self._std = cp.std(features, axis=0)
        self._std = cp.where(self._std < 1e-8, 1.0, self._std)  # Avoid division by zero
        self._fitted = True

    def transform(self, features: cp.ndarray) -> cp.ndarray:
        """
        Transform features using fitted parameters.

        Args:
            features: 2D array to transform.

        Returns:
            Normalized features.
        """
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        features = cp.asarray(features)
        return (features - self._mean) / self._std

    def fit_transform(self, features: cp.ndarray) -> cp.ndarray:
        """Fit and transform in one call."""
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, features: cp.ndarray) -> cp.ndarray:
        """Reverse the normalization."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted.")

        features = cp.asarray(features)
        return features * self._std + self._mean

    @property
    def is_fitted(self) -> bool:
        return self._fitted
