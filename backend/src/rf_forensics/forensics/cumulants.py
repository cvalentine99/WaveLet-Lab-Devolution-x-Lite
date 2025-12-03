"""
GPU RF Forensics Engine - Higher-Order Cumulants

Cumulant-based modulation classification using GPU acceleration.
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


@dataclass
class CumulantVector:
    """Higher-order cumulant values for a signal."""

    c20: complex  # Second-order cumulant
    c21: complex  # Second-order cumulant
    c40: complex  # Fourth-order cumulant
    c41: complex  # Fourth-order cumulant
    c42: complex  # Fourth-order cumulant


@dataclass
class ModulationClassification:
    """Result of modulation classification."""

    modulation_type: str
    confidence: float
    cumulants: CumulantVector
    all_distances: dict[str, float]


class CumulantCalculator:
    """
    GPU-accelerated higher-order cumulant calculator.

    Cumulants are statistical measures that can distinguish between
    different modulation types. For modulated signals, the cumulants
    have specific theoretical values that can be used for classification.

    Key relationships:
    - M_pq = E[x^p × (x*)^q]  (moments)
    - C_20 = M_20
    - C_21 = M_21
    - C_40 = M_40 - 3×M_20²
    - C_42 = M_42 - |M_20|² - 2×M_21²

    Theoretical cumulant values:
    - BPSK:   C_40 = -2,    C_42 = -2
    - QPSK:   C_40 = 1,     C_42 = -1
    - 8PSK:   C_40 = 0,     C_42 = 0
    - 16QAM:  C_40 = -0.68, C_42 = -0.68
    - 64QAM:  C_40 = -0.62, C_42 = -0.62
    """

    # Theoretical cumulant values for common modulations
    THEORETICAL_CUMULANTS = {
        "BPSK": {"c40": -2.0, "c42": -2.0},
        "QPSK": {"c40": 1.0, "c42": -1.0},
        "8PSK": {"c40": 0.0, "c42": 0.0},
        "16QAM": {"c40": -0.68, "c42": -0.68},
        "64QAM": {"c40": -0.619, "c42": -0.619},
        "256QAM": {"c40": -0.605, "c42": -0.605},
        "OOK": {"c40": 0.5, "c42": 0.5},
        "FSK": {"c40": 0.0, "c42": 0.0},  # Approximate
    }

    def __init__(self, normalize: bool = True):
        """
        Initialize cumulant calculator.

        Args:
            normalize: Whether to normalize signal to unit energy.
        """
        self._normalize = normalize

    def compute_cumulants(self, signal: cp.ndarray) -> CumulantVector:
        """
        Compute higher-order cumulants for a signal.

        Args:
            signal: Complex input signal.

        Returns:
            CumulantVector with computed cumulants.
        """
        signal = cp.asarray(signal)

        # Normalize signal
        if self._normalize:
            signal = self._normalize_signal(signal)

        # Compute moments M_pq = E[x^p × (x*)^q]
        m20 = self._moment(signal, 2, 0)
        m21 = self._moment(signal, 2, 1)
        m40 = self._moment(signal, 4, 0)
        m41 = self._moment(signal, 4, 1)
        m42 = self._moment(signal, 4, 2)

        # Compute cumulants from moments
        c20 = m20
        c21 = m21
        c40 = m40 - 3 * m20**2
        c41 = m41 - 3 * m20 * m21
        c42 = m42 - abs(m20) ** 2 - 2 * m21**2

        # Convert to Python complex
        def to_complex(x):
            if CUPY_AVAILABLE:
                x = cp.asnumpy(x)
            return complex(x)

        return CumulantVector(
            c20=to_complex(c20),
            c21=to_complex(c21),
            c40=to_complex(c40),
            c41=to_complex(c41),
            c42=to_complex(c42),
        )

    def _normalize_signal(self, signal: cp.ndarray) -> cp.ndarray:
        """Normalize signal to zero mean and unit energy."""
        # Remove mean
        signal = signal - cp.mean(signal)

        # Normalize to unit energy
        energy = cp.sqrt(cp.mean(cp.abs(signal) ** 2))
        if energy > 0:
            signal = signal / energy

        return signal

    def _moment(self, signal: cp.ndarray, p: int, q: int) -> cp.ndarray:
        """
        Compute moment M_pq = E[x^p × (x*)^q].

        Args:
            signal: Complex signal.
            p: Power of x.
            q: Power of x*.

        Returns:
            Complex moment value.
        """
        return cp.mean(signal**p * cp.conj(signal) ** q)

    def classify_modulation(
        self, cumulants: CumulantVector, snr_db: float | None = None
    ) -> ModulationClassification:
        """
        Classify modulation type based on cumulants.

        Args:
            cumulants: Computed cumulant vector.
            snr_db: Optional SNR estimate for confidence adjustment.

        Returns:
            ModulationClassification with type and confidence.
        """
        # Calculate distance to each theoretical modulation
        distances = {}

        for mod_type, theoretical in self.THEORETICAL_CUMULANTS.items():
            c40_dist = abs(cumulants.c40.real - theoretical["c40"])
            c42_dist = abs(cumulants.c42.real - theoretical["c42"])

            # Euclidean distance in cumulant space
            distance = np.sqrt(c40_dist**2 + c42_dist**2)
            distances[mod_type] = distance

        # Find closest modulation
        best_mod = min(distances, key=distances.get)
        best_distance = distances[best_mod]

        # Calculate confidence (inverse of distance, normalized)
        # Apply SNR-based adjustment if available
        if best_distance < 0.001:
            confidence = 1.0
        else:
            confidence = 1.0 / (1.0 + best_distance)

        if snr_db is not None and snr_db < 10:
            # Reduce confidence for low SNR
            confidence *= snr_db / 10
            confidence = max(0.1, min(1.0, confidence))

        return ModulationClassification(
            modulation_type=best_mod,
            confidence=confidence,
            cumulants=cumulants,
            all_distances=distances,
        )

    def compute_and_classify(
        self, signal: cp.ndarray, snr_db: float | None = None
    ) -> ModulationClassification:
        """
        Compute cumulants and classify modulation in one call.

        Args:
            signal: Complex input signal.
            snr_db: Optional SNR estimate.

        Returns:
            ModulationClassification result.
        """
        cumulants = self.compute_cumulants(signal)
        return self.classify_modulation(cumulants, snr_db)


class KurtosisClassifier:
    """
    Simple kurtosis-based classifier for quick modulation identification.

    Kurtosis (normalized fourth moment) provides a fast preliminary
    classification without full cumulant computation.
    """

    # Theoretical kurtosis values
    THEORETICAL_KURTOSIS = {
        "BPSK": 1.0,
        "QPSK": 1.0,
        "8PSK": 1.0,
        "16QAM": 1.32,
        "64QAM": 1.38,
        "Gaussian": 2.0,
    }

    def compute_kurtosis(self, signal: cp.ndarray) -> float:
        """
        Compute normalized kurtosis.

        Args:
            signal: Complex input signal.

        Returns:
            Kurtosis value.
        """
        signal = cp.asarray(signal)

        # Normalize
        signal = signal - cp.mean(signal)
        variance = cp.mean(cp.abs(signal) ** 2)

        if variance < 1e-12:
            return 0.0

        signal = signal / cp.sqrt(variance)

        # Fourth moment / variance^2
        m4 = cp.mean(cp.abs(signal) ** 4)
        m2 = cp.mean(cp.abs(signal) ** 2)

        kurtosis = m4 / (m2**2)

        if CUPY_AVAILABLE:
            kurtosis = float(cp.asnumpy(kurtosis))

        return float(kurtosis)

    def classify(self, signal: cp.ndarray) -> tuple[str, float]:
        """
        Classify based on kurtosis.

        Args:
            signal: Complex input signal.

        Returns:
            Tuple of (modulation_type, confidence).
        """
        kurtosis = self.compute_kurtosis(signal)

        # Find closest match
        best_match = None
        best_distance = float("inf")

        for mod_type, theoretical in self.THEORETICAL_KURTOSIS.items():
            distance = abs(kurtosis - theoretical)
            if distance < best_distance:
                best_distance = distance
                best_match = mod_type

        confidence = 1.0 / (1.0 + best_distance)

        return best_match, confidence
