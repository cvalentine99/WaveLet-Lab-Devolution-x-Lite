"""
GPU RF Forensics Engine - Symbol Rate Estimation

Multiple methods for estimating symbol rate from received signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np


@dataclass
class SymbolRateResult:
    """Result of symbol rate estimation."""

    rate_hz: float
    confidence: float
    method_used: str
    all_estimates: dict[str, float]
    spectral_peak_snr_db: float = 0.0


class SymbolRateEstimator:
    """
    GPU-accelerated symbol rate estimation using multiple methods.

    Methods:
    1. Squaring Loop: |x(t)|² → FFT → peak detection
    2. Fourth-power: x(t)^4 → FFT (good for PSK)
    3. Cyclic Feature: Use FAM output
    4. Combined: Weighted voting of all methods
    """

    def __init__(
        self,
        method: Literal["squaring", "fourth_power", "combined"] = "combined",
        fft_size: int = 8192,
        min_rate_hz: float = 1e3,
        max_rate_hz: float = 10e6,
    ):
        """
        Initialize symbol rate estimator.

        Args:
            method: Estimation method to use.
            fft_size: FFT size for spectral analysis.
            min_rate_hz: Minimum expected symbol rate.
            max_rate_hz: Maximum expected symbol rate.
        """
        self._method = method
        self._fft_size = fft_size
        self._min_rate = min_rate_hz
        self._max_rate = max_rate_hz

    def estimate(self, signal: cp.ndarray, sample_rate: float) -> SymbolRateResult:
        """
        Estimate symbol rate from signal.

        Args:
            signal: Complex input signal.
            sample_rate: Sample rate in Hz.

        Returns:
            SymbolRateResult with estimated rate and confidence.
        """
        signal = cp.asarray(signal, copy=False)

        # Run selected method(s)
        estimates = {}
        confidences = {}

        if self._method in ["squaring", "combined"]:
            rate, conf, snr = self._squaring_method(signal, sample_rate)
            estimates["squaring"] = rate
            confidences["squaring"] = conf

        if self._method in ["fourth_power", "combined"]:
            rate, conf, snr = self._fourth_power_method(signal, sample_rate)
            estimates["fourth_power"] = rate
            confidences["fourth_power"] = conf

        # Combine estimates
        if self._method == "combined":
            rate, confidence = self._combine_estimates(estimates, confidences)
            method_used = "combined"
        else:
            rate = estimates[self._method]
            confidence = confidences[self._method]
            method_used = self._method

        return SymbolRateResult(
            rate_hz=rate,
            confidence=confidence,
            method_used=method_used,
            all_estimates=estimates,
            spectral_peak_snr_db=snr if "snr" in dir() else 0.0,
        )

    def _squaring_method(
        self, signal: cp.ndarray, sample_rate: float
    ) -> tuple[float, float, float]:
        """
        Squaring loop method: |x(t)|² → FFT → peak

        The squared magnitude of a BPSK/QPSK signal has a strong
        spectral component at the symbol rate.
        """
        # Compute squared magnitude
        squared = cp.abs(signal) ** 2

        # Remove DC
        squared = squared - cp.mean(squared)

        # High-resolution FFT
        n_fft = max(self._fft_size, len(squared))
        spectrum = cp.abs(cp.fft.fft(squared, n_fft))

        # Frequency axis
        freqs = cp.fft.fftfreq(n_fft, 1 / sample_rate)

        # Only search positive frequencies in valid range
        mask = (freqs >= self._min_rate) & (freqs <= self._max_rate)
        valid_spectrum = spectrum.copy()
        valid_spectrum[~mask] = 0

        # Find peak
        peak_idx = int(cp.argmax(valid_spectrum))
        peak_freq = float(freqs[peak_idx])
        peak_mag = float(spectrum[peak_idx])

        # Estimate SNR
        noise_floor = float(cp.median(spectrum[mask]))
        snr_db = 10 * np.log10(peak_mag / (noise_floor + 1e-12))

        # Confidence based on SNR
        confidence = min(1.0, max(0.0, (snr_db - 3) / 20))

        return abs(peak_freq), confidence, snr_db

    def _fourth_power_method(
        self, signal: cp.ndarray, sample_rate: float
    ) -> tuple[float, float, float]:
        """
        Fourth-power method: x(t)^4 → FFT → peak / 4

        For PSK signals, raising to the 4th power removes modulation
        and creates a tone at 4× the carrier offset, with harmonics
        at the symbol rate.
        """
        # Fourth power
        fourth = signal**4

        # FFT
        n_fft = max(self._fft_size, len(fourth))
        spectrum = cp.abs(cp.fft.fft(fourth, n_fft))

        # Frequency axis (divide by 4 for actual rate)
        freqs = cp.fft.fftfreq(n_fft, 1 / sample_rate)

        # Valid range (multiply by 4 since we divide later)
        mask = (cp.abs(freqs) >= self._min_rate * 4) & (cp.abs(freqs) <= self._max_rate * 4)
        valid_spectrum = spectrum.copy()
        valid_spectrum[~mask] = 0

        # Find peak
        peak_idx = int(cp.argmax(valid_spectrum))
        peak_freq = float(freqs[peak_idx]) / 4  # Divide by 4
        peak_mag = float(spectrum[peak_idx])

        # SNR estimate
        noise_floor = float(cp.median(spectrum[mask]))
        snr_db = 10 * np.log10(peak_mag / (noise_floor + 1e-12))

        confidence = min(1.0, max(0.0, (snr_db - 3) / 20))

        return abs(peak_freq), confidence, snr_db

    def _combine_estimates(
        self, estimates: dict[str, float], confidences: dict[str, float]
    ) -> tuple[float, float]:
        """Combine multiple estimates using weighted average."""
        total_weight = 0.0
        weighted_sum = 0.0

        for method, rate in estimates.items():
            weight = confidences.get(method, 0.5)

            # Check if estimates are consistent
            if rate > 0:
                weighted_sum += rate * weight
                total_weight += weight

        if total_weight > 0:
            combined_rate = weighted_sum / total_weight

            # Confidence increases if methods agree
            rate_variance = np.var(list(estimates.values()))
            mean_rate = np.mean([r for r in estimates.values() if r > 0])

            if mean_rate > 0:
                agreement = 1.0 - min(1.0, rate_variance / (mean_rate**2))
                combined_confidence = np.mean(list(confidences.values())) * agreement
            else:
                combined_confidence = 0.0
        else:
            combined_rate = 0.0
            combined_confidence = 0.0

        return combined_rate, combined_confidence


class TimingErrorDetector:
    """
    Timing error detectors for symbol rate tracking/refinement.

    These can be used after initial estimation to track symbol timing
    in real-time or refine the estimate.
    """

    def __init__(self, samples_per_symbol: int):
        """
        Initialize timing error detector.

        Args:
            samples_per_symbol: Expected samples per symbol.
        """
        self._sps = samples_per_symbol
        self._mu = 0.0  # Fractional timing offset

    def gardner_ted(self, samples: cp.ndarray) -> float:
        """
        Gardner Timing Error Detector (vectorized).

        Non-data-aided TED that works well for BPSK/QPSK.
        Uses: e(n) = Re{[y(nT) - y((n-1)T)] × y*((n-0.5)T)}

        Vectorized implementation eliminates GPU-CPU sync per iteration.
        """
        samples = cp.asarray(samples, copy=False)
        n_samples = len(samples)
        sps = self._sps

        # Calculate number of complete symbols we can process
        # Need: current symbol (n*sps), previous symbol ((n-1)*sps), midpoint (n*sps - sps//2)
        # For n starting at 1, max index is n*sps where n = n_symbols
        n_symbols = n_samples // sps - 1

        if n_symbols < 1:
            return 0.0

        # Vectorized indexing - extract all samples at once
        # Current symbol samples: indices [sps, 2*sps, 3*sps, ..., n_symbols*sps]
        # Previous symbol samples: indices [0, sps, 2*sps, ..., (n_symbols-1)*sps]
        # Midpoint samples: indices [sps - sps//2, 2*sps - sps//2, ...]

        # Create index arrays (GPU-side)
        n_range = cp.arange(1, n_symbols + 1)

        y_n_indices = n_range * sps
        y_n1_indices = (n_range - 1) * sps
        y_mid_indices = n_range * sps - sps // 2

        # Bounds check - ensure all indices are valid
        valid_mask = (y_n_indices < n_samples) & (y_mid_indices < n_samples) & (y_mid_indices >= 0)
        if not cp.any(valid_mask):
            return 0.0

        # Extract samples using vectorized indexing
        y_n = samples[y_n_indices[valid_mask]]
        y_n1 = samples[y_n1_indices[valid_mask]]
        y_mid = samples[y_mid_indices[valid_mask]]

        # Vectorized Gardner error computation (all on GPU)
        errors = ((y_n - y_n1) * cp.conj(y_mid)).real

        # Single GPU-to-CPU transfer at the end
        error_sum = float(cp.sum(errors))
        n_valid = int(cp.sum(valid_mask))

        return error_sum / n_valid if n_valid > 0 else 0.0

    def mm_ted(self, samples: cp.ndarray, decisions: cp.ndarray) -> float:
        """
        Mueller and Müller Timing Error Detector (vectorized).

        Decision-directed TED that requires symbol decisions.
        Uses: e(n) = Re{d*(n)×y(n-1) - d*(n-1)×y(n)}

        Vectorized implementation eliminates GPU-CPU sync per iteration.
        """
        samples = cp.asarray(samples, copy=False)
        decisions = cp.asarray(decisions, copy=False)
        sps = self._sps

        # Number of symbols we can process
        n_sample_symbols = len(samples) // sps
        n_decision_symbols = len(decisions)
        n_symbols = min(n_sample_symbols, n_decision_symbols) - 1

        if n_symbols < 1:
            return 0.0

        # Vectorized indexing for samples
        n_range = cp.arange(1, n_symbols + 1)

        y_n_indices = n_range * sps
        y_n1_indices = (n_range - 1) * sps

        # Bounds check
        valid_mask = y_n_indices < len(samples)
        if not cp.any(valid_mask):
            return 0.0

        # Extract samples and decisions using vectorized indexing
        y_n = samples[y_n_indices[valid_mask]]
        y_n1 = samples[y_n1_indices[valid_mask]]

        # Decision indices are just n_range and n_range-1 (already 0-indexed)
        d_n = decisions[n_range[valid_mask]]
        d_n1 = decisions[(n_range - 1)[valid_mask]]

        # Vectorized MM error computation (all on GPU)
        errors = (cp.conj(d_n) * y_n1 - cp.conj(d_n1) * y_n).real

        # Single GPU-to-CPU transfer at the end
        error_sum = float(cp.sum(errors))
        n_valid = int(cp.sum(valid_mask))

        return error_sum / n_valid if n_valid > 0 else 0.0


def estimate_symbol_rate(signal: cp.ndarray, sample_rate: float, method: str = "combined") -> float:
    """
    Convenience function for symbol rate estimation.

    Args:
        signal: Complex input signal.
        sample_rate: Sample rate in Hz.
        method: Estimation method.

    Returns:
        Estimated symbol rate in Hz.
    """
    estimator = SymbolRateEstimator(method=method)
    result = estimator.estimate(signal, sample_rate)
    return result.rate_hz
