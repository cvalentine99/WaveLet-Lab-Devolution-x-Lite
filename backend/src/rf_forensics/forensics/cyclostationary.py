"""
GPU RF Forensics Engine - Cyclostationary Analysis

FFT Accumulation Method (FAM) for Spectral Correlation Density estimation.
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

from rf_forensics.dsp.windows import get_window


@dataclass
class CyclicFeature:
    """A detected cyclic feature."""

    cyclic_freq_hz: float
    spectral_freq_hz: float
    magnitude: float
    phase: float
    snr_db: float


@dataclass
class SCDResult:
    """Result of Spectral Correlation Density computation."""

    scd: np.ndarray  # 2D SCD matrix
    frequency_axis: np.ndarray  # Spectral frequency axis
    cyclic_axis: np.ndarray  # Cyclic frequency axis
    sample_rate: float
    resolution_hz: float


class FAMAnalyzer:
    """
    FFT Accumulation Method for Spectral Correlation Density estimation.

    The FAM algorithm computes the Spectral Correlation Density (SCD) which
    reveals cyclostationary features of modulated signals at their symbol rate.

    Algorithm steps:
    1. Windowed Channelization: Divide signal into blocks, apply window, FFT
    2. Product Formation: X(t,f) × X*(t,f-α) for cyclic frequencies α
    3. FFT Accumulation: Second FFT along time axis for smoothing

    The resulting bifrequency plane shows:
    - Vertical axis: Spectral frequency
    - Horizontal axis: Cyclic frequency (related to symbol rate)
    """

    def __init__(
        self, fft_size: int = 256, num_blocks: int = 256, window: str = "hann", overlap: float = 0.5
    ):
        """
        Initialize FAM analyzer.

        Args:
            fft_size: FFT size for channelization (frequency resolution).
            num_blocks: Number of time blocks (cyclic resolution).
            window: Window function type.
            overlap: Overlap fraction between blocks.
        """
        self._fft_size = fft_size
        self._num_blocks = num_blocks
        self._window_type = window
        self._overlap = overlap

        # Get window
        self._window = get_window(window, fft_size)

        # Calculate derived parameters
        self._hop_size = int(fft_size * (1 - overlap))

    def compute_scd(
        self,
        signal: cp.ndarray,
        sample_rate: float,
        alpha_range: tuple[float, float] | None = None,
        alpha_resolution: float | None = None,
    ) -> SCDResult:
        """
        Compute Spectral Correlation Density using FAM.

        Args:
            signal: Input signal (complex).
            sample_rate: Sample rate in Hz.
            alpha_range: Tuple of (min, max) cyclic frequencies to compute.
                        If None, computes full range.
            alpha_resolution: Resolution for cyclic frequency axis.

        Returns:
            SCDResult with SCD matrix and axes.
        """
        signal = cp.asarray(signal)
        n_samples = len(signal)

        # Calculate required signal length
        required_length = (self._num_blocks - 1) * self._hop_size + self._fft_size
        if n_samples < required_length:
            # Pad signal
            signal = cp.pad(signal, (0, required_length - n_samples))

        # Step 1: Windowed Channelization
        # Create time-frequency matrix X(t, f)
        channelized = self._channelize(signal)

        # Step 2 & 3: Compute SCD for cyclic frequencies
        if alpha_range is None:
            alpha_range = (-sample_rate / 2, sample_rate / 2)

        if alpha_resolution is None:
            alpha_resolution = sample_rate / (self._num_blocks * self._hop_size)

        # Generate cyclic frequency axis
        alpha_min, alpha_max = alpha_range
        num_alphas = int((alpha_max - alpha_min) / alpha_resolution) + 1
        alpha_axis = cp.linspace(alpha_min, alpha_max, num_alphas)

        # Frequency axis
        freq_axis = cp.fft.fftshift(cp.fft.fftfreq(self._fft_size, 1 / sample_rate))

        # Compute SCD
        scd = self._compute_scd_matrix(channelized, alpha_axis, sample_rate)

        # Convert to NumPy
        scd_np = cp.asnumpy(scd) if CUPY_AVAILABLE else scd
        freq_np = cp.asnumpy(freq_axis) if CUPY_AVAILABLE else freq_axis
        alpha_np = cp.asnumpy(alpha_axis) if CUPY_AVAILABLE else alpha_axis

        return SCDResult(
            scd=scd_np,
            frequency_axis=freq_np,
            cyclic_axis=alpha_np,
            sample_rate=sample_rate,
            resolution_hz=alpha_resolution,
        )

    def _channelize(self, signal: cp.ndarray) -> cp.ndarray:
        """
        Channelize signal into time-frequency matrix.

        Returns:
            Matrix of shape (num_blocks, fft_size) containing X(t, f).
        """
        # Create index array for gathering blocks
        starts = cp.arange(self._num_blocks) * self._hop_size
        indices = starts[:, None] + cp.arange(self._fft_size)

        # Gather blocks
        blocks = signal[indices]

        # Apply window
        windowed = blocks * self._window

        # FFT each block
        channelized = cp.fft.fftshift(cp.fft.fft(windowed, axis=1), axes=1)

        return channelized

    def _compute_scd_matrix(
        self, channelized: cp.ndarray, alpha_axis: cp.ndarray, sample_rate: float
    ) -> cp.ndarray:
        """
        Compute SCD matrix for given cyclic frequencies.

        Args:
            channelized: Time-frequency matrix X(t, f).
            alpha_axis: Cyclic frequencies to compute.
            sample_rate: Sample rate in Hz.

        Returns:
            SCD matrix of shape (len(alpha_axis), fft_size).
        """
        num_alphas = len(alpha_axis)
        scd = cp.zeros((num_alphas, self._fft_size), dtype=cp.complex64)

        freq_resolution = sample_rate / self._fft_size

        for i, alpha in enumerate(alpha_axis):
            # Frequency shift in bins
            shift_bins = int(alpha / freq_resolution)

            if abs(shift_bins) >= self._fft_size // 2:
                continue

            # Product formation: X(t, f) × X*(t, f - α)
            # Shift channelized matrix
            shifted = cp.roll(channelized, shift_bins, axis=1)

            # Conjugate product
            product = channelized * cp.conj(shifted)

            # FFT accumulation (along time axis)
            # This averages/smooths the spectral correlation
            scd_row = cp.mean(product, axis=0)
            scd[i] = scd_row

        return scd

    def find_cyclic_features(
        self,
        scd_result: SCDResult,
        threshold_db: float = 10.0,
        exclude_dc: bool = True,
        window: int = 2,
    ) -> list[CyclicFeature]:
        """
        Find dominant cyclic features in SCD (vectorized).

        Uses vectorized local maximum detection to eliminate nested Python loops.

        Args:
            scd_result: Result from compute_scd.
            threshold_db: Detection threshold above noise floor.
            exclude_dc: Whether to exclude features near DC.
            window: Window size for local maximum detection.

        Returns:
            List of detected cyclic features.
        """
        scd = scd_result.scd
        freq_axis = scd_result.frequency_axis
        alpha_axis = scd_result.cyclic_axis

        # Compute magnitude and dB (vectorized)
        scd_mag = np.abs(scd)
        scd_db = 20 * np.log10(scd_mag + 1e-12)

        # Estimate noise floor
        noise_floor = np.median(scd_db)
        threshold = noise_floor + threshold_db

        # Vectorized threshold mask
        above_threshold = scd_db > threshold

        # Vectorized DC exclusion mask
        if exclude_dc:
            dc_mask = np.abs(alpha_axis) >= scd_result.resolution_hz * 2
            # Broadcast to 2D: only keep rows where alpha is not near DC
            above_threshold = above_threshold & dc_mask[:, np.newaxis]

        # Vectorized local maximum detection using max pooling approach
        local_max_mask = self._find_local_maxima_2d(scd_db, window)

        # Combine masks: above threshold AND local maximum
        peak_mask = above_threshold & local_max_mask

        # Extract peak indices (vectorized)
        peak_indices = np.argwhere(peak_mask)

        if len(peak_indices) == 0:
            return []

        # Extract all values at once (batch extraction)
        i_indices = peak_indices[:, 0]
        j_indices = peak_indices[:, 1]

        # Batch extract all feature values
        cyclic_freqs = alpha_axis[i_indices]
        spectral_freqs = freq_axis[j_indices]
        magnitudes = scd_mag[i_indices, j_indices]
        phases = np.angle(scd[i_indices, j_indices])
        snr_dbs = scd_db[i_indices, j_indices] - noise_floor

        # Create features from batched arrays
        features = [
            CyclicFeature(
                cyclic_freq_hz=float(cyclic_freqs[k]),
                spectral_freq_hz=float(spectral_freqs[k]),
                magnitude=float(magnitudes[k]),
                phase=float(phases[k]),
                snr_db=float(snr_dbs[k]),
            )
            for k in range(len(peak_indices))
        ]

        # Sort by magnitude
        features.sort(key=lambda x: x.magnitude, reverse=True)

        return features

    def _find_local_maxima_2d(self, data: np.ndarray, window: int = 2) -> np.ndarray:
        """
        Find local maxima in 2D array using vectorized max pooling.

        Args:
            data: 2D input array.
            window: Half-window size for local maximum detection.

        Returns:
            Boolean mask where True indicates local maximum.
        """
        from scipy.ndimage import maximum_filter

        # Use scipy's maximum_filter for efficient local max detection
        # This is much faster than nested loops
        footprint_size = 2 * window + 1
        local_max = maximum_filter(data, size=footprint_size, mode="constant", cval=-np.inf)

        # A point is a local max if it equals the local maximum
        return data >= local_max

    def estimate_symbol_rate(
        self, signal: cp.ndarray, sample_rate: float, min_rate: float = 1e3, max_rate: float = 1e6
    ) -> float:
        """
        Estimate symbol rate from cyclic features.

        Args:
            signal: Input signal.
            sample_rate: Sample rate in Hz.
            min_rate: Minimum expected symbol rate.
            max_rate: Maximum expected symbol rate.

        Returns:
            Estimated symbol rate in Hz.
        """
        # Compute SCD with targeted alpha range
        result = self.compute_scd(
            signal, sample_rate, alpha_range=(min_rate, max_rate), alpha_resolution=min_rate / 10
        )

        # Find cyclic features
        features = self.find_cyclic_features(result, threshold_db=6.0)

        if not features:
            return 0.0

        # The strongest cyclic feature typically corresponds to symbol rate
        # (or a harmonic of it)
        return abs(features[0].cyclic_freq_hz)


class SSCAAnalyzer:
    """
    Strip Spectral Correlation Analyzer.

    More efficient than FAM for targeting specific cyclic frequencies.
    """

    def __init__(self, fft_size: int = 1024):
        self._fft_size = fft_size

    def compute_sca(
        self, signal: cp.ndarray, sample_rate: float, target_alpha: float
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Compute Spectral Correlation at specific cyclic frequency.

        Args:
            signal: Input signal.
            sample_rate: Sample rate in Hz.
            target_alpha: Target cyclic frequency.

        Returns:
            Tuple of (frequencies, correlation) arrays.
        """
        signal = cp.asarray(signal)

        # Frequency shift for target alpha
        t = cp.arange(len(signal)) / sample_rate
        shifter = cp.exp(-1j * cp.pi * target_alpha * t)

        # Create shifted signal
        signal_shifted = signal * shifter

        # Compute cross-spectrum
        n_segments = len(signal) // self._fft_size
        correlation = cp.zeros(self._fft_size, dtype=cp.complex64)

        for i in range(n_segments):
            start = i * self._fft_size
            end = start + self._fft_size

            seg1 = signal[start:end]
            seg2 = signal_shifted[start:end]

            fft1 = cp.fft.fft(seg1)
            fft2 = cp.fft.fft(seg2)

            correlation += fft1 * cp.conj(fft2)

        correlation /= n_segments

        freqs = cp.fft.fftshift(cp.fft.fftfreq(self._fft_size, 1 / sample_rate))
        correlation = cp.fft.fftshift(correlation)

        return freqs, correlation
