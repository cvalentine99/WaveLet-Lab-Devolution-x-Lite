"""
GPU RF Forensics Engine - Power Spectral Density Estimation

High-performance Welch PSD estimator using GPU acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rf_forensics.core.gpu_compat import CUPY_AVAILABLE, cp, np

# Import FFT plan for optimized GPU FFT
try:
    from cupyx.scipy.fft import get_fft_plan
except ImportError:
    get_fft_plan = None

from rf_forensics.dsp.windows import WindowGenerator, get_correction_factors


@dataclass
class PSDResult:
    """Result of PSD computation."""

    frequencies: np.ndarray  # Frequency axis in Hz
    psd: np.ndarray  # Power spectral density
    psd_db: np.ndarray  # PSD in dB
    num_segments: int  # Number of segments averaged
    resolution_hz: float  # Frequency resolution


class WelchPSD:
    """
    GPU-accelerated Welch Power Spectral Density estimator.

    Implements the Welch method using batched matrix operations:
    1. Stride input into (K, M) matrix (K segments, M samples each)
    2. Apply window via broadcast multiplication
    3. Batched FFT across all segments
    4. Magnitude-squared and row averaging

    Features:
    - Optimized for GPU processing with CuPy
    - Support for 'density' (V²/Hz) and 'spectrum' (V²) scaling
    - GPU-side dB conversion
    - Rolling/streaming PSD updates
    - Spectrogram generation for waterfall displays
    """

    def __init__(
        self,
        fft_size: int = 1024,
        overlap: float = 0.5,
        window: str = "hann",
        sample_rate: float = 1e6,
        scaling: Literal["density", "spectrum"] = "density",
        detrend: bool = False,
    ):
        """
        Initialize Welch PSD estimator.

        Args:
            fft_size: FFT size (number of samples per segment).
            overlap: Overlap fraction (0.0 to 0.9).
            window: Window type.
            sample_rate: Sample rate in Hz.
            scaling: Output scaling ('density' for V²/Hz, 'spectrum' for V²).
            detrend: Whether to remove mean before FFT.
        """
        if fft_size & (fft_size - 1) != 0:
            raise ValueError("fft_size must be a power of 2")
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in [0, 1)")

        self._fft_size = fft_size
        self._overlap = overlap
        self._window_type = window
        self._sample_rate = sample_rate
        self._scaling = scaling
        self._detrend = detrend

        # Compute derived parameters
        self._hop_size = int(fft_size * (1 - overlap))
        self._window_gen = WindowGenerator()

        # Get window and correction factors
        self._window = self._window_gen.get_window(window, fft_size)
        self._corrections = get_correction_factors(window, fft_size)

        # Precompute frequency axis (GPU version for internal use)
        self._freq_axis = cp.fft.fftshift(cp.fft.fftfreq(fft_size, 1.0 / sample_rate))

        # Cache frequency axis on CPU (static - never changes after init)
        # This avoids GPU->CPU transfer every frame
        if CUPY_AVAILABLE:
            self._freq_axis_cpu = cp.asnumpy(self._freq_axis)
        else:
            self._freq_axis_cpu = self._freq_axis.copy()

        # Pre-allocate log floor constant to avoid temp allocation in log10
        self._log_floor = cp.float32(1e-12) if CUPY_AVAILABLE else np.float32(1e-12)

        # Scaling factor
        if scaling == "density":
            # V²/Hz scaling
            self._scale = 1.0 / (sample_rate * cp.sum(self._window**2))
        else:
            # V² scaling
            self._scale = 1.0 / (cp.sum(self._window) ** 2)

        # Rolling average state
        self._rolling_psd = None
        self._rolling_count = 0

        # Cached FFT plan for performance (reuse same plan for same-sized inputs)
        self._fft_plan = None
        self._fft_plan_shape = None

        # Cached segment indices - avoids 8.2 KB/frame allocation in _segment_signal()
        self._cached_indices = {}

    @property
    def fft_size(self) -> int:
        return self._fft_size

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @property
    def resolution_hz(self) -> float:
        """Frequency resolution in Hz."""
        return self._sample_rate / self._fft_size

    @property
    def freq_axis(self) -> cp.ndarray:
        """Frequency axis array (GPU)."""
        return self._freq_axis

    @property
    def freq_axis_cpu(self) -> np.ndarray:
        """
        Frequency axis array (CPU-cached).

        Use this when you need the frequency axis on CPU to avoid
        repeated GPU->CPU transfers. The array is computed once at
        initialization and cached.
        """
        return self._freq_axis_cpu

    @property
    def log_floor(self):
        """Pre-allocated log floor constant for dB conversion."""
        return self._log_floor

    def compute_psd(
        self, signal: cp.ndarray, return_segments: bool = False
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Compute power spectral density using Welch method.

        Args:
            signal: Input signal (1D complex array).
            return_segments: If True, return individual segment PSDs.

        Returns:
            Tuple of (frequencies, psd) arrays.
            If return_segments=True, psd has shape (num_segments, fft_size).
        """
        signal = cp.asarray(signal)
        n_samples = len(signal)

        # Calculate number of segments
        num_segments = max(1, (n_samples - self._fft_size) // self._hop_size + 1)

        if num_segments < 1:
            raise ValueError(
                f"Signal too short: {n_samples} samples, need at least {self._fft_size}"
            )

        # Create segment matrix using stride tricks
        segments = self._segment_signal(signal, num_segments)

        # Remove mean if requested
        if self._detrend:
            segments = segments - cp.mean(segments, axis=1, keepdims=True)

        # Apply window (broadcast multiplication)
        windowed = segments * self._window

        # Compute FFT for all segments with cached plan for performance
        if CUPY_AVAILABLE and get_fft_plan is not None:
            # Cache FFT plan for same-sized inputs
            if self._fft_plan is None or self._fft_plan_shape != windowed.shape:
                self._fft_plan = get_fft_plan(windowed, axes=(1,))
                self._fft_plan_shape = windowed.shape

            with self._fft_plan:
                fft_result = cp.fft.fft(windowed, axis=1)
        else:
            fft_result = cp.fft.fft(windowed, axis=1)

        # Shift to center DC
        fft_result = cp.fft.fftshift(fft_result, axes=1)

        # Compute magnitude squared
        psd_segments = cp.abs(fft_result) ** 2

        # Apply scaling
        psd_segments = psd_segments * float(self._scale)

        if return_segments:
            return self._freq_axis, psd_segments

        # Average across segments
        psd = cp.mean(psd_segments, axis=0)

        return self._freq_axis, psd

    def compute_psd_db(
        self, signal: cp.ndarray, ref_power: float = 1.0
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Compute PSD in decibels.

        Args:
            signal: Input signal.
            ref_power: Reference power for dB calculation.

        Returns:
            Tuple of (frequencies, psd_db) arrays.
        """
        freqs, psd = self.compute_psd(signal)

        # Convert to dB on GPU
        psd_db = 10 * cp.log10(psd / ref_power + 1e-12)

        return freqs, psd_db

    def compute_spectrogram(
        self, signal: cp.ndarray, time_segments: int | None = None
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """
        Compute spectrogram (time-frequency representation).

        Args:
            signal: Input signal.
            time_segments: Number of time segments (auto if None).

        Returns:
            Tuple of (times, frequencies, spectrogram) arrays.
            Spectrogram has shape (time_segments, fft_size).
        """
        signal = cp.asarray(signal)
        n_samples = len(signal)

        if time_segments is None:
            # Auto-calculate based on signal length
            time_segments = (n_samples - self._fft_size) // self._hop_size + 1

        # Segment the signal
        segments = self._segment_signal(signal, time_segments)

        # Apply window and compute FFT
        windowed = segments * self._window
        fft_result = cp.fft.fftshift(cp.fft.fft(windowed, axis=1), axes=1)

        # Compute PSD for each time segment
        spectrogram = cp.abs(fft_result) ** 2 * float(self._scale)

        # Time axis
        times = cp.arange(time_segments) * self._hop_size / self._sample_rate

        return times, self._freq_axis, spectrogram

    def compute_spectrogram_db(
        self, signal: cp.ndarray, time_segments: int | None = None, ref_power: float = 1.0
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Compute spectrogram in dB."""
        times, freqs, spec = self.compute_spectrogram(signal, time_segments)
        spec_db = 10 * cp.log10(spec / ref_power + 1e-12)
        return times, freqs, spec_db

    def update_rolling(
        self, new_samples: cp.ndarray, alpha: float = 0.1
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Update rolling PSD estimate with new samples.

        Uses exponential moving average for smooth updates.

        Args:
            new_samples: New signal samples.
            alpha: Smoothing factor (0-1, higher = more responsive).

        Returns:
            Tuple of (frequencies, rolling_psd) arrays.
        """
        _, new_psd = self.compute_psd(new_samples)

        if self._rolling_psd is None:
            self._rolling_psd = new_psd
            self._rolling_count = 1
        else:
            # Exponential moving average
            self._rolling_psd = alpha * new_psd + (1 - alpha) * self._rolling_psd
            self._rolling_count += 1

        return self._freq_axis, self._rolling_psd

    def get_rolling_psd_db(self, ref_power: float = 1.0) -> tuple[cp.ndarray, cp.ndarray]:
        """Get current rolling PSD in dB."""
        if self._rolling_psd is None:
            raise RuntimeError("No rolling PSD available. Call update_rolling first.")

        psd_db = 10 * cp.log10(self._rolling_psd / ref_power + 1e-12)
        return self._freq_axis, psd_db

    def reset_rolling(self) -> None:
        """Reset rolling PSD state."""
        self._rolling_psd = None
        self._rolling_count = 0

    def cleanup(self) -> None:
        """
        Release all GPU resources held by this PSD estimator.

        Should be called before discarding the object to free GPU memory.
        """
        self._freq_axis = None
        self._freq_axis_cpu = None
        self._window = None
        self._rolling_psd = None
        self._rolling_count = 0
        self._fft_plan = None
        self._fft_plan_shape = None
        self._log_floor = None
        self._cached_indices.clear()

    def _segment_signal(self, signal: cp.ndarray, num_segments: int) -> cp.ndarray:
        """
        Segment signal into overlapping chunks.

        Uses cached index arrays to avoid per-frame allocation (~8.2 KB/frame).
        """
        sig_len = len(signal)
        cache_key = (sig_len, num_segments)

        if cache_key not in self._cached_indices:
            # Compute indices once and cache
            starts = cp.arange(num_segments) * self._hop_size
            indices = starts[:, None] + cp.arange(self._fft_size)
            self._cached_indices[cache_key] = cp.clip(indices, 0, sig_len - 1)

        # Gather segments using cached indices
        return signal[self._cached_indices[cache_key]]

    def to_numpy(self, *arrays) -> tuple[np.ndarray, ...]:
        """Convert CuPy arrays to NumPy."""
        if CUPY_AVAILABLE:
            return tuple(cp.asnumpy(arr) for arr in arrays)
        return arrays

    def get_result(self, signal: cp.ndarray) -> PSDResult:
        """
        Compute PSD and return structured result.

        Args:
            signal: Input signal.

        Returns:
            PSDResult with frequencies, PSD, and metadata.
        """
        freqs, psd = self.compute_psd(signal)
        psd_db = 10 * cp.log10(psd + 1e-12)

        # Convert to NumPy for result
        freqs_np, psd_np, psd_db_np = self.to_numpy(freqs, psd, psd_db)

        num_segments = max(1, (len(signal) - self._fft_size) // self._hop_size + 1)

        return PSDResult(
            frequencies=freqs_np,
            psd=psd_np,
            psd_db=psd_db_np,
            num_segments=num_segments,
            resolution_hz=self.resolution_hz,
        )


class SpectrogramBuffer:
    """
    Circular buffer for streaming spectrogram updates.

    Maintains a fixed-size time-frequency history for waterfall displays.

    Performance optimizations:
    - Pre-allocated reorder buffer eliminates cp.roll() allocation in hot path
    - update() returns current row view instead of full spectrogram
    - get_latest_psd_db() provides zero-allocation access to newest data
    """

    def __init__(
        self,
        num_time_bins: int = 256,
        fft_size: int = 1024,
        sample_rate: float = 1e6,
        window: str = "hann",
        overlap: float = 0.5,
    ):
        """
        Initialize spectrogram buffer.

        Args:
            num_time_bins: Number of time bins to maintain.
            fft_size: FFT size for spectral analysis.
            sample_rate: Sample rate in Hz.
            window: Window function type.
            overlap: Overlap fraction.
        """
        self._num_time_bins = num_time_bins
        self._fft_size = fft_size
        self._psd = WelchPSD(fft_size, overlap, window, sample_rate)

        # Initialize buffer
        self._buffer = cp.zeros((num_time_bins, fft_size), dtype=cp.float32)
        self._write_idx = 0
        self._filled = False

        # Pre-allocated reorder buffer - eliminates 1 MB/frame cp.roll() allocation
        self._reorder_buffer = cp.zeros((num_time_bins, fft_size), dtype=cp.float32)

    def update(self, signal: cp.ndarray) -> cp.ndarray:
        """
        Update buffer with new signal data.

        Args:
            signal: New signal samples.

        Returns:
            Current PSD row (view into buffer, no allocation).
            Use get_spectrogram() if full time-ordered spectrogram is needed.
        """
        # Compute PSD for new data
        _, psd = self._psd.compute_psd(signal)

        # Convert to dB
        psd_db = 10 * cp.log10(psd + 1e-12)

        # Write to circular buffer
        self._buffer[self._write_idx] = psd_db

        # Save index of latest row before incrementing
        latest_idx = self._write_idx

        self._write_idx = (self._write_idx + 1) % self._num_time_bins

        if self._write_idx == 0:
            self._filled = True

        # Return view of current row (zero allocation)
        return self._buffer[latest_idx]

    def get_latest_psd_db(self) -> cp.ndarray:
        """
        Get the most recent PSD (dB) without any allocation.

        Returns:
            View of the latest PSD row in the buffer.
        """
        # Most recent write is at (_write_idx - 1) % num_time_bins
        latest_idx = (self._write_idx - 1) % self._num_time_bins
        return self._buffer[latest_idx]

    def update_from_psd_db(self, psd_db: cp.ndarray) -> cp.ndarray:
        """
        Update buffer with pre-computed PSD (in dB).

        This is more efficient than update() when PSD is already computed
        elsewhere (avoids duplicate PSD computation).

        Args:
            psd_db: Pre-computed PSD in dB scale (1D array).

        Returns:
            View of the current PSD row (zero allocation).
        """
        # Write to circular buffer
        self._buffer[self._write_idx] = psd_db

        # Save index of latest row before incrementing
        latest_idx = self._write_idx

        self._write_idx = (self._write_idx + 1) % self._num_time_bins

        if self._write_idx == 0:
            self._filled = True

        # Return view of current row (zero allocation)
        return self._buffer[latest_idx]

    def get_spectrogram(self) -> cp.ndarray:
        """
        Get current spectrogram with correct time ordering.

        Uses pre-allocated reorder buffer - zero allocation in hot path.

        Returns:
            2D array with shape (num_time_bins, fft_size).
        """
        if not self._filled:
            return self._buffer[: self._write_idx]

        # Reorder to correct time sequence using pre-allocated buffer
        # Copy tail (oldest data) to beginning of reorder buffer
        tail_size = self._num_time_bins - self._write_idx
        self._reorder_buffer[:tail_size] = self._buffer[self._write_idx :]
        # Copy head (newest data) to end of reorder buffer
        self._reorder_buffer[tail_size:] = self._buffer[: self._write_idx]

        return self._reorder_buffer

    def get_spectrogram_uint8(self, min_db: float = -120, max_db: float = -20) -> cp.ndarray:
        """
        Get spectrogram quantized to uint8 for visualization.

        Args:
            min_db: Minimum dB value (maps to 0).
            max_db: Maximum dB value (maps to 255).

        Returns:
            uint8 array suitable for texture upload.
        """
        spec = self.get_spectrogram()

        # Clip and normalize
        spec_clipped = cp.clip(spec, min_db, max_db)
        spec_norm = (spec_clipped - min_db) / (max_db - min_db)

        return (spec_norm * 255).astype(cp.uint8)

    @property
    def freq_axis(self) -> cp.ndarray:
        return self._psd.freq_axis

    @property
    def resolution_hz(self) -> float:
        return self._psd.resolution_hz

    def cleanup(self) -> None:
        """
        Release all GPU resources held by this spectrogram buffer.

        Should be called before discarding the object to free GPU memory.
        """
        self._buffer = None
        self._reorder_buffer = None
        if self._psd:
            self._psd.cleanup()
            self._psd = None
        self._write_idx = 0
        self._filled = False
