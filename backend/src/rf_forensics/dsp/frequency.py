"""
GPU RF Forensics Engine - Frequency Processing

GPU-accelerated frequency shifting, decimation, filtering, and channelization.
"""

from __future__ import annotations

from dataclasses import dataclass

from rf_forensics.core.gpu_compat import cp


@dataclass
class FilterCoefficients:
    """Filter coefficients for GPU-based filtering."""

    b: cp.ndarray  # Numerator coefficients
    a: cp.ndarray  # Denominator coefficients (FIR if a=[1])
    order: int
    cutoff_hz: float
    sample_rate_hz: float


class FrequencyShifter:
    """
    GPU-accelerated frequency shifter (complex mixer).

    Caches complex exponentials for repeated shifts to avoid recomputation.
    """

    def __init__(self, sample_rate: float, cache_size: int = 10):
        """
        Initialize frequency shifter.

        Args:
            sample_rate: Sample rate in Hz.
            cache_size: Maximum number of cached exponentials.
        """
        self._sample_rate = sample_rate
        self._cache_size = cache_size
        self._exp_cache: dict[tuple[float, int], cp.ndarray] = {}

    def shift(self, signal: cp.ndarray, shift_hz: float) -> cp.ndarray:
        """
        Frequency shift a signal.

        Args:
            signal: Input signal (complex).
            shift_hz: Frequency shift in Hz (positive = up, negative = down).

        Returns:
            Frequency-shifted signal.
        """
        signal = cp.asarray(signal)
        n_samples = len(signal)

        # Get or generate complex exponential
        exp = self._get_exponential(shift_hz, n_samples)

        # Apply shift via multiplication
        return signal * exp

    def _get_exponential(self, shift_hz: float, n_samples: int) -> cp.ndarray:
        """Get cached or generate new complex exponential."""
        cache_key = (shift_hz, n_samples)

        if cache_key in self._exp_cache:
            return self._exp_cache[cache_key]

        # Generate complex exponential
        t = cp.arange(n_samples, dtype=cp.float64) / self._sample_rate
        exp = cp.exp(2j * cp.pi * shift_hz * t).astype(cp.complex64)

        # Cache with LRU eviction
        if len(self._exp_cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._exp_cache))
            del self._exp_cache[oldest_key]

        self._exp_cache[cache_key] = exp
        return exp

    def clear_cache(self) -> None:
        """Clear exponential cache."""
        self._exp_cache.clear()


class Decimator:
    """
    GPU-accelerated decimation with anti-aliasing filter.

    Implements polyphase decimation for efficiency.
    """

    def __init__(
        self, factor: int, sample_rate: float, filter_order: int = 64, cutoff_ratio: float = 0.8
    ):
        """
        Initialize decimator.

        Args:
            factor: Decimation factor.
            sample_rate: Input sample rate in Hz.
            filter_order: Anti-aliasing filter order.
            cutoff_ratio: Filter cutoff as ratio of Nyquist frequency.
        """
        if factor < 1:
            raise ValueError("Decimation factor must be >= 1")

        self._factor = factor
        self._sample_rate = sample_rate
        self._output_rate = sample_rate / factor

        # Design anti-aliasing filter
        cutoff = (sample_rate / factor / 2) * cutoff_ratio
        self._filter = self._design_lowpass(filter_order, cutoff, sample_rate)

    def _design_lowpass(self, order: int, cutoff: float, sample_rate: float) -> cp.ndarray:
        """Design lowpass FIR filter using windowed sinc method."""
        # Normalized cutoff frequency
        fc = cutoff / sample_rate

        # Generate filter coefficients
        n = cp.arange(order + 1, dtype=cp.float64)
        m = n - order / 2

        # Sinc function with handling for center point
        h = cp.where(m == 0, 2 * fc, cp.sin(2 * cp.pi * fc * m) / (cp.pi * m))

        # Apply Hamming window
        window = 0.54 - 0.46 * cp.cos(2 * cp.pi * n / order)
        h = h * window

        # Normalize
        h = h / cp.sum(h)

        return h.astype(cp.float32)

    def decimate(self, signal: cp.ndarray) -> cp.ndarray:
        """
        Decimate signal with anti-aliasing filtering.

        Args:
            signal: Input signal.

        Returns:
            Decimated signal at reduced sample rate.
        """
        signal = cp.asarray(signal)

        # Apply anti-aliasing filter
        filtered = self._convolve(signal, self._filter)

        # Downsample
        return filtered[:: self._factor]

    def _convolve(self, signal: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
        """GPU-accelerated convolution using FFT."""
        n_signal = len(signal)
        n_kernel = len(kernel)
        n_fft = n_signal + n_kernel - 1

        # Pad to next power of 2 for efficiency
        n_fft_padded = 1 << (n_fft - 1).bit_length()

        # FFT convolution
        signal_fft = cp.fft.fft(signal, n_fft_padded)
        kernel_fft = cp.fft.fft(kernel, n_fft_padded)
        result = cp.fft.ifft(signal_fft * kernel_fft)

        # Trim to valid length and compensate for filter delay
        delay = n_kernel // 2
        return result[delay : delay + n_signal]

    @property
    def factor(self) -> int:
        return self._factor

    @property
    def output_sample_rate(self) -> float:
        return self._output_rate


class Channelizer:
    """
    GPU-accelerated polyphase channelizer.

    Splits a wideband signal into multiple narrow-band channels.
    """

    def __init__(self, num_channels: int, sample_rate: float, filter_taps_per_channel: int = 16):
        """
        Initialize channelizer.

        Args:
            num_channels: Number of output channels.
            sample_rate: Input sample rate in Hz.
            filter_taps_per_channel: Filter taps per polyphase branch.
        """
        self._num_channels = num_channels
        self._sample_rate = sample_rate
        self._channel_rate = sample_rate / num_channels
        self._taps_per_channel = filter_taps_per_channel

        # Design prototype lowpass filter
        self._prototype = self._design_prototype_filter()

        # Decompose into polyphase branches
        self._polyphase_filters = self._decompose_polyphase()

    def _design_prototype_filter(self) -> cp.ndarray:
        """Design prototype lowpass filter."""
        num_taps = self._num_channels * self._taps_per_channel
        cutoff = 0.5 / self._num_channels  # Normalized cutoff

        n = cp.arange(num_taps, dtype=cp.float64)
        m = n - (num_taps - 1) / 2

        # Sinc filter
        h = cp.where(m == 0, 2 * cutoff, cp.sin(2 * cp.pi * cutoff * m) / (cp.pi * m))

        # Kaiser window for better stopband
        beta = 8.0
        alpha = (num_taps - 1) / 2
        x = beta * cp.sqrt(1 - ((n - alpha) / alpha) ** 2)
        window = cp.i0(x) / cp.i0(cp.array([beta]))[0]

        return (h * window).astype(cp.float32)

    def _decompose_polyphase(self) -> cp.ndarray:
        """Decompose prototype filter into polyphase branches."""
        # Reshape into polyphase matrix
        filters = self._prototype.reshape(self._taps_per_channel, self._num_channels).T
        return filters  # Shape: (num_channels, taps_per_channel)

    def channelize(self, signal: cp.ndarray) -> list[cp.ndarray]:
        """
        Split signal into multiple channels.

        Args:
            signal: Input wideband signal.

        Returns:
            List of narrow-band channel signals.
        """
        signal = cp.asarray(signal)

        # Ensure signal length is multiple of num_channels
        pad_len = (self._num_channels - len(signal) % self._num_channels) % self._num_channels
        if pad_len > 0:
            signal = cp.pad(signal, (0, pad_len))

        # Reshape for polyphase processing
        n_blocks = len(signal) // self._num_channels
        signal_matrix = signal.reshape(n_blocks, self._num_channels)

        # Apply polyphase filters
        filtered = cp.zeros((self._num_channels, n_blocks), dtype=cp.complex64)
        for i in range(self._num_channels):
            filtered[i] = self._fir_filter(signal_matrix[:, i], self._polyphase_filters[i])

        # Apply FFT to complete channelization
        channels = cp.fft.fft(filtered, axis=0)

        return [channels[i] for i in range(self._num_channels)]

    def _fir_filter(self, signal: cp.ndarray, coeffs: cp.ndarray) -> cp.ndarray:
        """Apply FIR filter to signal."""
        return cp.convolve(signal, coeffs, mode="same")

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    def channel_sample_rate(self) -> float:
        return self._channel_rate

    @property
    def channel_bandwidth(self) -> float:
        return self._sample_rate / self._num_channels


def frequency_shift(signal: cp.ndarray, shift_hz: float, sample_rate: float) -> cp.ndarray:
    """
    Frequency shift a signal (convenience function).

    Args:
        signal: Input signal.
        shift_hz: Frequency shift in Hz.
        sample_rate: Sample rate in Hz.

    Returns:
        Shifted signal.
    """
    shifter = FrequencyShifter(sample_rate)
    return shifter.shift(signal, shift_hz)


def decimate(
    signal: cp.ndarray, factor: int, sample_rate: float, filter_order: int = 64
) -> cp.ndarray:
    """
    Decimate a signal (convenience function).

    Args:
        signal: Input signal.
        factor: Decimation factor.
        sample_rate: Sample rate in Hz.
        filter_order: Anti-aliasing filter order.

    Returns:
        Decimated signal.
    """
    decimator = Decimator(factor, sample_rate, filter_order)
    return decimator.decimate(signal)


def hilbert(signal: cp.ndarray) -> cp.ndarray:
    """
    Compute the analytic signal using Hilbert transform.

    Args:
        signal: Real-valued input signal.

    Returns:
        Complex analytic signal.
    """
    signal = cp.asarray(signal)
    n = len(signal)

    # FFT of signal
    spectrum = cp.fft.fft(signal)

    # Create frequency-domain Hilbert transform
    h = cp.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2

    # Apply and inverse FFT
    analytic = cp.fft.ifft(spectrum * h)

    return analytic.astype(cp.complex64)


def compute_instantaneous_frequency(signal: cp.ndarray, sample_rate: float) -> cp.ndarray:
    """
    Compute instantaneous frequency from complex signal.

    Uses the derivative of the phase: f(t) = (1/2π) * dφ/dt

    Args:
        signal: Complex input signal.
        sample_rate: Sample rate in Hz.

    Returns:
        Instantaneous frequency array in Hz.
    """
    signal = cp.asarray(signal)

    # Compute phase derivative via angle difference
    phase = cp.angle(signal)

    # Unwrap phase
    phase_diff = cp.diff(phase)
    phase_diff = cp.where(phase_diff > cp.pi, phase_diff - 2 * cp.pi, phase_diff)
    phase_diff = cp.where(phase_diff < -cp.pi, phase_diff + 2 * cp.pi, phase_diff)

    # Convert to frequency
    inst_freq = phase_diff * sample_rate / (2 * cp.pi)

    # Pad to match input length
    return cp.pad(inst_freq, (0, 1), mode="edge")


def compute_envelope(signal: cp.ndarray) -> cp.ndarray:
    """
    Compute signal envelope (magnitude of analytic signal).

    Args:
        signal: Input signal (real or complex).

    Returns:
        Envelope array.
    """
    signal = cp.asarray(signal)

    if cp.iscomplexobj(signal):
        return cp.abs(signal)
    else:
        analytic = hilbert(signal)
        return cp.abs(analytic)
