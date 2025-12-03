"""
GPU RF Forensics Engine - PSD Tests

Unit tests for Welch PSD estimation.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from rf_forensics.dsp.psd import WelchPSD, SpectrogramBuffer


class TestWelchPSD:
    """Tests for WelchPSD class."""

    def test_init_valid_fft_size(self, sample_rate):
        """Test initialization with valid FFT size."""
        psd = WelchPSD(fft_size=1024, sample_rate=sample_rate)
        assert psd.fft_size == 1024
        assert psd.sample_rate == sample_rate

    def test_init_invalid_fft_size(self, sample_rate):
        """Test that non-power-of-2 FFT size raises error."""
        with pytest.raises(ValueError):
            WelchPSD(fft_size=1000, sample_rate=sample_rate)

    def test_compute_psd_shape(self, sample_rate, gpu_array):
        """Test PSD output shape."""
        psd = WelchPSD(fft_size=1024, sample_rate=sample_rate)
        signal = gpu_array(np.random.randn(10000).astype(np.complex64))

        freqs, result = psd.compute_psd(signal)

        assert len(freqs) == 1024
        assert len(result) == 1024

    def test_psd_detects_tone(self, sample_rate, test_signal_with_tone, gpu_array, to_numpy):
        """Test that PSD correctly identifies a tone."""
        # Create signal with tone at 100 kHz
        tone_freq = 100e3
        signal = test_signal_with_tone(100000, tone_freq, tone_power_dbm=-30)

        psd = WelchPSD(fft_size=1024, sample_rate=sample_rate)
        freqs, result = psd.compute_psd_db(gpu_array(signal))

        freqs_np = to_numpy(freqs)
        result_np = to_numpy(result)

        # Find peak
        peak_idx = np.argmax(result_np)
        peak_freq = freqs_np[peak_idx]

        # Check peak is near tone frequency
        assert abs(peak_freq - tone_freq) < sample_rate / 1024 * 2

    def test_resolution(self, sample_rate):
        """Test frequency resolution calculation."""
        fft_size = 2048
        psd = WelchPSD(fft_size=fft_size, sample_rate=sample_rate)

        expected_resolution = sample_rate / fft_size
        assert abs(psd.resolution_hz - expected_resolution) < 1

    def test_rolling_update(self, sample_rate, gpu_array):
        """Test rolling PSD update."""
        psd = WelchPSD(fft_size=512, sample_rate=sample_rate)

        for _ in range(5):
            signal = gpu_array(np.random.randn(5000).astype(np.complex64))
            freqs, rolling = psd.update_rolling(signal, alpha=0.3)

        assert rolling is not None
        assert len(rolling) == 512


class TestSpectrogramBuffer:
    """Tests for SpectrogramBuffer class."""

    def test_init(self, sample_rate):
        """Test spectrogram buffer initialization."""
        buf = SpectrogramBuffer(
            num_time_bins=128,
            fft_size=512,
            sample_rate=sample_rate
        )
        assert buf.resolution_hz == sample_rate / 512

    def test_update(self, sample_rate, gpu_array, to_numpy):
        """Test spectrogram update."""
        buf = SpectrogramBuffer(
            num_time_bins=64,
            fft_size=256,
            sample_rate=sample_rate
        )

        for _ in range(10):
            signal = gpu_array(np.random.randn(10000).astype(np.complex64))
            spec = buf.update(signal)

        spec_np = to_numpy(spec)
        assert spec_np.shape[1] == 256

    def test_uint8_quantization(self, sample_rate, gpu_array, to_numpy):
        """Test uint8 quantization for visualization."""
        buf = SpectrogramBuffer(
            num_time_bins=32,
            fft_size=128,
            sample_rate=sample_rate
        )

        signal = gpu_array(np.random.randn(5000).astype(np.complex64))
        buf.update(signal)

        spec_uint8 = buf.get_spectrogram_uint8(min_db=-100, max_db=-20)
        spec_np = to_numpy(spec_uint8)

        assert spec_np.dtype == np.uint8
        assert spec_np.min() >= 0
        assert spec_np.max() <= 255
