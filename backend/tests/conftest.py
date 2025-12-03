"""
GPU RF Forensics Engine - Test Configuration

Pytest fixtures and configuration for testing.
"""

import pytest
import numpy as np

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np


@pytest.fixture
def sample_rate():
    """Default sample rate for tests."""
    return 10e6


@pytest.fixture
def fft_size():
    """Default FFT size for tests."""
    return 1024


@pytest.fixture
def complex_noise(sample_rate):
    """Generate complex Gaussian noise."""
    def _generate(num_samples, snr_db=-100):
        noise_power = 10 ** (snr_db / 10)
        noise_std = np.sqrt(noise_power / 2)
        return (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) * noise_std
    return _generate


@pytest.fixture
def tone_signal(sample_rate):
    """Generate a complex tone signal."""
    def _generate(num_samples, freq_hz, power_dbm=-50):
        t = np.arange(num_samples) / sample_rate
        amplitude = np.sqrt(10 ** ((power_dbm - 30) / 10))
        return amplitude * np.exp(2j * np.pi * freq_hz * t)
    return _generate


@pytest.fixture
def modulated_signal(sample_rate):
    """Generate a modulated signal (QPSK)."""
    def _generate(num_samples, symbol_rate=10e3, power_dbm=-50):
        symbols_per_sample = symbol_rate / sample_rate
        symbol_indices = (np.arange(num_samples) * symbols_per_sample).astype(int)
        phases = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=symbol_indices.max() + 1)
        baseband = np.exp(1j * phases[symbol_indices])
        amplitude = np.sqrt(10 ** ((power_dbm - 30) / 10))
        return amplitude * baseband.astype(np.complex64)
    return _generate


@pytest.fixture
def test_signal_with_tone(complex_noise, tone_signal, sample_rate):
    """Generate test signal with tone in noise."""
    def _generate(num_samples, tone_freq_hz, tone_power_dbm=-50, noise_floor_dbm=-100):
        noise = complex_noise(num_samples, noise_floor_dbm)
        tone = tone_signal(num_samples, tone_freq_hz, tone_power_dbm)
        return (noise + tone).astype(np.complex64)
    return _generate


@pytest.fixture
def gpu_array():
    """Convert numpy array to GPU array if available."""
    def _convert(arr):
        if CUPY_AVAILABLE:
            return cp.asarray(arr)
        return arr
    return _convert


@pytest.fixture
def to_numpy():
    """Convert GPU array to numpy."""
    def _convert(arr):
        if CUPY_AVAILABLE and hasattr(arr, 'get'):
            return arr.get()
        return arr
    return _convert


# Markers for GPU tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CuPy not available."""
    if not CUPY_AVAILABLE:
        skip_gpu = pytest.mark.skip(reason="CuPy not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
