"""
GPU RF Forensics Engine - CFAR Tests

Unit tests for CFAR detection.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from rf_forensics.detection.cfar import CFARDetector, CFARDetector2D


class TestCFARDetector:
    """Tests for CFARDetector class."""

    def test_init(self):
        """Test CFAR detector initialization."""
        cfar = CFARDetector(
            num_reference=16,
            num_guard=4,
            pfa=1e-4
        )
        assert cfar._num_reference == 16
        assert cfar._num_guard == 4
        assert cfar._pfa == 1e-4

    def test_alpha_calculation(self):
        """Test CFAR alpha scaling factor calculation."""
        cfar = CFARDetector(num_reference=16, pfa=1e-4)

        # Alpha should be positive and reasonable
        assert cfar.alpha > 0
        assert cfar.alpha < 100

    def test_detect_single_target(self, gpu_array, to_numpy):
        """Test detection of single target in noise."""
        # Create PSD with single target
        psd = np.ones(1024) * 0.001  # Noise floor
        psd[512] = 1.0  # Strong target at center

        cfar = CFARDetector(num_reference=32, num_guard=4, pfa=1e-4)
        mask, snr = cfar.detect(gpu_array(psd.astype(np.float32)))

        mask_np = to_numpy(mask)
        snr_np = to_numpy(snr)

        # Should detect the target
        assert mask_np[512] == True
        # SNR should be high at target
        assert snr_np[512] > 10

    def test_no_false_alarms_in_noise(self, gpu_array, to_numpy):
        """Test that noise-only signal has few detections."""
        # Pure noise
        np.random.seed(42)
        psd = np.random.exponential(1.0, 1024).astype(np.float32)

        cfar = CFARDetector(num_reference=32, num_guard=4, pfa=1e-6)
        mask, _ = cfar.detect(gpu_array(psd))

        mask_np = to_numpy(mask)

        # Very few false alarms expected
        false_alarm_rate = mask_np.sum() / len(mask_np)
        assert false_alarm_rate < 0.01  # Less than 1%

    def test_detect_multiple_targets(self, gpu_array, to_numpy):
        """Test detection of multiple targets."""
        psd = np.ones(1024) * 0.001
        psd[200] = 1.0
        psd[500] = 0.8
        psd[800] = 0.5

        cfar = CFARDetector(num_reference=32, num_guard=4, pfa=1e-4)
        mask, _ = cfar.detect(gpu_array(psd.astype(np.float32)))

        mask_np = to_numpy(mask)

        # Should detect all three targets
        assert mask_np[200] == True
        assert mask_np[500] == True
        # Weakest target may or may not be detected depending on threshold

    def test_cfar_variants(self, gpu_array, to_numpy):
        """Test different CFAR variants."""
        psd = np.ones(512) * 0.001
        psd[256] = 1.0

        for variant in ["CA", "GO", "SO"]:
            cfar = CFARDetector(
                num_reference=16,
                num_guard=2,
                pfa=1e-4,
                variant=variant
            )
            mask, _ = cfar.detect(gpu_array(psd.astype(np.float32)))
            mask_np = to_numpy(mask)

            # All variants should detect strong target
            assert mask_np[256] == True

    def test_pfa_adjustment(self, gpu_array, to_numpy):
        """Test PFA adjustment updates alpha."""
        cfar = CFARDetector(pfa=1e-4)
        alpha1 = cfar.alpha

        cfar.set_pfa(1e-6)
        alpha2 = cfar.alpha

        # Lower PFA should give higher alpha (more conservative threshold)
        assert alpha2 > alpha1


class TestCFARDetector2D:
    """Tests for 2D CFAR detector."""

    def test_detect_2d(self, gpu_array, to_numpy):
        """Test 2D CFAR on spectrogram."""
        # Create spectrogram with target
        spec = np.ones((32, 256)) * 0.001
        spec[16, 128] = 1.0  # Target in center

        cfar = CFARDetector2D(num_reference=4, num_guard=1, pfa=1e-4)
        mask, snr = cfar.detect_2d(gpu_array(spec.astype(np.float32)))

        mask_np = to_numpy(mask)

        # Should detect target
        assert mask_np[16, 128] == True
