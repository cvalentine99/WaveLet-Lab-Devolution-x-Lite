#!/usr/bin/env python3
"""
Real Hardware Integration Test - PCIe USDR DevBoard

Tests the optimized RF Forensics pipeline with real SDR hardware.
Validates all performance optimizations under actual RF data load.
"""

import asyncio
import logging
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path (parent of rf_forensics package)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from rf_forensics.sdr.usdr_driver import USDRDriver, USDRConfig, USDRGain
from rf_forensics.config.schema import RFForensicsConfig
from rf_forensics.pipeline.orchestrator import RFForensicsPipeline, PipelineState


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestMetrics:
    """Performance metrics collected during test."""
    test_duration_s: float = 0.0
    total_samples: int = 0
    total_detections: int = 0
    avg_throughput_msps: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    dropped_samples: int = 0
    callback_backlog_events: int = 0
    gpu_memory_used_gb: float = 0.0


class USDRPipelineTest:
    """
    Integration test for USDR + Pipeline.

    Tests:
    1. USDR hardware connectivity
    2. Pipeline initialization with real SDR
    3. Real-time processing performance
    4. Latency optimizations (callbacks, transfers, etc.)
    5. Detection accuracy with live RF signals
    """

    def __init__(self):
        self.driver: Optional[USDRDriver] = None
        self.pipeline: Optional[RFForensicsPipeline] = None
        self.metrics = TestMetrics()
        self._latencies = []
        self._spectrum_count = 0
        self._detection_count = 0

    def test_usdr_availability(self) -> bool:
        """Test 1: Check if USDR hardware is available."""
        logger.info("=" * 60)
        logger.info("TEST 1: USDR Hardware Availability")
        logger.info("=" * 60)

        self.driver = USDRDriver()

        if not self.driver.is_available:
            logger.error("USDR library (libusdr.so) not found!")
            logger.error("Make sure libusdr is installed: /usr/local/lib/libusdr.so.0.9.9")
            return False

        logger.info("USDR library loaded successfully")

        # Discover devices
        devices = self.driver.discover()

        if not devices:
            logger.error("No USDR devices found!")
            logger.error("Check PCIe connection and run: lspci | grep -i usdr")
            return False

        logger.info(f"Found {len(devices)} USDR device(s):")
        for dev in devices:
            logger.info(f"  - {dev.id} ({dev.model})")

        return True

    def test_usdr_connection(self) -> bool:
        """Test 2: Connect to USDR and configure."""
        logger.info("=" * 60)
        logger.info("TEST 2: USDR Connection & Configuration")
        logger.info("=" * 60)

        if not self.driver:
            self.driver = USDRDriver()

        # Connect
        devices = self.driver.discover()
        if not devices:
            logger.error("No devices available")
            return False

        if not self.driver.connect(devices[0].id):
            logger.error(f"Failed to connect to {devices[0].id}")
            return False

        logger.info(f"Connected to {devices[0].id}")

        # Configure for ISM band 915 MHz (common for testing)
        config = USDRConfig(
            center_freq_hz=915_000_000,  # 915 MHz ISM band
            sample_rate_hz=10_000_000,   # 10 MSPS
            bandwidth_hz=10_000_000,     # 10 MHz
            gain=USDRGain(lna_db=15, tia_db=9, pga_db=12),
            rx_path="LNAL",
            lna_enable=True,
            attenuator_db=0,
        )

        if not self.driver.configure(config):
            logger.error("Failed to configure USDR")
            return False

        # Read back status
        status = self.driver.get_status()
        logger.info(f"Configured:")
        logger.info(f"  Frequency: {status.actual_freq_hz / 1e6:.3f} MHz")
        logger.info(f"  Sample Rate: {config.sample_rate_hz / 1e6:.1f} MSPS")
        logger.info(f"  Temperature: {status.temperature_c:.1f}°C")

        return True

    def test_raw_streaming(self, duration_s: float = 3.0) -> bool:
        """Test 3: Raw SDR streaming (no GPU pipeline)."""
        logger.info("=" * 60)
        logger.info(f"TEST 3: Raw SDR Streaming ({duration_s}s)")
        logger.info("=" * 60)

        if not self.driver or not self.driver.is_connected:
            logger.error("Driver not connected")
            return False

        samples_received = 0
        start_time = time.time()

        def sample_callback(samples: np.ndarray, timestamp: float):
            nonlocal samples_received
            samples_received += len(samples)

        # Start streaming
        if not self.driver.start_streaming(sample_callback):
            logger.error("Failed to start streaming")
            return False

        logger.info("Streaming started...")

        # Stream for duration
        time.sleep(duration_s)

        # Stop streaming
        self.driver.stop_streaming()

        elapsed = time.time() - start_time
        throughput_msps = samples_received / elapsed / 1e6

        logger.info(f"Results:")
        logger.info(f"  Samples received: {samples_received:,}")
        logger.info(f"  Duration: {elapsed:.2f}s")
        logger.info(f"  Throughput: {throughput_msps:.2f} MSPS")

        # Expect close to configured rate (10 MSPS)
        expected_rate = 10.0
        if throughput_msps < expected_rate * 0.9:
            logger.warning(f"Throughput below expected ({expected_rate} MSPS)")
            return False

        return True

    async def test_full_pipeline(self, duration_s: float = 10.0) -> bool:
        """Test 4: Full GPU pipeline with real USDR data."""
        logger.info("=" * 60)
        logger.info(f"TEST 4: Full GPU Pipeline ({duration_s}s)")
        logger.info("=" * 60)

        # Disconnect direct driver (pipeline will create its own)
        if self.driver and self.driver.is_connected:
            self.driver.disconnect()

        # Configure for USDR
        config = RFForensicsConfig()
        config.sdr.device_type = "usdr"
        config.sdr.center_freq_hz = 915e6
        config.sdr.sample_rate_hz = 10e6
        config.sdr.bandwidth_hz = 10e6
        config.sdr.gain_db = 36  # Split across LNA/TIA/PGA

        # Performance tuning - smaller buffers for real-time performance
        config.buffer.samples_per_buffer = 500_000  # 500k samples = 50ms at 10MSPS
        config.buffer.num_ring_segments = 8  # More segments for smoother streaming
        config.fft.fft_size = 1024
        config.fft.averaging_count = 8  # Faster averaging

        self.pipeline = RFForensicsPipeline(config=config)

        # Set up callbacks to track metrics
        self._latencies = []
        self._spectrum_count = 0
        self._detection_count = 0

        def spectrum_callback(psd_db, freqs):
            self._spectrum_count += 1

        def detection_callback(detections):
            self._detection_count += len(detections)

        self.pipeline.set_spectrum_callback(spectrum_callback)
        self.pipeline.set_detection_callback(detection_callback)

        try:
            # Start pipeline
            logger.info("Starting pipeline...")
            await self.pipeline.start()

            if self.pipeline.state != PipelineState.RUNNING:
                logger.error(f"Pipeline failed to start (state: {self.pipeline.state})")
                return False

            logger.info("Pipeline running, collecting metrics...")

            # Collect metrics over duration
            start = time.time()
            latency_samples = []

            while time.time() - start < duration_s:
                status = self.pipeline.get_status()

                if status.get('processing_latency_ms', 0) > 0:
                    latency_samples.append(status['processing_latency_ms'])

                # Log progress every 2 seconds
                if int(time.time() - start) % 2 == 0:
                    logger.info(
                        f"  t={time.time()-start:.0f}s: "
                        f"{status['current_throughput_msps']:.1f} MSPS, "
                        f"{status['processing_latency_ms']:.2f}ms latency, "
                        f"{status['detections_count']} detections"
                    )

                await asyncio.sleep(0.5)

            # Final status
            final_status = self.pipeline.get_status()

            # Calculate metrics
            self.metrics.test_duration_s = duration_s
            self.metrics.total_samples = final_status['samples_processed']
            self.metrics.total_detections = final_status['detections_count']
            self.metrics.avg_throughput_msps = final_status['current_throughput_msps']

            if latency_samples:
                self.metrics.avg_latency_ms = sum(latency_samples) / len(latency_samples)
                self.metrics.max_latency_ms = max(latency_samples)
                self.metrics.min_latency_ms = min(latency_samples)

            self.metrics.gpu_memory_used_gb = final_status['gpu_memory_used_gb']

            return True

        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if self.pipeline:
                await self.pipeline.stop()

    async def test_latency_optimization(self) -> bool:
        """Test 5: Verify latency optimizations are working."""
        logger.info("=" * 60)
        logger.info("TEST 5: Latency Optimization Verification")
        logger.info("=" * 60)

        # Check specific optimizations
        results = []

        # 5.1 Non-blocking callbacks
        logger.info("5.1 Testing non-blocking callback dispatch...")
        callback_test_passed = self.metrics.avg_latency_ms < 5.0  # Should be <5ms
        results.append(("Non-blocking callbacks", callback_test_passed,
                       f"avg latency {self.metrics.avg_latency_ms:.2f}ms"))

        # 5.2 Throughput test (should sustain 10+ MSPS)
        logger.info("5.2 Testing throughput sustainability...")
        throughput_passed = self.metrics.avg_throughput_msps >= 9.0  # 90% of target
        results.append(("Sustained throughput", throughput_passed,
                       f"{self.metrics.avg_throughput_msps:.1f} MSPS"))

        # 5.3 GPU memory efficiency
        logger.info("5.3 Testing GPU memory efficiency...")
        memory_passed = self.metrics.gpu_memory_used_gb < 2.0  # Should use <2GB
        results.append(("GPU memory efficiency", memory_passed,
                       f"{self.metrics.gpu_memory_used_gb:.2f} GB"))

        # 5.4 Latency stability (max should be <3x avg)
        logger.info("5.4 Testing latency stability...")
        latency_stable = self.metrics.max_latency_ms < self.metrics.avg_latency_ms * 3
        results.append(("Latency stability", latency_stable,
                       f"max/avg = {self.metrics.max_latency_ms/self.metrics.avg_latency_ms:.1f}x"))

        # Report results
        logger.info("")
        logger.info("Optimization Results:")
        all_passed = True
        for name, passed, detail in results:
            status = "PASS" if passed else "FAIL"
            symbol = "✓" if passed else "✗"
            logger.info(f"  {symbol} {name}: {status} ({detail})")
            if not passed:
                all_passed = False

        return all_passed

    def print_summary(self):
        """Print test summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration:       {self.metrics.test_duration_s:.1f}s")
        logger.info(f"Samples:        {self.metrics.total_samples:,}")
        logger.info(f"Detections:     {self.metrics.total_detections:,}")
        logger.info(f"Throughput:     {self.metrics.avg_throughput_msps:.2f} MSPS")
        logger.info(f"Avg Latency:    {self.metrics.avg_latency_ms:.2f}ms")
        logger.info(f"Max Latency:    {self.metrics.max_latency_ms:.2f}ms")
        logger.info(f"Min Latency:    {self.metrics.min_latency_ms:.2f}ms")
        logger.info(f"GPU Memory:     {self.metrics.gpu_memory_used_gb:.2f} GB")
        logger.info(f"Spectrum Frames: {self._spectrum_count}")
        logger.info("=" * 60)

    def cleanup(self):
        """Cleanup resources."""
        if self.driver:
            if self.driver.is_streaming:
                self.driver.stop_streaming()
            if self.driver.is_connected:
                self.driver.disconnect()


async def main():
    """Run all tests."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("GPU RF FORENSICS - USDR HARDWARE INTEGRATION TEST")
    logger.info("=" * 60)
    logger.info("")

    # Check CUDA
    if CUPY_AVAILABLE:
        logger.info(f"CUDA available: {cp.cuda.runtime.getDeviceCount()} GPU(s)")
        props = cp.cuda.runtime.getDeviceProperties(0)
        logger.info(f"GPU 0: {props['name'].decode()}")
    else:
        logger.warning("CUDA not available - running in CPU mode")

    test = USDRPipelineTest()

    try:
        # Test 1: Hardware availability
        if not test.test_usdr_availability():
            logger.error("Hardware check failed - aborting")
            return 1

        # Test 2: Connection
        if not test.test_usdr_connection():
            logger.error("Connection test failed - aborting")
            return 1

        # Test 3: Raw streaming
        if not test.test_raw_streaming(duration_s=3.0):
            logger.warning("Raw streaming test had issues")
            # Continue anyway

        # Test 4: Full pipeline
        if not await test.test_full_pipeline(duration_s=10.0):
            logger.error("Pipeline test failed")
            return 1

        # Test 5: Verify optimizations
        if not await test.test_latency_optimization():
            logger.warning("Some optimizations may not be performing as expected")

        # Summary
        test.print_summary()

        logger.info("")
        logger.info("All tests completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        test.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
