"""
RF Forensics Holoscan Pipeline - Python Interface

High-performance GPU signal processing pipeline using NVIDIA Holoscan.
Target: 9.5+ MSPS throughput, <1ms latency (vs 6.57 MSPS Python/CuPy baseline)

Usage:
    from rf_forensics.holoscan.python import HoloscanPipeline

    pipeline = HoloscanPipeline()
    pipeline.configure(
        center_freq=915e6,
        sample_rate=10e6,
        simulate=True  # Use simulated data for testing
    )
    pipeline.start()

    # Get detections
    while pipeline.is_running():
        detections = pipeline.get_detections(timeout_ms=100)
        for det in detections:
            print(f"{det.frequency_hz/1e6:.3f} MHz: {det.power_db:.1f} dB")

    pipeline.stop()
"""

try:
    # Try to import the compiled C++ module
    from rf_forensics_holoscan import (
        HoloscanPipeline,
        Detection,
        __version__,
        __target_throughput__,
        __target_latency__,
    )
except ImportError:
    # Fallback: provide a stub that raises a helpful error
    import warnings

    warnings.warn(
        "rf_forensics_holoscan C++ module not found. "
        "Build with: cd holoscan && mkdir build && cd build && cmake .. && make"
    )

    class Detection:
        """Detection result from CFAR."""
        def __init__(self):
            self.bin_index = 0
            self.frequency_hz = 0.0
            self.power_db = 0.0
            self.snr_db = 0.0
            self.timestamp_ns = 0

    class HoloscanPipeline:
        """Stub - build the C++ module to use the real pipeline."""
        def __init__(self):
            raise NotImplementedError(
                "Holoscan C++ module not built. Run:\n"
                "  cd rf_forensics/holoscan\n"
                "  mkdir build && cd build\n"
                "  cmake .. && make"
            )

    __version__ = "1.0.0-stub"
    __target_throughput__ = "9.5+ MSPS"
    __target_latency__ = "<1ms"


__all__ = [
    "HoloscanPipeline",
    "Detection",
    "__version__",
    "__target_throughput__",
    "__target_latency__",
]
