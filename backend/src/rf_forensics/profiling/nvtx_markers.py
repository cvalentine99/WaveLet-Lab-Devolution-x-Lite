"""
NVTX Instrumentation for Nsight Systems Profiling

Provides NVTX markers and ranges for GPU timeline visualization.
Use with: nsys profile --trace=cuda,nvtx python script.py
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

try:
    import cupy as cp
    from cupyx.profiler import time_range as nvtx_range

    # Also try native NVTX for more control
    try:
        import nvtx

        NVTX_NATIVE = True
    except ImportError:
        nvtx = None
        NVTX_NATIVE = False
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    nvtx = None
    CUPY_AVAILABLE = False
    NVTX_NATIVE = False


# Color palette for different pipeline stages (ARGB format)
NVTX_COLORS = {
    "sdr_ingest": 0xFF4CAF50,  # Green - SDR data arrival
    "h2d_transfer": 0xFF2196F3,  # Blue - Host to Device
    "psd_compute": 0xFFFF9800,  # Orange - PSD/FFT
    "cfar_detect": 0xFFE91E63,  # Pink - CFAR detection
    "peak_extract": 0xFF9C27B0,  # Purple - Peak extraction
    "d2h_transfer": 0xFF00BCD4,  # Cyan - Device to Host
    "clustering": 0xFFFFEB3B,  # Yellow - Clustering
    "websocket": 0xFF795548,  # Brown - WebSocket send
    "callback": 0xFF607D8B,  # Gray - Callbacks
    "error": 0xFFF44336,  # Red - Errors
}


@dataclass
class TimingStats:
    """Accumulated timing statistics."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    last_ms: float = 0.0

    def update(self, duration_ms: float):
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        self.last_ms = duration_ms

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "avg_ms": self.avg_ms,
            "min_ms": self.min_ms if self.min_ms != float("inf") else 0.0,
            "max_ms": self.max_ms,
            "last_ms": self.last_ms,
        }


class NVTXProfiler:
    """
    NVTX profiling manager for RF Forensics pipeline.

    Provides:
    - Named ranges for Nsight Systems visualization
    - Automatic timing collection
    - Stream-aware GPU timing with CUDA events
    - Hierarchical range nesting

    Usage:
        profiler = NVTXProfiler()

        with profiler.range("psd_compute"):
            psd = compute_psd(signal)

        # Or as decorator
        @profiler.trace("cfar_detect")
        def detect(psd):
            ...

        # GPU-timed range (uses CUDA events)
        with profiler.gpu_range("d2h_transfer", stream):
            cp.asnumpy(data)
    """

    def __init__(self, enabled: bool = True, collect_stats: bool = True):
        """
        Initialize NVTX profiler.

        Args:
            enabled: Whether to emit NVTX markers (disable for production)
            collect_stats: Whether to collect timing statistics
        """
        self._enabled = enabled and CUPY_AVAILABLE
        self._collect_stats = collect_stats
        self._stats: dict[str, TimingStats] = {}
        self._active_ranges: list = []

        # CUDA events for GPU timing
        self._events: dict[str, tuple] = {}  # name -> (start_event, end_event)

    @contextmanager
    def range(self, name: str, color: int | None = None):
        """
        Context manager for NVTX range.

        Args:
            name: Range name (appears in Nsight timeline)
            color: Optional ARGB color (uses default palette if not specified)
        """
        if not self._enabled:
            yield
            return

        color = color or NVTX_COLORS.get(name, 0xFF808080)
        start_time = time.perf_counter()

        # Push NVTX range
        if NVTX_NATIVE and nvtx is not None:
            nvtx.push_range(name, color=color)
        elif CUPY_AVAILABLE:
            # CuPy's nvtx wrapper
            self._active_ranges.append(name)

        try:
            yield
        finally:
            end_time = time.perf_counter()

            # Pop NVTX range
            if NVTX_NATIVE and nvtx is not None:
                nvtx.pop_range()
            elif self._active_ranges:
                self._active_ranges.pop()

            # Collect CPU timing stats
            if self._collect_stats:
                duration_ms = (end_time - start_time) * 1000
                if name not in self._stats:
                    self._stats[name] = TimingStats()
                self._stats[name].update(duration_ms)

    @contextmanager
    def gpu_range(self, name: str, stream=None, color: int | None = None):
        """
        Context manager for GPU-timed NVTX range using CUDA events.

        More accurate than CPU timing for GPU operations.

        Args:
            name: Range name
            stream: CUDA stream (uses default if None)
            color: Optional ARGB color
        """
        if not self._enabled or not CUPY_AVAILABLE:
            yield
            return

        color = color or NVTX_COLORS.get(name, 0xFF808080)

        # Create CUDA events
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Record start
        if stream is not None:
            start_event.record(stream)
        else:
            start_event.record()

        # Push NVTX range
        if NVTX_NATIVE and nvtx is not None:
            nvtx.push_range(name, color=color)

        try:
            yield
        finally:
            # Record end
            if stream is not None:
                end_event.record(stream)
            else:
                end_event.record()

            # Pop NVTX range
            if NVTX_NATIVE and nvtx is not None:
                nvtx.pop_range()

            # Collect GPU timing stats (requires sync)
            if self._collect_stats:
                end_event.synchronize()
                duration_ms = cp.cuda.get_elapsed_time(start_event, end_event)
                if name not in self._stats:
                    self._stats[name] = TimingStats()
                self._stats[name].update(duration_ms)

    def mark(self, name: str, color: int | None = None):
        """
        Emit a single NVTX marker (instant event).

        Args:
            name: Marker name
            color: Optional ARGB color
        """
        if not self._enabled:
            return

        if NVTX_NATIVE and nvtx is not None:
            nvtx.mark(name, color=color or 0xFFFFFFFF)

    def trace(self, name: str, color: int | None = None):
        """
        Decorator to trace a function with NVTX range.

        Args:
            name: Range name
            color: Optional ARGB color
        """

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.range(name, color):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def gpu_trace(self, name: str, stream_arg: str = "stream", color: int | None = None):
        """
        Decorator to trace a function with GPU-timed NVTX range.

        Args:
            name: Range name
            stream_arg: Name of stream argument in function signature
            color: Optional ARGB color
        """

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                stream = kwargs.get(stream_arg)
                with self.gpu_range(name, stream, color):
                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get all collected timing statistics."""
        return {name: stats.to_dict() for name, stats in self._stats.items()}

    def print_stats(self):
        """Print timing statistics summary."""
        print("\n" + "=" * 70)
        print("NVTX Profiling Statistics")
        print("=" * 70)
        print(f"{'Stage':<25} {'Count':>8} {'Avg':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)

        for name, stats in sorted(self._stats.items()):
            print(
                f"{name:<25} {stats.count:>8} {stats.avg_ms:>9.3f}ms "
                f"{stats.min_ms:>9.3f}ms {stats.max_ms:>9.3f}ms"
            )

        print("=" * 70)

    def reset_stats(self):
        """Reset all timing statistics."""
        self._stats.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value and CUPY_AVAILABLE


# Global profiler instance
_global_profiler: NVTXProfiler | None = None


def get_profiler() -> NVTXProfiler:
    """Get or create global NVTX profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = NVTXProfiler()
    return _global_profiler


def nvtx_range(name: str, color: int | None = None):
    """Convenience function for global profiler range."""
    return get_profiler().range(name, color)


def nvtx_gpu_range(name: str, stream=None, color: int | None = None):
    """Convenience function for global profiler GPU range."""
    return get_profiler().gpu_range(name, stream, color)


def nvtx_mark(name: str, color: int | None = None):
    """Convenience function for global profiler mark."""
    get_profiler().mark(name, color)


def nvtx_trace(name: str, color: int | None = None):
    """Convenience decorator for global profiler trace."""
    return get_profiler().trace(name, color)
