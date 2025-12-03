"""
GPU RF Forensics Engine - GPU Profiling Infrastructure

Provides CUDA event-based timing for measuring GPU operation performance.
Use this to benchmark before/after optimization changes.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class TimingStats:
    """Statistics for a named timing region."""

    name: str
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    total_ms: float


class GPUProfiler:
    """
    GPU-aware profiling using CUDA events.

    Uses CUDA events for accurate GPU timing that accounts for
    asynchronous execution. Falls back to CPU timing when GPU
    is not available.

    Usage:
        from rf_forensics.core.profiling import profiler

        with profiler.time_gpu("cfar_detect"):
            result = cfar.detect(psd)

        # Get statistics
        stats = profiler.report()
        print(f"CFAR mean: {stats['cfar_detect']['mean_ms']:.3f} ms")
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize profiler.

        Args:
            enabled: Whether profiling is active. Set False in production.
        """
        self._enabled = enabled
        self._results: dict[str, list[float]] = {}
        self._active_timers: dict[str, tuple] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @contextmanager
    def time_gpu(self, name: str):
        """
        Context manager for timing a GPU operation using CUDA events.

        Args:
            name: Identifier for this timing region.

        Yields:
            None - use as context manager.
        """
        if not self._enabled:
            yield
            return

        if CUPY_AVAILABLE:
            # Use CUDA events for accurate GPU timing
            start_event = cp.cuda.Event()
            end_event = cp.cuda.Event()

            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                end_event.synchronize()

                elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
                self._record(name, elapsed_ms)
        else:
            # Fallback to CPU timing
            start_time = time.perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._record(name, elapsed_ms)

    def time_cpu(self, name: str):
        """
        Context manager for timing a CPU operation.

        Args:
            name: Identifier for this timing region.
        """
        return self._time_cpu_impl(name)

    @contextmanager
    def _time_cpu_impl(self, name: str):
        if not self._enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._record(name, elapsed_ms)

    def _record(self, name: str, elapsed_ms: float) -> None:
        """Record a timing measurement."""
        if name not in self._results:
            self._results[name] = []
        self._results[name].append(elapsed_ms)

    def report(self) -> dict[str, dict[str, float]]:
        """
        Get timing statistics for all measured regions.

        Returns:
            Dictionary mapping region names to statistics.
        """
        stats = {}
        for name, times in self._results.items():
            if not times:
                continue
            arr = np.array(times)
            stats[name] = {
                "count": len(times),
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "total_ms": float(np.sum(arr)),
            }
        return stats

    def get_stats(self, name: str) -> TimingStats | None:
        """
        Get statistics for a specific timing region.

        Args:
            name: Region identifier.

        Returns:
            TimingStats or None if region not found.
        """
        if name not in self._results or not self._results[name]:
            return None

        times = np.array(self._results[name])
        return TimingStats(
            name=name,
            count=len(times),
            mean_ms=float(np.mean(times)),
            std_ms=float(np.std(times)),
            min_ms=float(np.min(times)),
            max_ms=float(np.max(times)),
            total_ms=float(np.sum(times)),
        )

    def clear(self, name: str | None = None) -> None:
        """
        Clear timing results.

        Args:
            name: Specific region to clear, or None for all.
        """
        if name is None:
            self._results.clear()
        elif name in self._results:
            del self._results[name]

    def summary(self) -> str:
        """
        Generate human-readable summary of all timings.

        Returns:
            Formatted string with timing statistics.
        """
        stats = self.report()
        if not stats:
            return "No timing data recorded."

        lines = ["GPU Profiling Summary", "=" * 60]

        # Sort by total time descending
        sorted_names = sorted(stats.keys(), key=lambda n: stats[n]["total_ms"], reverse=True)

        for name in sorted_names:
            s = stats[name]
            lines.append(
                f"{name:30s} | "
                f"mean: {s['mean_ms']:8.3f} ms | "
                f"std: {s['std_ms']:7.3f} ms | "
                f"n={s['count']:5d} | "
                f"total: {s['total_ms']:10.1f} ms"
            )

        return "\n".join(lines)

    def compare(self, name1: str, name2: str) -> dict[str, float] | None:
        """
        Compare two timing regions.

        Args:
            name1: First region (typically "before").
            name2: Second region (typically "after").

        Returns:
            Dictionary with comparison metrics, or None if data missing.
        """
        stats1 = self.get_stats(name1)
        stats2 = self.get_stats(name2)

        if stats1 is None or stats2 is None:
            return None

        speedup = stats1.mean_ms / stats2.mean_ms if stats2.mean_ms > 0 else float("inf")

        return {
            f"{name1}_mean_ms": stats1.mean_ms,
            f"{name2}_mean_ms": stats2.mean_ms,
            "speedup": speedup,
            "time_saved_ms": stats1.mean_ms - stats2.mean_ms,
            "time_saved_percent": (1 - stats2.mean_ms / stats1.mean_ms) * 100
            if stats1.mean_ms > 0
            else 0,
        }


# Global profiler instance - import and use directly
profiler = GPUProfiler(enabled=True)


def benchmark(
    func, *args, name: str | None = None, iterations: int = 10, warmup: int = 2, **kwargs
):
    """
    Benchmark a function with warmup iterations.

    Args:
        func: Function to benchmark.
        *args: Positional arguments for func.
        name: Name for timing region (defaults to func.__name__).
        iterations: Number of timed iterations.
        warmup: Number of warmup iterations (not timed).
        **kwargs: Keyword arguments for func.

    Returns:
        Tuple of (result, stats) where result is from last iteration.
    """
    region_name = name or func.__name__

    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)

    # Ensure GPU sync after warmup
    if CUPY_AVAILABLE:
        cp.cuda.Device().synchronize()

    # Clear any previous results for this region
    profiler.clear(region_name)

    # Timed iterations
    for _ in range(iterations):
        with profiler.time_gpu(region_name):
            result = func(*args, **kwargs)

    stats = profiler.get_stats(region_name)
    return result, stats
