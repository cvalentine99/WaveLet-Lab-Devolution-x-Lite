"""
GPU Metrics Collection for RF Forensics Pipeline

Provides utilities for measuring:
- GPU memory usage and fragmentation
- D2H/H2D transfer bandwidth
- Kernel occupancy estimates
- Stream utilization
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

import numpy as np


@dataclass
class MemorySnapshot:
    """GPU memory state at a point in time."""

    timestamp: float
    total_bytes: int
    used_bytes: int
    free_bytes: int
    pool_used_bytes: int
    pool_free_bytes: int

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024**3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024**3)

    @property
    def utilization(self) -> float:
        return self.used_bytes / self.total_bytes if self.total_bytes > 0 else 0.0


@dataclass
class TransferMeasurement:
    """Single transfer bandwidth measurement."""

    direction: str  # "h2d" or "d2h"
    size_bytes: int
    duration_ms: float
    pinned: bool
    stream_async: bool

    @property
    def bandwidth_gbps(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return (self.size_bytes / 1e9) / (self.duration_ms / 1000)

    @property
    def bandwidth_effective(self) -> float:
        """Effective bandwidth considering PCIe overhead."""
        return self.bandwidth_gbps * 0.95  # ~5% PCIe overhead


@dataclass
class KernelStats:
    """Kernel execution statistics."""

    name: str
    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    shared_mem_bytes: int
    duration_ms: float

    @property
    def total_threads(self) -> int:
        return (
            self.grid_dim[0]
            * self.grid_dim[1]
            * self.grid_dim[2]
            * self.block_dim[0]
            * self.block_dim[1]
            * self.block_dim[2]
        )

    @property
    def threads_per_block(self) -> int:
        return self.block_dim[0] * self.block_dim[1] * self.block_dim[2]


class GPUMetricsCollector:
    """
    Collects GPU performance metrics for profiling.

    Provides:
    - Memory snapshots over time
    - Transfer bandwidth measurements
    - Kernel timing
    - Aggregate statistics

    Usage:
        collector = GPUMetricsCollector()

        # Memory tracking
        collector.snapshot_memory("before_psd")

        # Transfer measurement
        bw = collector.measure_d2h(gpu_array, pinned_buffer)

        # Get report
        report = collector.get_report()
    """

    def __init__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPUMetricsCollector requires CuPy")

        self._memory_snapshots: dict[str, MemorySnapshot] = {}
        self._transfers: list[TransferMeasurement] = []
        self._kernels: list[KernelStats] = []
        self._lock = threading.Lock()

        # Get device properties
        self._device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(self._device.id)
        self._device_props = {
            "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "compute_capability": (props.get("major", 0), props.get("minor", 0)),
            "total_memory_gb": self._device.mem_info[1] / (1024**3),
            "multiprocessor_count": props.get("multiProcessorCount", 0),
            "max_threads_per_block": props.get("maxThreadsPerBlock", 0),
            "max_shared_memory_per_block": props.get("sharedMemPerBlock", 0),
            "warp_size": props.get("warpSize", 32),
        }

    def snapshot_memory(self, label: str = "") -> MemorySnapshot:
        """
        Take a snapshot of current GPU memory state.

        Args:
            label: Optional label for this snapshot

        Returns:
            MemorySnapshot with current state
        """
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_bytes=total,
            used_bytes=total - free,
            free_bytes=free,
            pool_used_bytes=pool.used_bytes(),
            pool_free_bytes=pool.free_bytes(),
        )

        with self._lock:
            key = label or f"snapshot_{len(self._memory_snapshots)}"
            self._memory_snapshots[key] = snapshot

        return snapshot

    def measure_h2d(
        self, host_array: np.ndarray, stream=None, use_pinned: bool = True
    ) -> TransferMeasurement:
        """
        Measure Host-to-Device transfer bandwidth.

        Args:
            host_array: NumPy array to transfer
            stream: Optional CUDA stream
            use_pinned: Whether to use pinned memory staging

        Returns:
            TransferMeasurement with bandwidth info
        """
        size_bytes = host_array.nbytes

        # Ensure pinned memory if requested
        if use_pinned:
            pinned = cp.cuda.alloc_pinned_memory(size_bytes)
            pinned_arr = np.frombuffer(pinned, dtype=host_array.dtype).reshape(host_array.shape)
            np.copyto(pinned_arr, host_array)
            src = pinned_arr
        else:
            src = host_array

        # Create events for timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Allocate destination
        dst = cp.empty_like(host_array)

        # Time the transfer
        if stream is not None:
            start_event.record(stream)
            dst.set(src)  # CuPy H2D transfer
            end_event.record(stream)
            end_event.synchronize()
        else:
            start_event.record()
            dst.set(src)
            end_event.record()
            end_event.synchronize()

        duration_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        measurement = TransferMeasurement(
            direction="h2d",
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            pinned=use_pinned,
            stream_async=stream is not None,
        )

        with self._lock:
            self._transfers.append(measurement)

        return measurement

    def measure_d2h(self, gpu_array, stream=None, use_pinned: bool = True) -> TransferMeasurement:
        """
        Measure Device-to-Host transfer bandwidth.

        Args:
            gpu_array: CuPy array to transfer
            stream: Optional CUDA stream
            use_pinned: Whether to use pinned memory destination

        Returns:
            TransferMeasurement with bandwidth info
        """
        size_bytes = gpu_array.nbytes

        # Prepare destination
        if use_pinned:
            pinned = cp.cuda.alloc_pinned_memory(size_bytes)
            dst = np.frombuffer(pinned, dtype=gpu_array.dtype).reshape(gpu_array.shape)
        else:
            dst = np.empty(gpu_array.shape, dtype=gpu_array.dtype)

        # Create events for timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Time the transfer
        if stream is not None:
            start_event.record(stream)
            # Use memcpyAsync for accurate async timing
            src_ptr = gpu_array.data.ptr
            dst_ptr = dst.ctypes.data
            cp.cuda.runtime.memcpyAsync(
                dst_ptr,
                src_ptr,
                size_bytes,
                cp.cuda.runtime.memcpyDeviceToHost,
                stream.ptr if hasattr(stream, "ptr") else 0,
            )
            end_event.record(stream)
            end_event.synchronize()
        else:
            start_event.record()
            dst[:] = cp.asnumpy(gpu_array)
            end_event.record()
            end_event.synchronize()

        duration_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        measurement = TransferMeasurement(
            direction="d2h",
            size_bytes=size_bytes,
            duration_ms=duration_ms,
            pinned=use_pinned,
            stream_async=stream is not None,
        )

        with self._lock:
            self._transfers.append(measurement)

        return measurement

    def measure_d2h_async_overlap(
        self, gpu_array, compute_func, stream=None
    ) -> tuple[TransferMeasurement, float]:
        """
        Measure D2H transfer with compute overlap.

        Returns both transfer measurement and overlap ratio.

        Args:
            gpu_array: CuPy array to transfer
            compute_func: Function to run on GPU during transfer
            stream: Transfer stream (compute runs on default stream)

        Returns:
            Tuple of (transfer_measurement, overlap_ratio)
        """
        size_bytes = gpu_array.nbytes

        # Create streams
        transfer_stream = stream or cp.cuda.Stream(non_blocking=True)
        compute_stream = cp.cuda.Stream(non_blocking=True)

        # Pinned destination
        pinned = cp.cuda.alloc_pinned_memory(size_bytes)
        dst = np.frombuffer(pinned, dtype=gpu_array.dtype).reshape(gpu_array.shape)

        # Events for timing
        transfer_start = cp.cuda.Event()
        transfer_end = cp.cuda.Event()
        compute_start = cp.cuda.Event()
        compute_end = cp.cuda.Event()

        # Start transfer
        transfer_start.record(transfer_stream)
        src_ptr = gpu_array.data.ptr
        dst_ptr = dst.ctypes.data
        cp.cuda.runtime.memcpyAsync(
            dst_ptr, src_ptr, size_bytes, cp.cuda.runtime.memcpyDeviceToHost, transfer_stream.ptr
        )
        transfer_end.record(transfer_stream)

        # Start compute on different stream
        compute_start.record(compute_stream)
        compute_func()
        compute_end.record(compute_stream)

        # Synchronize both
        transfer_end.synchronize()
        compute_end.synchronize()

        # Calculate times
        transfer_ms = cp.cuda.get_elapsed_time(transfer_start, transfer_end)
        compute_ms = cp.cuda.get_elapsed_time(compute_start, compute_end)

        # Calculate overlap ratio
        # If transfer and compute finish at similar times, good overlap
        # Overlap = min(transfer, compute) / max(transfer, compute)
        if max(transfer_ms, compute_ms) > 0:
            overlap_ratio = min(transfer_ms, compute_ms) / max(transfer_ms, compute_ms)
        else:
            overlap_ratio = 0.0

        measurement = TransferMeasurement(
            direction="d2h",
            size_bytes=size_bytes,
            duration_ms=transfer_ms,
            pinned=True,
            stream_async=True,
        )

        with self._lock:
            self._transfers.append(measurement)

        return measurement, overlap_ratio

    def get_memory_delta(self, label1: str, label2: str) -> int:
        """Get memory difference between two snapshots (bytes)."""
        with self._lock:
            if label1 not in self._memory_snapshots or label2 not in self._memory_snapshots:
                return 0
            return (
                self._memory_snapshots[label2].used_bytes
                - self._memory_snapshots[label1].used_bytes
            )

    def get_transfer_stats(self, direction: str | None = None) -> dict:
        """
        Get aggregate transfer statistics.

        Args:
            direction: Filter by "h2d" or "d2h" (None for all)

        Returns:
            Dict with bandwidth statistics
        """
        with self._lock:
            transfers = self._transfers
            if direction:
                transfers = [t for t in transfers if t.direction == direction]

        if not transfers:
            return {"count": 0}

        bandwidths = [t.bandwidth_gbps for t in transfers]
        sizes = [t.size_bytes for t in transfers]
        durations = [t.duration_ms for t in transfers]

        return {
            "count": len(transfers),
            "total_bytes": sum(sizes),
            "total_gb": sum(sizes) / 1e9,
            "avg_bandwidth_gbps": np.mean(bandwidths),
            "max_bandwidth_gbps": max(bandwidths),
            "min_bandwidth_gbps": min(bandwidths),
            "avg_duration_ms": np.mean(durations),
            "pinned_ratio": sum(1 for t in transfers if t.pinned) / len(transfers),
            "async_ratio": sum(1 for t in transfers if t.stream_async) / len(transfers),
        }

    def get_report(self) -> dict:
        """Get complete profiling report."""
        return {
            "device": self._device_props,
            "memory_snapshots": {
                label: {
                    "used_gb": snap.used_gb,
                    "free_gb": snap.free_gb,
                    "utilization": snap.utilization,
                    "pool_used_mb": snap.pool_used_bytes / 1e6,
                }
                for label, snap in self._memory_snapshots.items()
            },
            "transfers": {
                "h2d": self.get_transfer_stats("h2d"),
                "d2h": self.get_transfer_stats("d2h"),
                "all": self.get_transfer_stats(),
            },
        }

    def print_report(self):
        """Print formatted profiling report."""
        report = self.get_report()

        print("\n" + "=" * 70)
        print("GPU Metrics Report")
        print("=" * 70)

        print(f"\nDevice: {report['device']['name']}")
        print(f"  Compute Capability: {report['device']['compute_capability']}")
        print(f"  Total Memory: {report['device']['total_memory_gb']:.1f} GB")
        print(f"  Multiprocessors: {report['device']['multiprocessor_count']}")

        print("\nMemory Snapshots:")
        for label, snap in report["memory_snapshots"].items():
            print(
                f"  {label}: {snap['used_gb']:.2f} GB used "
                f"({snap['utilization'] * 100:.1f}% utilization)"
            )

        print("\nTransfer Statistics:")
        for direction, stats in report["transfers"].items():
            if direction != "all" and stats["count"] > 0:
                print(f"  {direction.upper()}:")
                print(f"    Count: {stats['count']}")
                print(f"    Avg Bandwidth: {stats['avg_bandwidth_gbps']:.2f} GB/s")
                print(f"    Total Data: {stats['total_gb']:.3f} GB")

        print("=" * 70)

    def reset(self):
        """Reset all collected metrics."""
        with self._lock:
            self._memory_snapshots.clear()
            self._transfers.clear()
            self._kernels.clear()


def measure_pcie_bandwidth() -> dict[str, float]:
    """
    Benchmark PCIe bandwidth with various transfer sizes.

    Returns:
        Dict mapping size to bandwidth in GB/s
    """
    if not CUPY_AVAILABLE:
        return {}

    results = {}
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]  # 1KB to 16MB

    for size in sizes:
        # Create test data
        host_data = np.random.randn(size // 8).astype(np.float64)
        pinned = cp.cuda.alloc_pinned_memory(host_data.nbytes)
        pinned_arr = np.frombuffer(pinned, dtype=np.float64)
        np.copyto(pinned_arr, host_data)

        gpu_data = cp.empty_like(host_data)

        # Warmup
        for _ in range(3):
            gpu_data.set(pinned_arr)
            cp.cuda.Device().synchronize()

        # Measure H2D
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(10):
            gpu_data.set(pinned_arr)
        end.record()
        end.synchronize()
        h2d_ms = cp.cuda.get_elapsed_time(start, end) / 10

        # Measure D2H
        start.record()
        for _ in range(10):
            pinned_arr[:] = cp.asnumpy(gpu_data)
        end.record()
        end.synchronize()
        d2h_ms = cp.cuda.get_elapsed_time(start, end) / 10

        results[f"{size // 1024}KB"] = {
            "h2d_gbps": (size / 1e9) / (h2d_ms / 1000),
            "d2h_gbps": (size / 1e9) / (d2h_ms / 1000),
        }

    return results
