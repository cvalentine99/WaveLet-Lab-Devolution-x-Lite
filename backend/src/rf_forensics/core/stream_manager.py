"""
GPU RF Forensics Engine - CUDA Stream Manager

Coordinates asynchronous memory transfers and compute operations to maximize
GPU utilization through concurrent execution.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class StreamPriority(Enum):
    """CUDA stream priority levels."""

    HIGH = -1
    NORMAL = 0
    LOW = 1


@dataclass
class StreamTiming:
    """Timing information for a stream operation."""

    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None


@dataclass
class StreamStats:
    """Statistics for stream operations."""

    total_h2d_transfers: int = 0
    total_d2h_transfers: int = 0
    total_compute_ops: int = 0
    total_h2d_bytes: int = 0
    total_d2h_bytes: int = 0
    avg_h2d_bandwidth_gbps: float = 0.0
    avg_d2h_bandwidth_gbps: float = 0.0


class CUDAStreamManager:
    """
    Manages CUDA streams for asynchronous GPU operations.

    Provides separate streams for:
    - Host-to-Device transfers (stream_h2d)
    - Device-to-Host transfers (stream_d2h)
    - Compute operations (stream_compute)
    - Auxiliary operations (stream_aux)

    Supports triple-buffering coordination for continuous SDR streaming:
    - Buffer N: Being filled by SDR
    - Buffer N-1: Being transferred to GPU
    - Buffer N-2: Being processed on GPU
    """

    def __init__(self, enable_timing: bool = True):
        """
        Initialize the CUDA stream manager.

        Args:
            enable_timing: Whether to record timing metrics.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for CUDAStreamManager")

        self._enable_timing = enable_timing
        self._lock = threading.Lock()

        # Create dedicated streams
        self._streams = {
            "h2d": cp.cuda.Stream(non_blocking=True),
            "d2h": cp.cuda.Stream(non_blocking=True),
            "compute": cp.cuda.Stream(non_blocking=True),
            "aux": cp.cuda.Stream(non_blocking=True),
        }

        # Event pool for synchronization
        self._events: dict[str, cp.cuda.Event] = {}

        # Timing records
        self._timings: list[StreamTiming] = []
        self._stats = StreamStats()

        # Triple buffer state
        self._triple_buffer_idx = 0

    def get_stream(self, name: str) -> cp.cuda.Stream:
        """
        Get a stream by name.

        Args:
            name: Stream name (h2d, d2h, compute, aux).

        Returns:
            CUDA stream object.
        """
        if name not in self._streams:
            raise KeyError(f"Unknown stream: {name}")
        return self._streams[name]

    def record_event(self, stream_name: str, event_name: str | None = None) -> cp.cuda.Event:
        """
        Record an event on a stream.

        Args:
            stream_name: Stream to record event on.
            event_name: Optional name for the event.

        Returns:
            CUDA event object.
        """
        stream = self.get_stream(stream_name)
        event = cp.cuda.Event()
        event.record(stream)

        if event_name:
            with self._lock:
                self._events[event_name] = event

        return event

    def wait_event(self, stream_name: str, event: cp.cuda.Event) -> None:
        """
        Make a stream wait for an event.

        Args:
            stream_name: Stream that should wait.
            event: Event to wait for.
        """
        stream = self.get_stream(stream_name)
        stream.wait_event(event)

    def wait_named_event(self, stream_name: str, event_name: str) -> None:
        """
        Make a stream wait for a named event.

        Args:
            stream_name: Stream that should wait.
            event_name: Name of the event to wait for.
        """
        with self._lock:
            if event_name not in self._events:
                raise KeyError(f"Event '{event_name}' not found")
            event = self._events[event_name]

        self.wait_event(stream_name, event)

    def async_copy_h2d(
        self, host_buffer: np.ndarray, device_buffer: cp.ndarray, stream_name: str = "h2d"
    ) -> cp.cuda.Event:
        """
        Asynchronously copy data from host to device.

        Args:
            host_buffer: Source NumPy array (should be pinned memory).
            device_buffer: Destination CuPy array.
            stream_name: Stream to use for transfer.

        Returns:
            Event marking completion of transfer.
        """
        stream = self.get_stream(stream_name)

        start_time = time.perf_counter()

        with stream:
            device_buffer.set(host_buffer)

        event = self.record_event(stream_name)

        # Update stats
        with self._lock:
            self._stats.total_h2d_transfers += 1
            self._stats.total_h2d_bytes += host_buffer.nbytes

            if self._enable_timing:
                self._timings.append(
                    StreamTiming(
                        name=f"h2d_{self._stats.total_h2d_transfers}", start_time=start_time
                    )
                )

        return event

    def async_copy_d2h(
        self, device_buffer: cp.ndarray, host_buffer: np.ndarray, stream_name: str = "d2h"
    ) -> cp.cuda.Event:
        """
        Asynchronously copy data from device to host.

        Args:
            device_buffer: Source CuPy array.
            host_buffer: Destination NumPy array (should be pinned memory).
            stream_name: Stream to use for transfer.

        Returns:
            Event marking completion of transfer.
        """
        stream = self.get_stream(stream_name)

        start_time = time.perf_counter()

        with stream:
            device_buffer.get(out=host_buffer)

        event = self.record_event(stream_name)

        # Update stats
        with self._lock:
            self._stats.total_d2h_transfers += 1
            self._stats.total_d2h_bytes += device_buffer.nbytes

            if self._enable_timing:
                self._timings.append(
                    StreamTiming(
                        name=f"d2h_{self._stats.total_d2h_transfers}", start_time=start_time
                    )
                )

        return event

    def async_compute(
        self, func: Callable, *args, stream_name: str = "compute", **kwargs
    ) -> cp.cuda.Event:
        """
        Execute a compute function asynchronously on a stream.

        Args:
            func: Function to execute.
            *args: Positional arguments for func.
            stream_name: Stream to use.
            **kwargs: Keyword arguments for func.

        Returns:
            Event marking completion.
        """
        stream = self.get_stream(stream_name)

        with stream:
            func(*args, **kwargs)

        event = self.record_event(stream_name)

        with self._lock:
            self._stats.total_compute_ops += 1

        return event

    def synchronize_stream(self, stream_name: str) -> None:
        """
        Synchronize a specific stream.

        Args:
            stream_name: Stream to synchronize.
        """
        stream = self.get_stream(stream_name)
        stream.synchronize()

    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        for stream in self._streams.values():
            stream.synchronize()

    def get_timing_stats(self) -> dict:
        """
        Get timing statistics for stream operations.

        Returns:
            Dictionary with timing statistics.
        """
        with self._lock:
            # Calculate bandwidth stats
            h2d_gbps = 0.0
            d2h_gbps = 0.0

            if self._stats.total_h2d_transfers > 0:
                # Rough estimate - would need actual timing for accuracy
                h2d_gbps = self._stats.total_h2d_bytes / (1024**3)

            if self._stats.total_d2h_transfers > 0:
                d2h_gbps = self._stats.total_d2h_bytes / (1024**3)

            return {
                "total_h2d_transfers": self._stats.total_h2d_transfers,
                "total_d2h_transfers": self._stats.total_d2h_transfers,
                "total_compute_ops": self._stats.total_compute_ops,
                "total_h2d_bytes": self._stats.total_h2d_bytes,
                "total_d2h_bytes": self._stats.total_d2h_bytes,
                "total_h2d_gb": self._stats.total_h2d_bytes / (1024**3),
                "total_d2h_gb": self._stats.total_d2h_bytes / (1024**3),
            }

    def reset_stats(self) -> None:
        """Reset all timing statistics."""
        with self._lock:
            self._stats = StreamStats()
            self._timings.clear()

    def cleanup(self) -> None:
        """
        Cleanup all CUDA streams and resources.

        Synchronizes all streams before clearing to ensure pending
        operations complete.
        """
        with self._lock:
            # Synchronize all streams first
            for stream in self._streams.values():
                stream.synchronize()

            # Clear streams dict (allows GC)
            self._streams.clear()

            # Clear events
            self._events.clear()

            # Reset stats
            self._stats = StreamStats()
            self._timings.clear()

    def __enter__(self) -> CUDAStreamManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()


class StreamContext:
    """Context manager for executing operations on a specific stream."""

    def __init__(self, manager: CUDAStreamManager, stream_name: str):
        self._manager = manager
        self._stream_name = stream_name
        self._stream = manager.get_stream(stream_name)

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream.__exit__(exc_type, exc_val, exc_tb)
