"""
GPU RF Forensics Engine - Pinned Memory Manager

Handles page-locked host memory allocation that enables zero-copy DMA
transfers from SDR hardware to GPU VRAM.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class BufferInfo:
    """Information about an allocated buffer."""

    name: str
    samples: int
    dtype: np.dtype
    pinned_memory: object  # cp.cuda.PinnedMemoryPointer
    numpy_view: np.ndarray
    size_bytes: int
    in_use: bool = False


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_allocated_bytes: int = 0
    peak_allocated_bytes: int = 0
    num_buffers: int = 0
    num_allocations: int = 0
    num_deallocations: int = 0


class PinnedMemoryManager:
    """
    Manages page-locked (pinned) host memory for zero-copy DMA transfers.

    Pinned memory enables asynchronous transfers between host and device,
    which is critical for achieving maximum PCIe bandwidth with SDR data.

    Usage:
        with PinnedMemoryManager() as mm:
            buffer = mm.allocate_buffer("iq_data", 10_000_000)
            # SDR writes directly to buffer
            # Transfer to GPU is automatic via pinned memory
    """

    def __init__(self, default_buffer_samples: int = 10_000_000):
        """
        Initialize the pinned memory manager.

        Args:
            default_buffer_samples: Default number of samples for new buffers.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is required for PinnedMemoryManager")

        self._default_samples = default_buffer_samples
        self._buffers: dict[str, BufferInfo] = {}
        self._stats = MemoryStats()
        self._lock = threading.Lock()

        # Initialize CuPy's pinned memory pool
        self._memory_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self._memory_pool.malloc)

    def allocate_buffer(
        self, name: str, samples: int | None = None, dtype: np.dtype = np.complex64
    ) -> np.ndarray:
        """
        Allocate a named pinned memory buffer.

        Args:
            name: Unique identifier for the buffer.
            samples: Number of samples (uses default if None).
            dtype: NumPy dtype for the buffer (default: complex64).

        Returns:
            NumPy array view into pinned memory.

        Raises:
            ValueError: If buffer name already exists.
            MemoryError: If allocation fails.
        """
        with self._lock:
            if name in self._buffers:
                raise ValueError(
                    f"Buffer '{name}' already exists. Use get_buffer() or release_buffer() first."
                )

            samples = samples or self._default_samples
            dtype = np.dtype(dtype)
            size_bytes = samples * dtype.itemsize

            try:
                # Allocate pinned memory
                pinned_mem = cp.cuda.alloc_pinned_memory(size_bytes)

                # Create NumPy view into pinned memory
                numpy_view = np.frombuffer(pinned_mem, dtype=dtype, count=samples)

                # Store buffer info
                buffer_info = BufferInfo(
                    name=name,
                    samples=samples,
                    dtype=dtype,
                    pinned_memory=pinned_mem,
                    numpy_view=numpy_view,
                    size_bytes=size_bytes,
                    in_use=True,
                )
                self._buffers[name] = buffer_info

                # Update stats
                self._stats.total_allocated_bytes += size_bytes
                self._stats.peak_allocated_bytes = max(
                    self._stats.peak_allocated_bytes, self._stats.total_allocated_bytes
                )
                self._stats.num_buffers += 1
                self._stats.num_allocations += 1

                return numpy_view

            except Exception as e:
                raise MemoryError(f"Failed to allocate pinned memory for '{name}': {e}")

    def get_buffer(self, name: str) -> np.ndarray:
        """
        Get an existing buffer by name.

        Args:
            name: Buffer identifier.

        Returns:
            NumPy array view into the buffer.

        Raises:
            KeyError: If buffer doesn't exist.
        """
        with self._lock:
            if name not in self._buffers:
                raise KeyError(f"Buffer '{name}' not found")
            return self._buffers[name].numpy_view

    def get_buffer_info(self, name: str) -> BufferInfo:
        """Get detailed information about a buffer."""
        with self._lock:
            if name not in self._buffers:
                raise KeyError(f"Buffer '{name}' not found")
            return self._buffers[name]

    def release_buffer(self, name: str) -> None:
        """
        Release a buffer back to the pool.

        Args:
            name: Buffer identifier to release.

        Raises:
            KeyError: If buffer doesn't exist.
        """
        with self._lock:
            if name not in self._buffers:
                raise KeyError(f"Buffer '{name}' not found")

            buffer_info = self._buffers.pop(name)

            # Update stats
            self._stats.total_allocated_bytes -= buffer_info.size_bytes
            self._stats.num_buffers -= 1
            self._stats.num_deallocations += 1

    def get_or_create_buffer(
        self, name: str, samples: int | None = None, dtype: np.dtype = np.complex64
    ) -> np.ndarray:
        """
        Get existing buffer or create new one.

        Args:
            name: Buffer identifier.
            samples: Number of samples (for new buffers).
            dtype: Data type (for new buffers).

        Returns:
            NumPy array view into the buffer.
        """
        with self._lock:
            if name in self._buffers:
                return self._buffers[name].numpy_view

        # Release lock before allocation to avoid deadlock
        return self.allocate_buffer(name, samples, dtype)

    def resize_buffer(self, name: str, new_samples: int) -> np.ndarray:
        """
        Resize an existing buffer.

        Args:
            name: Buffer to resize.
            new_samples: New sample count.

        Returns:
            New NumPy array view.
        """
        with self._lock:
            if name not in self._buffers:
                raise KeyError(f"Buffer '{name}' not found")

            old_info = self._buffers[name]
            dtype = old_info.dtype

        # Release lock, then deallocate and reallocate
        self.release_buffer(name)
        return self.allocate_buffer(name, new_samples, dtype)

    def get_stats(self) -> dict:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory statistics.
        """
        with self._lock:
            return {
                "total_allocated_bytes": self._stats.total_allocated_bytes,
                "total_allocated_mb": self._stats.total_allocated_bytes / (1024 * 1024),
                "peak_allocated_bytes": self._stats.peak_allocated_bytes,
                "peak_allocated_mb": self._stats.peak_allocated_bytes / (1024 * 1024),
                "num_buffers": self._stats.num_buffers,
                "num_allocations": self._stats.num_allocations,
                "num_deallocations": self._stats.num_deallocations,
                "buffer_names": list(self._buffers.keys()),
            }

    def list_buffers(self) -> list[str]:
        """List all allocated buffer names."""
        with self._lock:
            return list(self._buffers.keys())

    def clear_all(self) -> None:
        """Release all allocated buffers."""
        with self._lock:
            buffer_names = list(self._buffers.keys())

        for name in buffer_names:
            self.release_buffer(name)

    def __enter__(self) -> PinnedMemoryManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - clean up all buffers."""
        self.clear_all()

    def __del__(self):
        """Destructor - ensure cleanup."""
        try:
            self.clear_all()
        except Exception:
            pass


class BufferPool:
    """
    Pool of pre-allocated pinned memory buffers for recycling.

    Avoids allocation overhead by recycling buffers of the same size.
    """

    def __init__(self, num_buffers: int, samples_per_buffer: int, dtype: np.dtype = np.complex64):
        """
        Initialize buffer pool.

        Args:
            num_buffers: Number of buffers to pre-allocate.
            samples_per_buffer: Samples in each buffer.
            dtype: Data type for buffers.
        """
        self._manager = PinnedMemoryManager()
        self._samples = samples_per_buffer
        self._dtype = dtype
        self._available: list[str] = []
        self._in_use: set[str] = set()
        self._lock = threading.Lock()

        # Pre-allocate buffers
        for i in range(num_buffers):
            name = f"pool_buffer_{i}"
            self._manager.allocate_buffer(name, samples_per_buffer, dtype)
            self._available.append(name)

    def acquire(self) -> tuple[str, np.ndarray]:
        """
        Acquire a buffer from the pool.

        Returns:
            Tuple of (buffer_name, numpy_array).

        Raises:
            RuntimeError: If no buffers available.
        """
        with self._lock:
            if not self._available:
                raise RuntimeError("No buffers available in pool")

            name = self._available.pop()
            self._in_use.add(name)
            return name, self._manager.get_buffer(name)

    def release(self, name: str) -> None:
        """
        Return a buffer to the pool.

        Args:
            name: Buffer name to release.
        """
        with self._lock:
            if name not in self._in_use:
                raise ValueError(f"Buffer '{name}' not in use")

            self._in_use.remove(name)
            self._available.append(name)

    def available_count(self) -> int:
        """Number of available buffers."""
        with self._lock:
            return len(self._available)

    def in_use_count(self) -> int:
        """Number of buffers in use."""
        with self._lock:
            return len(self._in_use)

    def __enter__(self) -> BufferPool:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._manager.clear_all()
