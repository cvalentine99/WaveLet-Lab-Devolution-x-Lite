"""
GPU RF Forensics Engine - Async GPU Transfer Manager

Non-blocking GPU↔CPU transfer manager for pipeline optimization.
Uses double-buffering and dedicated CUDA streams to overlap
transfers with GPU compute.
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np


class AsyncD2HTransfer:
    """
    Non-blocking GPU→CPU (D2H) transfer manager with double-buffering.

    This manager enables pipelining by:
    1. Starting async transfer of current frame's results
    2. Immediately returning previous frame's results
    3. Using dedicated CUDA stream for transfers

    The one-frame latency is acceptable because peak detection
    has no inter-frame dependencies.

    Usage:
        transfer = AsyncD2HTransfer(max_size=1024)

        # In processing loop:
        result = transfer.transfer(gpu_array)
        if result is not None:
            # Process previous frame's data
            process(result)

        # After loop ends:
        final = transfer.flush()
        if final is not None:
            process(final)
    """

    def __init__(
        self,
        max_size: int = 65536,
        dtype: np.dtype = np.float32,
        stream: cp.cuda.Stream | None = None,
    ):
        """
        Initialize async D2H transfer manager.

        Args:
            max_size: Maximum array size to transfer.
            dtype: Data type of arrays.
            stream: CUDA stream for transfers (creates new if None).
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("AsyncD2HTransfer requires CuPy")

        self._max_size = max_size
        self._dtype = np.dtype(dtype)

        # Create dedicated D2H stream if not provided
        self._stream = stream or cp.cuda.Stream(non_blocking=True)
        self._owns_stream = stream is None

        # Double-buffer: ping-pong between two pinned host buffers
        # Buffer A: receiving current transfer
        # Buffer B: available for reading (previous transfer)
        size_bytes = max_size * self._dtype.itemsize
        self._pinned_a = cp.cuda.alloc_pinned_memory(size_bytes)
        self._pinned_b = cp.cuda.alloc_pinned_memory(size_bytes)
        self._buffer_a = np.frombuffer(self._pinned_a, dtype=self._dtype, count=max_size)
        self._buffer_b = np.frombuffer(self._pinned_b, dtype=self._dtype, count=max_size)

        # Track which buffer is currently being filled (A) vs ready (B)
        self._current_is_a = True  # A is filling, B is ready
        self._pending_size = 0  # Size of data in filling buffer
        self._ready_size = 0  # Size of data in ready buffer
        self._has_pending = False  # Whether a transfer is in progress

        # Event for checking transfer completion
        self._transfer_event: cp.cuda.Event | None = None

    def transfer(self, gpu_array: cp.ndarray) -> np.ndarray | None:
        """
        Start async transfer and return previous frame's result.

        This method:
        1. Waits for any in-progress transfer to complete
        2. Swaps buffers (previous becomes ready, current becomes filling)
        3. Starts TRUE async D2H transfer using memcpyAsync
        4. Returns the now-ready previous data

        Args:
            gpu_array: GPU array to transfer (will be copied).

        Returns:
            Previous frame's data, or None if this is the first transfer.
        """
        size = len(gpu_array)
        if size > self._max_size:
            raise ValueError(f"Array size {size} exceeds max {self._max_size}")

        # Ensure previous transfer is complete before swapping
        if self._has_pending and self._transfer_event is not None:
            self._transfer_event.synchronize()

        # Get result from ready buffer (previous frame)
        result = None
        if self._has_pending:
            ready_buffer = self._buffer_b if self._current_is_a else self._buffer_a
            result = ready_buffer[: self._ready_size].copy()

        # Swap buffers
        self._current_is_a = not self._current_is_a
        self._ready_size = self._pending_size

        # Start TRUE async transfer to pinned buffer using memcpyAsync
        filling_buffer = self._buffer_a if self._current_is_a else self._buffer_b

        # Get raw pointers for async memcpy
        src_ptr = gpu_array.data.ptr
        dst_ptr = filling_buffer.ctypes.data
        nbytes = size * self._dtype.itemsize

        # Async D2H copy on dedicated stream (does NOT block host)
        cp.cuda.runtime.memcpyAsync(
            dst_ptr,  # destination (pinned host)
            src_ptr,  # source (device)
            nbytes,  # size in bytes
            cp.cuda.runtime.memcpyDeviceToHost,  # direction
            self._stream.ptr,  # stream
        )

        # Record event for synchronization check
        self._transfer_event = cp.cuda.Event()
        self._transfer_event.record(self._stream)

        self._pending_size = size
        self._has_pending = True

        return result

    def flush(self) -> np.ndarray | None:
        """
        Flush any pending transfer and return final result.

        Call this after the processing loop ends to get the last frame's data.

        Returns:
            Final transferred data, or None if no pending transfer.
        """
        if not self._has_pending:
            return None

        # Wait for final transfer to complete
        if self._transfer_event is not None:
            self._transfer_event.synchronize()

        # Return data from the filling buffer (last transfer)
        filling_buffer = self._buffer_a if self._current_is_a else self._buffer_b
        return filling_buffer[: self._pending_size].copy()

    def cleanup(self):
        """Release all resources."""
        # Ensure any pending transfer is complete
        if self._transfer_event is not None:
            try:
                self._transfer_event.synchronize()
            except Exception:
                pass

        # Clear references to allow garbage collection
        self._buffer_a = None
        self._buffer_b = None
        self._pinned_a = None
        self._pinned_b = None
        self._transfer_event = None

        # Only delete stream if we created it
        if self._owns_stream and self._stream is not None:
            self._stream = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class AsyncTransferManager:
    """
    Manages multiple async D2H transfer channels.

    Provides named transfer channels for different data types
    (e.g., "psd" for PSD results, "cfar" for CFAR mask).

    Usage:
        manager = AsyncTransferManager()
        manager.create_channel("psd", max_size=1024, dtype=np.float32)
        manager.create_channel("cfar", max_size=1024, dtype=np.bool_)

        # In loop:
        psd_result = manager.transfer("psd", gpu_psd)
        cfar_result = manager.transfer("cfar", gpu_cfar)
    """

    def __init__(self, shared_stream: cp.cuda.Stream | None = None):
        """
        Initialize transfer manager.

        Args:
            shared_stream: Shared CUDA stream for all channels.
                          Creates per-channel streams if None.
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError("AsyncTransferManager requires CuPy")

        self._shared_stream = shared_stream
        self._channels: dict[str, AsyncD2HTransfer] = {}
        self._lock = threading.Lock()

    def create_channel(
        self, name: str, max_size: int, dtype: np.dtype = np.float32
    ) -> AsyncD2HTransfer:
        """
        Create a named transfer channel.

        Args:
            name: Channel name.
            max_size: Maximum array size for this channel.
            dtype: Data type of arrays.

        Returns:
            The created transfer channel.
        """
        with self._lock:
            if name in self._channels:
                raise ValueError(f"Channel '{name}' already exists")

            channel = AsyncD2HTransfer(max_size=max_size, dtype=dtype, stream=self._shared_stream)
            self._channels[name] = channel
            return channel

    def get_channel(self, name: str) -> AsyncD2HTransfer | None:
        """Get a channel by name."""
        return self._channels.get(name)

    def transfer(self, name: str, gpu_array: cp.ndarray) -> np.ndarray | None:
        """
        Transfer data on a named channel.

        Args:
            name: Channel name.
            gpu_array: GPU array to transfer.

        Returns:
            Previous frame's data, or None.
        """
        channel = self._channels.get(name)
        if channel is None:
            raise KeyError(f"Unknown channel: {name}")
        return channel.transfer(gpu_array)

    def flush_all(self) -> dict[str, np.ndarray | None]:
        """
        Flush all channels and return final results.

        Returns:
            Dict of channel name → final data.
        """
        results = {}
        for name, channel in self._channels.items():
            results[name] = channel.flush()
        return results

    def cleanup(self):
        """Release all resources."""
        with self._lock:
            for channel in self._channels.values():
                channel.cleanup()
            self._channels.clear()
            self._shared_stream = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
