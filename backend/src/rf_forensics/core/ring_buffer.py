"""
GPU RF Forensics Engine - Ring Buffer

Circular buffer implementation for continuous SDR data streaming with
GPU acceleration support.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class SegmentState(Enum):
    """State of a ring buffer segment."""

    EMPTY = 0  # Ready to be filled
    FILLING = 1  # Currently being written by producer
    READY = 2  # Filled and ready for processing
    PROCESSING = 3  # Currently being processed by consumer


@dataclass
class SegmentInfo:
    """Information about a buffer segment."""

    index: int
    state: SegmentState
    samples_written: int
    timestamp: float | None = None


class GPURingBuffer:
    """
    GPU-accelerated circular buffer for continuous data streaming.

    Implements a producer-consumer pattern optimized for SDR data:
    - Producer (SDR callback): Writes to current segment in pinned memory
    - Consumer (GPU pipeline): Reads oldest ready segment

    Features:
    - Configurable number of segments
    - Automatic segment state management
    - Thread-safe operations
    - Overflow handling (drop oldest or block)
    - Support for both pinned host memory and GPU device memory

    Usage:
        ring = GPURingBuffer(num_segments=4, samples_per_segment=1_000_000)

        # Producer thread (SDR callback)
        idx, buffer = ring.get_write_segment()
        buffer[:] = sdr_data
        ring.mark_segment_ready(idx)

        # Consumer thread (GPU processing)
        idx, gpu_buffer = ring.get_read_segment()
        process(gpu_buffer)
        ring.mark_segment_processed(idx)
    """

    def __init__(
        self,
        num_segments: int = 4,
        samples_per_segment: int = 1_000_000,
        dtype: np.dtype = np.complex64,
        overflow_policy: str = "drop_oldest",
        use_gpu_buffer: bool = True,
    ):
        """
        Initialize the ring buffer.

        Args:
            num_segments: Number of buffer segments.
            samples_per_segment: Samples in each segment.
            dtype: Data type for samples.
            overflow_policy: How to handle overflow ("drop_oldest" or "block").
            use_gpu_buffer: Whether to maintain GPU-side buffers.
        """
        if num_segments < 2:
            raise ValueError("num_segments must be >= 2")

        self._num_segments = num_segments
        self._samples_per_segment = samples_per_segment
        self._dtype = np.dtype(dtype)
        self._overflow_policy = overflow_policy
        self._use_gpu_buffer = use_gpu_buffer and CUPY_AVAILABLE

        # Segment state tracking
        self._states = [SegmentState.EMPTY] * num_segments
        self._samples_written = [0] * num_segments
        self._timestamps = [None] * num_segments

        # Write/read positions
        self._write_idx = 0
        self._read_idx = 0

        # Statistics
        self._total_written = 0
        self._total_read = 0
        self._overflow_count = 0
        self._underflow_count = 0

        # Backpressure tracking
        self._backpressure_events = 0
        self._backpressure_threshold = 0.75  # Signal backpressure at 75% fill

        # Threading
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

        # Allocate host buffers (pinned memory)
        self._host_buffers = []
        if CUPY_AVAILABLE:
            for i in range(num_segments):
                size_bytes = samples_per_segment * self._dtype.itemsize
                pinned_mem = cp.cuda.alloc_pinned_memory(size_bytes)
                buffer = np.frombuffer(pinned_mem, dtype=self._dtype, count=samples_per_segment)
                self._host_buffers.append(buffer)
        else:
            for i in range(num_segments):
                buffer = np.zeros(samples_per_segment, dtype=self._dtype)
                self._host_buffers.append(buffer)

        # Allocate GPU buffers if requested
        self._gpu_buffers = []
        if self._use_gpu_buffer:
            for i in range(num_segments):
                gpu_buffer = cp.zeros(samples_per_segment, dtype=self._dtype)
                self._gpu_buffers.append(gpu_buffer)

    @property
    def num_segments(self) -> int:
        """Number of buffer segments."""
        return self._num_segments

    @property
    def samples_per_segment(self) -> int:
        """Samples per segment."""
        return self._samples_per_segment

    @property
    def dtype(self) -> np.dtype:
        """Data type of samples."""
        return self._dtype

    def get_write_segment(self, timeout: float | None = None) -> tuple[int, np.ndarray]:
        """
        Get the next segment for writing.

        Args:
            timeout: Maximum time to wait (None = use default 5.0s to prevent deadlock).

        Returns:
            Tuple of (segment_index, host_buffer).

        Raises:
            TimeoutError: If timeout expires with no available segment.
            RuntimeError: If overflow occurs with drop_oldest policy.
        """
        # Default timeout to prevent indefinite blocking (deadlock prevention)
        if timeout is None:
            timeout = 5.0

        with self._not_full:
            start_time = None
            remaining_timeout = timeout

            # Wait for an empty segment
            while self._states[self._write_idx] != SegmentState.EMPTY:
                if self._overflow_policy == "drop_oldest":
                    # Try to drop oldest READY segment
                    dropped = self._drop_oldest_ready()
                    if dropped:
                        continue

                    # Try to reclaim stale FILLING segments (deadlock prevention)
                    # A segment stuck in FILLING for too long is likely abandoned
                    reclaimed = self._reclaim_stale_filling()
                    if reclaimed:
                        continue

                    # All segments busy - wait with timeout
                    if not self._not_full.wait(remaining_timeout):
                        raise TimeoutError("Timeout waiting for write segment")
                else:
                    # Block until segment becomes available
                    if not self._not_full.wait(remaining_timeout):
                        raise TimeoutError("Timeout waiting for write segment")

            # Mark segment as filling
            idx = self._write_idx
            self._states[idx] = SegmentState.FILLING
            self._samples_written[idx] = 0

            return idx, self._host_buffers[idx]

    def try_get_write_segment(self) -> tuple[int, np.ndarray] | None:
        """
        Non-blocking attempt to get a write segment.

        This method NEVER blocks - ideal for SDR callbacks that must return quickly.
        If no segment is available, it tries to drop the oldest ready segment.
        If that fails, returns None immediately.

        Also tracks backpressure events when buffer fill exceeds threshold.

        Returns:
            Tuple of (segment_index, host_buffer) if available, None otherwise.
        """
        with self._not_full:
            # Check fill level for backpressure tracking
            ready_count = sum(1 for s in self._states if s == SegmentState.READY)
            fill_level = ready_count / self._num_segments

            if fill_level > self._backpressure_threshold:
                self._backpressure_events += 1

            # Check if current write segment is available
            if self._states[self._write_idx] == SegmentState.EMPTY:
                idx = self._write_idx
                self._states[idx] = SegmentState.FILLING
                self._samples_written[idx] = 0
                return idx, self._host_buffers[idx]

            # Try to drop oldest ready segment (with drop_oldest policy)
            if self._overflow_policy == "drop_oldest":
                if self._drop_oldest_ready():
                    # Check again after dropping
                    if self._states[self._write_idx] == SegmentState.EMPTY:
                        idx = self._write_idx
                        self._states[idx] = SegmentState.FILLING
                        self._samples_written[idx] = 0
                        return idx, self._host_buffers[idx]

                # Last resort: reclaim stale filling segment
                if self._reclaim_stale_filling():
                    if self._states[self._write_idx] == SegmentState.EMPTY:
                        idx = self._write_idx
                        self._states[idx] = SegmentState.FILLING
                        self._samples_written[idx] = 0
                        return idx, self._host_buffers[idx]

            # No segment available - return immediately without blocking
            self._overflow_count += 1
            return None

    def mark_segment_ready(
        self, segment_id: int, samples_written: int | None = None, timestamp: float | None = None
    ) -> None:
        """
        Mark a segment as ready for processing.

        Args:
            segment_id: Segment index.
            samples_written: Number of samples written (default: full segment).
            timestamp: Optional timestamp for the data.
        """
        with self._not_empty:
            if self._states[segment_id] != SegmentState.FILLING:
                raise RuntimeError(f"Segment {segment_id} is not in FILLING state")

            self._states[segment_id] = SegmentState.READY
            self._samples_written[segment_id] = samples_written or self._samples_per_segment
            self._timestamps[segment_id] = timestamp

            # Advance write pointer
            self._write_idx = (self._write_idx + 1) % self._num_segments
            self._total_written += 1

            # Notify consumers
            self._not_empty.notify_all()

    def get_read_segment(
        self,
        timeout: float | None = None,
        transfer_to_gpu: bool = True,
        stream: cp.cuda.Stream | None = None,
    ) -> tuple[int, np.ndarray]:
        """
        Get the next segment for reading/processing.

        Args:
            timeout: Maximum time to wait (None = use default 5.0s to prevent deadlock).
            transfer_to_gpu: Whether to transfer data to GPU buffer.
            stream: Optional CUDA stream for async transfer. If None, transfer is synchronous.

        Returns:
            Tuple of (segment_index, buffer).
            Buffer is GPU array if transfer_to_gpu=True, else host array.

        Raises:
            TimeoutError: If timeout expires with no ready segment.
        """
        # Default timeout to prevent indefinite blocking (deadlock prevention)
        if timeout is None:
            timeout = 5.0

        with self._not_empty:
            # Wait for a ready segment
            while self._states[self._read_idx] != SegmentState.READY:
                if not self._not_empty.wait(timeout):
                    self._underflow_count += 1
                    raise TimeoutError("Timeout waiting for read segment")

            # Mark segment as processing
            idx = self._read_idx
            self._states[idx] = SegmentState.PROCESSING

        # Transfer to GPU outside the lock
        if transfer_to_gpu and self._use_gpu_buffer:
            # Only transfer the actual samples written (not full buffer)
            actual_count = self._samples_written[idx]
            if actual_count <= 0:
                actual_count = self._samples_per_segment  # Fallback to full segment

            if stream is not None and CUPY_AVAILABLE:
                # Async transfer on provided stream (non-blocking)
                with stream:
                    self._gpu_buffers[idx][:actual_count].set(
                        self._host_buffers[idx][:actual_count]
                    )
            else:
                # Synchronous transfer (blocking)
                self._gpu_buffers[idx][:actual_count].set(self._host_buffers[idx][:actual_count])
            return idx, self._gpu_buffers[idx][:actual_count]
        else:
            actual_count = self._samples_written[idx]
            if actual_count <= 0:
                actual_count = self._samples_per_segment
            return idx, self._host_buffers[idx][:actual_count]

    def get_read_segment_async(
        self, stream: cp.cuda.Stream, timeout: float | None = None
    ) -> tuple[int, np.ndarray, cp.cuda.Event] | None:
        """
        Get segment with async GPU transfer, returning event for synchronization.

        This enables true async pipeline:
        1. Start transfer on stream
        2. Return immediately with event
        3. Caller can synchronize later when data is needed

        Args:
            stream: CUDA stream for async transfer.
            timeout: Maximum time to wait for ready segment.

        Returns:
            Tuple of (segment_index, gpu_buffer, completion_event) or None if timeout.
        """
        if not CUPY_AVAILABLE or not self._use_gpu_buffer:
            raise RuntimeError("Async transfer requires CuPy and GPU buffers")

        # Default timeout
        if timeout is None:
            timeout = 5.0

        with self._not_empty:
            # Wait for a ready segment
            while self._states[self._read_idx] != SegmentState.READY:
                if not self._not_empty.wait(timeout):
                    self._underflow_count += 1
                    return None

            # Mark segment as processing
            idx = self._read_idx
            self._states[idx] = SegmentState.PROCESSING

        # Async transfer on provided stream (only actual samples)
        actual_count = self._samples_written[idx]
        if actual_count <= 0:
            actual_count = self._samples_per_segment

        with stream:
            self._gpu_buffers[idx][:actual_count].set(self._host_buffers[idx][:actual_count])

        # Record event for synchronization
        event = cp.cuda.Event()
        event.record(stream)

        return idx, self._gpu_buffers[idx][:actual_count], event

    def mark_segment_processed(self, segment_id: int) -> None:
        """
        Mark a segment as processed and available for reuse.

        Args:
            segment_id: Segment index.
        """
        with self._not_full:
            if self._states[segment_id] != SegmentState.PROCESSING:
                raise RuntimeError(f"Segment {segment_id} is not in PROCESSING state")

            self._states[segment_id] = SegmentState.EMPTY
            self._read_idx = (self._read_idx + 1) % self._num_segments
            self._total_read += 1

            # Notify producers
            self._not_full.notify_all()

    def _drop_oldest_ready(self) -> bool:
        """
        Drop the oldest READY segment.

        Returns:
            True if a segment was dropped, False otherwise.
        """
        # Find oldest READY segment
        for i in range(self._num_segments):
            idx = (self._read_idx + i) % self._num_segments
            if self._states[idx] == SegmentState.READY:
                self._states[idx] = SegmentState.EMPTY
                self._overflow_count += 1
                self._read_idx = (idx + 1) % self._num_segments
                # Notify waiting producers that a segment is now available
                self._not_full.notify_all()
                return True
        return False

    def _reclaim_stale_filling(self) -> bool:
        """
        Reclaim a segment stuck in FILLING state (deadlock prevention).

        This handles the case where a producer crashed or was interrupted
        while filling a segment, leaving it permanently stuck.

        Returns:
            True if a segment was reclaimed, False otherwise.
        """
        # Find any FILLING segment and force it back to EMPTY
        # This is a last resort to prevent deadlock
        for i in range(self._num_segments):
            idx = (self._write_idx + i) % self._num_segments
            if self._states[idx] == SegmentState.FILLING:
                self._states[idx] = SegmentState.EMPTY
                self._overflow_count += 1
                # Notify waiting threads
                self._not_full.notify_all()
                return True
        return False

    def peek_ready_count(self) -> int:
        """Get count of segments ready for processing."""
        with self._lock:
            return sum(1 for s in self._states if s == SegmentState.READY)

    def peek_empty_count(self) -> int:
        """Get count of empty segments."""
        with self._lock:
            return sum(1 for s in self._states if s == SegmentState.EMPTY)

    def get_status(self) -> dict:
        """
        Get detailed buffer status.

        Returns:
            Dictionary with buffer statistics.
        """
        with self._lock:
            state_counts = {
                "empty": sum(1 for s in self._states if s == SegmentState.EMPTY),
                "filling": sum(1 for s in self._states if s == SegmentState.FILLING),
                "ready": sum(1 for s in self._states if s == SegmentState.READY),
                "processing": sum(1 for s in self._states if s == SegmentState.PROCESSING),
            }

            return {
                "num_segments": self._num_segments,
                "samples_per_segment": self._samples_per_segment,
                "dtype": str(self._dtype),
                "write_idx": self._write_idx,
                "read_idx": self._read_idx,
                "total_written": self._total_written,
                "total_read": self._total_read,
                "overflow_count": self._overflow_count,
                "underflow_count": self._underflow_count,
                "backpressure_events": self._backpressure_events,
                "fill_level": state_counts["ready"] / self._num_segments,
                "segment_states": state_counts,
                "individual_states": [s.name for s in self._states],
            }

    def get_segment_info(self, segment_id: int) -> SegmentInfo:
        """Get information about a specific segment."""
        with self._lock:
            return SegmentInfo(
                index=segment_id,
                state=self._states[segment_id],
                samples_written=self._samples_written[segment_id],
                timestamp=self._timestamps[segment_id],
            )

    def flush(self) -> int:
        """
        Flush all ready segments (mark as empty).

        Returns:
            Number of segments flushed.
        """
        with self._lock:
            flushed = 0
            for i in range(self._num_segments):
                if self._states[i] == SegmentState.READY:
                    self._states[i] = SegmentState.EMPTY
                    flushed += 1
            self._read_idx = self._write_idx
            return flushed

    def reset(self) -> None:
        """Reset the buffer to initial state."""
        with self._lock:
            self._states = [SegmentState.EMPTY] * self._num_segments
            self._samples_written = [0] * self._num_segments
            self._timestamps = [None] * self._num_segments
            self._write_idx = 0
            self._read_idx = 0
            self._total_written = 0
            self._total_read = 0
            self._overflow_count = 0
            self._underflow_count = 0
            self._backpressure_events = 0

    def __len__(self) -> int:
        """Number of ready segments."""
        return self.peek_ready_count()

    def cleanup(self) -> None:
        """
        Release all buffer resources.

        Should be called when the ring buffer is no longer needed to
        free GPU memory and pinned host memory.
        """
        with self._lock:
            # Delete GPU buffer references to allow garbage collection
            for gpu_buf in self._gpu_buffers:
                del gpu_buf
            self._gpu_buffers.clear()

            # Delete host buffer references (pinned memory freed by GC)
            for host_buf in self._host_buffers:
                del host_buf
            self._host_buffers.clear()

            # Force memory pool cleanup if CuPy available
            if CUPY_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Reset state
            self._states = [SegmentState.EMPTY] * self._num_segments
            self._samples_written = [0] * self._num_segments
            self._timestamps = [None] * self._num_segments
            self._write_idx = 0
            self._read_idx = 0

    def __del__(self):
        """Destructor - ensure cleanup is called."""
        try:
            self.cleanup()
        except Exception:
            pass

    def __enter__(self) -> GPURingBuffer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
