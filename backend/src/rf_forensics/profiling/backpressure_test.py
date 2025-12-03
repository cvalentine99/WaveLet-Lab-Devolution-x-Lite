"""
Backpressure Stress Test for RF Forensics Pipeline

Tests the pipeline's behavior under load:
- Ring buffer fill dynamics
- SDR throttling activation
- Drop rate measurement
- Recovery time analysis
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class BackpressureEvent:
    """Record of a backpressure event."""

    timestamp: float
    event_type: str  # "throttle_start", "throttle_end", "drop"
    buffer_fill: float
    dropped_samples: int = 0


@dataclass
class BackpressureTestResult:
    """Results from backpressure stress test."""

    duration_seconds: float
    total_samples_generated: int
    total_samples_processed: int
    total_samples_dropped: int
    drop_rate: float
    max_buffer_fill: float
    throttle_activations: int
    avg_throttle_duration_ms: float
    recovery_time_ms: float  # Time from max fill to stable
    events: list[BackpressureEvent] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Backpressure Test Results:\n"
            f"  Duration: {self.duration_seconds:.1f}s\n"
            f"  Samples: {self.total_samples_processed:,} processed, "
            f"{self.total_samples_dropped:,} dropped\n"
            f"  Drop Rate: {self.drop_rate * 100:.2f}%\n"
            f"  Max Buffer Fill: {self.max_buffer_fill * 100:.1f}%\n"
            f"  Throttle Activations: {self.throttle_activations}\n"
            f"  Avg Throttle Duration: {self.avg_throttle_duration_ms:.1f}ms\n"
            f"  Recovery Time: {self.recovery_time_ms:.1f}ms"
        )


class BackpressureStressTest:
    """
    Stress test for pipeline backpressure handling.

    Simulates SDR data arrival at various rates and measures
    how the pipeline handles overload conditions.
    """

    def __init__(
        self,
        samples_per_buffer: int = 65536,
        num_ring_segments: int = 16,
        high_watermark: float = 0.85,
        low_watermark: float = 0.50,
    ):
        """
        Initialize stress test.

        Args:
            samples_per_buffer: Samples per ring buffer segment
            num_ring_segments: Number of ring buffer segments
            high_watermark: Buffer fill level to start throttling
            low_watermark: Buffer fill level to stop throttling
        """
        self._samples_per_buffer = samples_per_buffer
        self._num_segments = num_ring_segments
        self._high_watermark = high_watermark
        self._low_watermark = low_watermark

        # Simulated state
        self._buffer_fill = 0.0
        self._throttle_active = False
        self._samples_processed = 0
        self._samples_dropped = 0
        self._events: list[BackpressureEvent] = []

        # Control
        self._running = False
        self._lock = threading.Lock()

    def _simulate_producer(
        self, target_rate_msps: float, duration_seconds: float, burst_factor: float = 1.0
    ):
        """
        Simulate SDR data production.

        Args:
            target_rate_msps: Target sample rate in MSamples/sec
            duration_seconds: Test duration
            burst_factor: Multiply rate by this during bursts
        """
        samples_per_iteration = int(target_rate_msps * 1e6 * 0.001)  # Per millisecond
        iteration_time = 0.001  # 1ms

        start_time = time.time()
        burst_start = start_time + duration_seconds * 0.3
        burst_end = start_time + duration_seconds * 0.6

        while self._running and (time.time() - start_time) < duration_seconds:
            current_time = time.time()

            # Apply burst during middle portion
            if burst_start <= current_time <= burst_end:
                samples = int(samples_per_iteration * burst_factor)
            else:
                samples = samples_per_iteration

            with self._lock:
                if self._throttle_active:
                    # Throttled - drop samples
                    self._samples_dropped += samples
                    self._events.append(
                        BackpressureEvent(
                            timestamp=current_time,
                            event_type="drop",
                            buffer_fill=self._buffer_fill,
                            dropped_samples=samples,
                        )
                    )
                else:
                    # Add to buffer
                    self._buffer_fill += samples / (self._samples_per_buffer * self._num_segments)
                    self._buffer_fill = min(1.0, self._buffer_fill)

                    # Check throttle activation
                    if self._buffer_fill > self._high_watermark and not self._throttle_active:
                        self._throttle_active = True
                        self._events.append(
                            BackpressureEvent(
                                timestamp=current_time,
                                event_type="throttle_start",
                                buffer_fill=self._buffer_fill,
                            )
                        )

            time.sleep(iteration_time)

    def _simulate_consumer(self, processing_rate_msps: float):
        """
        Simulate pipeline consumption.

        Args:
            processing_rate_msps: Processing rate in MSamples/sec
        """
        samples_per_iteration = int(processing_rate_msps * 1e6 * 0.002)  # Per 2ms
        iteration_time = 0.002  # 2ms (simulates processing time)

        while self._running:
            with self._lock:
                if self._buffer_fill > 0:
                    # Process samples
                    consumed = min(
                        samples_per_iteration,
                        int(self._buffer_fill * self._samples_per_buffer * self._num_segments),
                    )
                    self._buffer_fill -= consumed / (self._samples_per_buffer * self._num_segments)
                    self._buffer_fill = max(0.0, self._buffer_fill)
                    self._samples_processed += consumed

                    # Check throttle deactivation
                    if self._buffer_fill < self._low_watermark and self._throttle_active:
                        self._throttle_active = False
                        self._events.append(
                            BackpressureEvent(
                                timestamp=time.time(),
                                event_type="throttle_end",
                                buffer_fill=self._buffer_fill,
                            )
                        )

            time.sleep(iteration_time)

    def run_test(
        self,
        producer_rate_msps: float = 50.0,
        consumer_rate_msps: float = 40.0,
        duration_seconds: float = 10.0,
        burst_factor: float = 1.5,
    ) -> BackpressureTestResult:
        """
        Run backpressure stress test.

        Args:
            producer_rate_msps: SDR production rate
            consumer_rate_msps: Pipeline processing rate
            duration_seconds: Test duration
            burst_factor: Rate multiplier during burst period

        Returns:
            BackpressureTestResult with detailed metrics
        """
        # Reset state
        self._buffer_fill = 0.0
        self._throttle_active = False
        self._samples_processed = 0
        self._samples_dropped = 0
        self._events = []
        self._running = True

        # Start threads
        producer = threading.Thread(
            target=self._simulate_producer,
            args=(producer_rate_msps, duration_seconds, burst_factor),
        )
        consumer = threading.Thread(target=self._simulate_consumer, args=(consumer_rate_msps,))

        start_time = time.time()
        producer.start()
        consumer.start()

        # Monitor buffer fill
        max_fill = 0.0
        fill_history = []

        while producer.is_alive():
            with self._lock:
                fill_history.append((time.time() - start_time, self._buffer_fill))
                max_fill = max(max_fill, self._buffer_fill)
            time.sleep(0.01)

        # Stop consumer
        self._running = False
        consumer.join(timeout=1.0)

        # Calculate metrics
        total_generated = self._samples_processed + self._samples_dropped
        drop_rate = self._samples_dropped / total_generated if total_generated > 0 else 0

        # Count throttle activations
        throttle_starts = [e for e in self._events if e.event_type == "throttle_start"]
        throttle_ends = [e for e in self._events if e.event_type == "throttle_end"]

        # Calculate average throttle duration
        throttle_durations = []
        for i, start in enumerate(throttle_starts):
            # Find matching end
            ends_after = [e for e in throttle_ends if e.timestamp > start.timestamp]
            if ends_after:
                throttle_durations.append((ends_after[0].timestamp - start.timestamp) * 1000)

        avg_throttle_duration = (
            sum(throttle_durations) / len(throttle_durations) if throttle_durations else 0
        )

        # Calculate recovery time (time from max fill to < low_watermark)
        recovery_time = 0.0
        max_fill_time = None
        for t, fill in fill_history:
            if fill == max_fill:
                max_fill_time = t
            if max_fill_time and fill < self._low_watermark:
                recovery_time = (t - max_fill_time) * 1000
                break

        return BackpressureTestResult(
            duration_seconds=duration_seconds,
            total_samples_generated=total_generated,
            total_samples_processed=self._samples_processed,
            total_samples_dropped=self._samples_dropped,
            drop_rate=drop_rate,
            max_buffer_fill=max_fill,
            throttle_activations=len(throttle_starts),
            avg_throttle_duration_ms=avg_throttle_duration,
            recovery_time_ms=recovery_time,
            events=self._events,
        )


class RealPipelineBackpressureTest:
    """
    Backpressure test using actual pipeline components.
    """

    def __init__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError("Requires CuPy")

    async def run_test(
        self, duration_seconds: float = 10.0, overload_factor: float = 1.2
    ) -> BackpressureTestResult:
        """
        Run backpressure test with real pipeline.

        Args:
            duration_seconds: Test duration
            overload_factor: Factor by which to overload the pipeline

        Returns:
            BackpressureTestResult
        """
        from rf_forensics.pipeline.orchestrator import RFForensicsPipeline

        # Create pipeline with test config
        pipeline = RFForensicsPipeline()

        events = []
        fill_history = []
        throttle_activations = 0
        last_throttle_state = False

        # Callback to monitor stats
        def stats_callback(stats):
            nonlocal throttle_activations, last_throttle_state

            fill = stats.get("buffer_fill_level", 0)
            throttled = stats.get("sdr_throttled", False)

            fill_history.append((time.time(), fill))

            if throttled and not last_throttle_state:
                throttle_activations += 1
                events.append(
                    BackpressureEvent(
                        timestamp=time.time(), event_type="throttle_start", buffer_fill=fill
                    )
                )
            elif not throttled and last_throttle_state:
                events.append(
                    BackpressureEvent(
                        timestamp=time.time(), event_type="throttle_end", buffer_fill=fill
                    )
                )

            last_throttle_state = throttled

        pipeline.set_stats_callback(stats_callback)

        try:
            await pipeline.start()

            # Run for duration
            start_time = time.time()
            while (time.time() - start_time) < duration_seconds:
                await asyncio.sleep(0.1)

            status = pipeline.get_status()

        finally:
            await pipeline.stop()

        # Calculate metrics
        total_processed = status.get("samples_processed", 0)
        total_dropped = status.get("dropped_samples", 0)
        total_generated = total_processed + total_dropped

        return BackpressureTestResult(
            duration_seconds=duration_seconds,
            total_samples_generated=total_generated,
            total_samples_processed=total_processed,
            total_samples_dropped=total_dropped,
            drop_rate=total_dropped / total_generated if total_generated > 0 else 0,
            max_buffer_fill=max(f for _, f in fill_history) if fill_history else 0,
            throttle_activations=throttle_activations,
            avg_throttle_duration_ms=0,  # Would need more tracking
            recovery_time_ms=0,
            events=events,
        )


def run_backpressure_suite(verbose: bool = True) -> dict[str, BackpressureTestResult]:
    """
    Run complete backpressure test suite.
    """
    results = {}

    if verbose:
        print("\n" + "=" * 70)
        print("Backpressure Stress Test Suite")
        print("=" * 70)

    # Test 1: Normal load (producer < consumer)
    if verbose:
        print("\n[Test 1: Normal Load]")
    test = BackpressureStressTest()
    results["normal"] = test.run_test(
        producer_rate_msps=30.0, consumer_rate_msps=40.0, duration_seconds=5.0
    )
    if verbose:
        print(f"  Drop Rate: {results['normal'].drop_rate * 100:.2f}%")
        print(f"  Max Fill: {results['normal'].max_buffer_fill * 100:.1f}%")

    # Test 2: Moderate overload
    if verbose:
        print("\n[Test 2: Moderate Overload (1.2x)]")
    results["moderate"] = test.run_test(
        producer_rate_msps=48.0, consumer_rate_msps=40.0, duration_seconds=5.0, burst_factor=1.0
    )
    if verbose:
        print(f"  Drop Rate: {results['moderate'].drop_rate * 100:.2f}%")
        print(f"  Throttle Activations: {results['moderate'].throttle_activations}")

    # Test 3: Heavy overload with burst
    if verbose:
        print("\n[Test 3: Heavy Overload with Burst (2x)]")
    results["heavy"] = test.run_test(
        producer_rate_msps=50.0, consumer_rate_msps=40.0, duration_seconds=5.0, burst_factor=2.0
    )
    if verbose:
        print(f"  Drop Rate: {results['heavy'].drop_rate * 100:.2f}%")
        print(f"  Max Fill: {results['heavy'].max_buffer_fill * 100:.1f}%")
        print(f"  Recovery Time: {results['heavy'].recovery_time_ms:.1f}ms")

    # Test 4: Sustained overload
    if verbose:
        print("\n[Test 4: Sustained Overload]")
    results["sustained"] = test.run_test(
        producer_rate_msps=60.0, consumer_rate_msps=40.0, duration_seconds=10.0, burst_factor=1.0
    )
    if verbose:
        print(f"  Drop Rate: {results['sustained'].drop_rate * 100:.2f}%")
        print(f"  Throttle Activations: {results['sustained'].throttle_activations}")
        print(f"  Avg Throttle Duration: {results['sustained'].avg_throttle_duration_ms:.1f}ms")

    if verbose:
        print("\n" + "=" * 70)
        print("Backpressure Tests Complete")
        print("=" * 70)

    return results


if __name__ == "__main__":
    run_backpressure_suite(verbose=True)
