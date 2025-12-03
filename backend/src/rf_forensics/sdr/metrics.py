"""
SDR Metrics - Real-time hardware metrics tracking.

Provides observability into SDR hardware state including:
- Overflow/drop tracking
- Hardware health (temperature, PLL lock)
- Streaming performance metrics
- Error history
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SDRMetrics:
    """Real-time SDR hardware metrics."""

    # Overflow tracking
    total_overflows: int = 0
    overflow_rate_per_sec: float = 0.0
    last_overflow_timestamp: float | None = None

    # Sample tracking
    total_samples_received: int = 0
    total_samples_dropped: int = 0
    drop_rate_percent: float = 0.0

    # Hardware health
    temperature_c: float = 0.0
    pll_locked: bool = False
    actual_sample_rate_hz: float = 0.0
    actual_freq_hz: float = 0.0

    # Streaming metrics
    streaming_uptime_seconds: float = 0.0
    streaming_start_time: float | None = None
    reconnect_count: int = 0
    last_error: str | None = None

    # Backpressure tracking
    backpressure_events: int = 0
    current_buffer_fill_percent: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "overflow": {
                "total": self.total_overflows,
                "rate_per_sec": self.overflow_rate_per_sec,
                "last_timestamp": self.last_overflow_timestamp,
            },
            "samples": {
                "total_received": self.total_samples_received,
                "total_dropped": self.total_samples_dropped,
                "drop_rate_percent": self.drop_rate_percent,
            },
            "hardware": {
                "temperature_c": self.temperature_c,
                "pll_locked": self.pll_locked,
                "actual_sample_rate_hz": self.actual_sample_rate_hz,
                "actual_freq_hz": self.actual_freq_hz,
            },
            "streaming": {
                "uptime_seconds": self.streaming_uptime_seconds,
                "reconnect_count": self.reconnect_count,
                "last_error": self.last_error,
            },
            "backpressure": {
                "events": self.backpressure_events,
                "buffer_fill_percent": self.current_buffer_fill_percent,
            },
        }


class MetricsTracker:
    """
    Thread-safe metrics tracker with rolling window calculations.

    Provides rate calculations over configurable time windows
    and maintains error history.
    """

    def __init__(self, window_seconds: float = 10.0, max_errors: int = 100):
        self._metrics = SDRMetrics()
        self._lock = threading.RLock()

        # Rolling window for rate calculations
        self._window_seconds = window_seconds
        self._overflow_timestamps: deque = deque(maxlen=1000)
        self._sample_timestamps: deque = deque(maxlen=1000)

        # Error history
        self._max_errors = max_errors
        self._error_history: deque = deque(maxlen=max_errors)

    @property
    def metrics(self) -> SDRMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            self._update_rates()
            self._update_uptime()
            return self._metrics

    def record_overflow(self, count: int = 1) -> None:
        """Record overflow event(s)."""
        with self._lock:
            now = time.time()
            self._metrics.total_overflows += count
            self._metrics.last_overflow_timestamp = now

            # Record timestamps for rate calculation
            for _ in range(count):
                self._overflow_timestamps.append(now)

    def record_samples(self, received: int, dropped: int = 0) -> None:
        """Record sample batch."""
        with self._lock:
            self._metrics.total_samples_received += received
            self._metrics.total_samples_dropped += dropped

            now = time.time()
            self._sample_timestamps.append((now, received, dropped))

    def record_error(self, error: str) -> None:
        """Record error event."""
        with self._lock:
            now = time.time()
            self._metrics.last_error = error
            self._error_history.append(
                {
                    "timestamp": now,
                    "error": error,
                }
            )

    def record_reconnect(self) -> None:
        """Record reconnection event."""
        with self._lock:
            self._metrics.reconnect_count += 1

    def record_backpressure(self) -> None:
        """Record backpressure event."""
        with self._lock:
            self._metrics.backpressure_events += 1

    def update_hardware_status(
        self,
        temperature_c: float | None = None,
        pll_locked: bool | None = None,
        actual_sample_rate_hz: float | None = None,
        actual_freq_hz: float | None = None,
    ) -> None:
        """Update hardware status metrics."""
        with self._lock:
            if temperature_c is not None:
                self._metrics.temperature_c = temperature_c
            if pll_locked is not None:
                self._metrics.pll_locked = pll_locked
            if actual_sample_rate_hz is not None:
                self._metrics.actual_sample_rate_hz = actual_sample_rate_hz
            if actual_freq_hz is not None:
                self._metrics.actual_freq_hz = actual_freq_hz

    def update_buffer_fill(self, fill_percent: float) -> None:
        """Update current buffer fill level."""
        with self._lock:
            self._metrics.current_buffer_fill_percent = fill_percent

    def start_streaming(self) -> None:
        """Mark streaming start time."""
        with self._lock:
            self._metrics.streaming_start_time = time.time()

    def stop_streaming(self) -> None:
        """Mark streaming stopped."""
        with self._lock:
            self._update_uptime()
            self._metrics.streaming_start_time = None

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics = SDRMetrics()
            self._overflow_timestamps.clear()
            self._sample_timestamps.clear()
            self._error_history.clear()

    def get_error_history(self) -> list[dict]:
        """Get recent error history."""
        with self._lock:
            return list(self._error_history)

    def _update_rates(self) -> None:
        """Update rate calculations based on rolling window."""
        now = time.time()
        cutoff = now - self._window_seconds

        # Overflow rate
        recent_overflows = sum(1 for t in self._overflow_timestamps if t > cutoff)
        self._metrics.overflow_rate_per_sec = recent_overflows / self._window_seconds

        # Drop rate
        if self._metrics.total_samples_received > 0:
            self._metrics.drop_rate_percent = (
                self._metrics.total_samples_dropped / self._metrics.total_samples_received * 100
            )

    def _update_uptime(self) -> None:
        """Update streaming uptime."""
        if self._metrics.streaming_start_time is not None:
            self._metrics.streaming_uptime_seconds = (
                time.time() - self._metrics.streaming_start_time
            )


@dataclass
class SDRCapabilities:
    """Hardware capabilities queried from SDR device."""

    freq_range_hz: tuple = (0, 0)  # (min, max)
    sample_rate_range_hz: tuple = (0, 0)  # (min, max)
    bandwidth_range_hz: tuple = (0, 0)  # (min, max)
    gain_range_db: tuple = (0, 0)  # (min, max)
    rx_paths: list[str] = field(default_factory=list)
    supported_formats: list[str] = field(default_factory=lambda: ["ci16", "cf32"])

    # Hardware info
    device_id: str = ""
    model: str = ""
    serial: str = ""
    firmware_version: str = ""

    # Feature flags
    supports_gpudirect: bool = False
    supports_timestamps: bool = False
    max_channels: int = 1

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "freq_range_hz": {"min": self.freq_range_hz[0], "max": self.freq_range_hz[1]},
            "sample_rate_range_hz": {
                "min": self.sample_rate_range_hz[0],
                "max": self.sample_rate_range_hz[1],
            },
            "bandwidth_range_hz": {
                "min": self.bandwidth_range_hz[0],
                "max": self.bandwidth_range_hz[1],
            },
            "gain_range_db": {"min": self.gain_range_db[0], "max": self.gain_range_db[1]},
            "rx_paths": self.rx_paths,
            "supported_formats": self.supported_formats,
            "device": {
                "id": self.device_id,
                "model": self.model,
                "serial": self.serial,
                "firmware_version": self.firmware_version,
            },
            "features": {
                "supports_gpudirect": self.supports_gpudirect,
                "supports_timestamps": self.supports_timestamps,
                "max_channels": self.max_channels,
            },
        }
