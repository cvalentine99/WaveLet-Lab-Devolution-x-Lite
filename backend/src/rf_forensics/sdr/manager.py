"""
SDR Manager - Unified SDR hardware ownership.

Thread-safe singleton that owns the SDR driver instance and provides:
- Single point of access for both API and pipeline
- Metrics tracking and exposure
- Auto-reconnection on disconnect
- Backpressure signaling
- Configuration state persistence
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from rf_forensics.sdr.metrics import MetricsTracker, SDRCapabilities, SDRMetrics

if TYPE_CHECKING:
    from rf_forensics.sdr.usdr_driver import USDRConfig, USDRDriver

logger = logging.getLogger(__name__)


class SDRManager:
    """
    Thread-safe singleton managing SDR hardware.

    Provides unified access to SDR hardware for both the API and pipeline,
    eliminating dual ownership and enabling centralized metrics/health tracking.

    Usage:
        manager = SDRManager()  # Always returns same instance
        driver = manager.get_driver()
        manager.configure(config)
        manager.start_streaming(callback)

    Architecture:
        ┌─────────────────────────────────────────────────┐
        │              SDRManager (singleton)              │
        │  ─────────────────────────────────────────────  │
        │  - Thread-safe singleton with RLock             │
        │  - Owns the single USDRDriver instance          │
        │  - Provides proxy for pipeline                  │
        │  - Exposes metrics & state to API               │
        │  - Handles reconnection & backpressure          │
        └─────────────────────────────────────────────────┘
                 ↑                        ↑
                 │                        │
            ┌────┴────┐              ┌────┴────┐
            │   API   │              │Pipeline │
            │ Router  │              │         │
            └─────────┘              └─────────┘
    """

    _instance: SDRManager | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """Thread-safe singleton instantiation."""
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(self):
        """Initialize the manager (only runs once)."""
        if self._initialized:
            return

        self._initialized = True

        # Driver management
        self._driver: USDRDriver | None = None
        self._driver_lock = threading.RLock()

        # Configuration persistence
        self._last_config: USDRConfig | None = None

        # Metrics tracking
        self._metrics_tracker = MetricsTracker()

        # Streaming state
        self._streaming = False
        self._stream_callback: Callable | None = None
        self._stream_start_time: float | None = None

        # Auto-reconnection
        self._reconnect_enabled = True
        self._reconnect_task: asyncio.Task | None = None
        self._reconnect_interval_seconds = 5.0
        self._max_reconnect_attempts = 10

        # Backpressure
        self._backpressure_threshold = 0.75  # 75% buffer fill triggers warning
        self._current_buffer_fill = 0.0

        logger.info("SDRManager initialized")

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (for testing only).

        Warning: This should only be used in tests to ensure clean state.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    cls._instance.shutdown()
                except Exception:
                    pass
                cls._instance = None

    def get_driver(self) -> USDRDriver:
        """
        Get the SDR driver instance (creates if needed).

        Returns:
            USDRDriver: The single driver instance.

        Thread-safe: Yes
        """
        with self._driver_lock:
            if self._driver is None:
                from rf_forensics.sdr.usdr_driver import USDRDriver

                self._driver = USDRDriver()
                logger.info("Created USDRDriver instance")
            return self._driver

    @property
    def is_available(self) -> bool:
        """Check if SDR library is available."""
        driver = self.get_driver()
        return driver.is_available

    @property
    def is_connected(self) -> bool:
        """Check if SDR is connected."""
        with self._driver_lock:
            if self._driver is None:
                return False
            return self._driver.is_connected

    @property
    def is_streaming(self) -> bool:
        """Check if streaming is active."""
        return self._streaming

    def discover(self) -> list:
        """
        Discover available SDR devices.

        Returns:
            List of USDRDevice objects (empty if none found).

        Note: Does NOT return placeholder devices.
        """
        driver = self.get_driver()
        devices = driver.discover()

        # Filter out placeholder devices
        real_devices = [
            d for d in devices if d.status == "available" and not d.id.startswith("placeholder")
        ]

        logger.info(f"Discovered {len(real_devices)} SDR device(s)")
        return real_devices

    def connect(self, device_id: str = "") -> bool:
        """
        Connect to SDR device.

        Args:
            device_id: Device connection string (empty for auto-select).

        Returns:
            True if connected successfully.
        """
        with self._driver_lock:
            driver = self.get_driver()

            if driver.is_connected:
                logger.info("Already connected, disconnecting first")
                driver.disconnect()

            success = driver.connect(device_id)

            if success:
                logger.info(f"Connected to SDR: {device_id or 'auto'}")
                self._metrics_tracker.reset()

                # Restore last config if available
                if self._last_config is not None:
                    self.configure(self._last_config)
            else:
                logger.error(f"Failed to connect to SDR: {device_id or 'auto'}")
                self._metrics_tracker.record_error(f"Connection failed: {device_id}")

            return success

    def disconnect(self) -> bool:
        """Disconnect from SDR device."""
        with self._driver_lock:
            if self._streaming:
                self.stop_streaming()

            if self._driver is not None:
                success = self._driver.disconnect()
                logger.info("Disconnected from SDR")
                return success

            return True

    def configure(self, config: USDRConfig) -> bool:
        """
        Apply configuration to SDR.

        Args:
            config: USDRConfig with desired settings.

        Returns:
            True if configuration applied successfully.

        Thread-safe: Yes
        """
        with self._driver_lock:
            self._last_config = config

            if self._driver is None or not self._driver.is_connected:
                logger.debug("Config saved, will apply when connected")
                return True

            success = self._driver.configure(config)

            if success:
                # Update metrics with new config
                status = self._driver.get_status()
                self._metrics_tracker.update_hardware_status(
                    temperature_c=status.temperature_c,
                    actual_sample_rate_hz=status.actual_sample_rate_hz,
                    actual_freq_hz=status.actual_freq_hz,
                )
                logger.info(
                    f"Configured SDR: {config.center_freq_hz / 1e6:.3f} MHz @ {config.sample_rate_hz / 1e6:.1f} MSPS"
                )
            else:
                self._metrics_tracker.record_error("Configuration failed")

            return success

    def start_streaming(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        """
        Start RX streaming with metrics tracking.

        Args:
            callback: Function called with (samples, timestamp) for each buffer.

        Returns:
            True if streaming started successfully.
        """
        with self._driver_lock:
            if self._streaming:
                logger.warning("Already streaming")
                return True

            driver = self.get_driver()

            if not driver.is_connected:
                logger.error("Cannot start streaming: not connected")
                return False

            # Wrap callback with metrics tracking
            self._stream_callback = callback

            def metrics_callback(samples: np.ndarray, timestamp: float):
                # Track samples received
                self._metrics_tracker.record_samples(received=len(samples))

                # Call user callback
                if self._stream_callback:
                    try:
                        self._stream_callback(samples, timestamp)
                    except Exception as e:
                        logger.error(f"Stream callback error: {e}")
                        self._metrics_tracker.record_error(f"Callback error: {e}")

            success = driver.start_streaming(metrics_callback)

            if success:
                self._streaming = True
                self._stream_start_time = time.time()
                self._metrics_tracker.start_streaming()
                logger.info("Started SDR streaming")
            else:
                self._metrics_tracker.record_error("Failed to start streaming")

            return success

    def stop_streaming(self) -> bool:
        """Stop RX streaming."""
        with self._driver_lock:
            if not self._streaming:
                return True

            if self._driver is not None:
                self._driver.stop_streaming()

            self._streaming = False
            self._stream_callback = None
            self._metrics_tracker.stop_streaming()
            logger.info("Stopped SDR streaming")

            return True

    def get_metrics(self) -> SDRMetrics:
        """
        Get current SDR metrics.

        Returns:
            SDRMetrics with current hardware/streaming metrics.
        """
        # Update hardware status if connected
        with self._driver_lock:
            if self._driver is not None and self._driver.is_connected:
                try:
                    status = self._driver.get_status()
                    self._metrics_tracker.update_hardware_status(
                        temperature_c=status.temperature_c,
                        actual_sample_rate_hz=status.actual_sample_rate_hz,
                        actual_freq_hz=status.actual_freq_hz,
                    )
                except Exception:
                    pass

        return self._metrics_tracker.metrics

    def get_capabilities(self) -> SDRCapabilities:
        """
        Get SDR hardware capabilities.

        Returns:
            SDRCapabilities with supported ranges and features.
        """
        caps = SDRCapabilities()

        with self._driver_lock:
            if self._driver is not None and self._driver.is_connected:
                status = self._driver.get_status()
                caps.device_id = status.device_id or ""

                # LMS7002M typical ranges
                caps.freq_range_hz = (70_000_000, 6_000_000_000)
                caps.sample_rate_range_hz = (100_000, 61_440_000)
                caps.bandwidth_range_hz = (100_000, 56_000_000)
                caps.gain_range_db = (0, 74)  # LNA(30) + TIA(12) + PGA(32)
                caps.rx_paths = ["LNAH", "LNAL", "LNAW"]
                caps.model = "uSDR DevBoard"
                caps.max_channels = 2

                # Check for GPUDirect support (nvidia-peermem loaded)
                try:
                    import os

                    if os.path.exists("/sys/module/nvidia_peermem"):
                        caps.supports_gpudirect = True
                except Exception:
                    pass

        return caps

    def record_overflow(self, count: int = 1) -> None:
        """Record overflow event from driver."""
        self._metrics_tracker.record_overflow(count)
        logger.warning(f"SDR overflow: {count} samples lost")

    def record_dropped_samples(self, count: int) -> None:
        """Record dropped samples (backpressure)."""
        self._metrics_tracker.record_samples(received=0, dropped=count)
        self._metrics_tracker.record_backpressure()

    def update_buffer_fill(self, fill_percent: float) -> None:
        """
        Update buffer fill level for backpressure tracking.

        Args:
            fill_percent: Buffer fill level (0.0 to 1.0).
        """
        self._current_buffer_fill = fill_percent
        self._metrics_tracker.update_buffer_fill(fill_percent * 100)

        if fill_percent > self._backpressure_threshold:
            self._metrics_tracker.record_backpressure()
            logger.warning(f"Buffer fill {fill_percent * 100:.1f}% - backpressure!")

    async def start_reconnection_loop(self) -> None:
        """
        Start background auto-reconnection task.

        Call this from an async context (e.g., FastAPI startup).
        """
        if self._reconnect_task is not None:
            return

        self._reconnect_task = asyncio.create_task(self._reconnection_loop())
        logger.info("Started SDR auto-reconnection loop")

    async def stop_reconnection_loop(self) -> None:
        """Stop the auto-reconnection task."""
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None
            logger.info("Stopped SDR auto-reconnection loop")

    async def _reconnection_loop(self) -> None:
        """Background task for auto-reconnection."""
        consecutive_failures = 0

        while True:
            try:
                await asyncio.sleep(self._reconnect_interval_seconds)

                if not self._reconnect_enabled:
                    continue

                with self._driver_lock:
                    # Check if we need to reconnect
                    if self._driver is None:
                        continue

                    if self._driver.is_connected:
                        consecutive_failures = 0
                        continue

                    # We were connected but now disconnected
                    if self._streaming or self._last_config is not None:
                        logger.info("SDR disconnected, attempting reconnect...")
                        self._metrics_tracker.record_reconnect()

                        devices = self._driver.discover()
                        real_devices = [d for d in devices if d.status == "available"]

                        if real_devices:
                            if self._driver.connect(real_devices[0].id):
                                logger.info("SDR reconnected successfully")

                                # Restore previous config
                                if self._last_config is not None:
                                    self._driver.configure(self._last_config)

                                # Restart streaming if it was active
                                if self._streaming and self._stream_callback:
                                    self._driver.start_streaming(self._stream_callback)

                                consecutive_failures = 0
                            else:
                                consecutive_failures += 1
                        else:
                            consecutive_failures += 1

                        if consecutive_failures >= self._max_reconnect_attempts:
                            logger.error(
                                f"Max reconnect attempts ({self._max_reconnect_attempts}) reached"
                            )
                            self._metrics_tracker.record_error("Max reconnect attempts reached")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconnection loop error: {e}")
                await asyncio.sleep(1)

    def shutdown(self) -> None:
        """Shutdown the manager and release resources."""
        logger.info("Shutting down SDRManager")

        with self._driver_lock:
            if self._streaming:
                self.stop_streaming()

            if self._driver is not None:
                self._driver.disconnect()
                self._driver = None

        self._last_config = None

    def get_status_dict(self) -> dict:
        """Get comprehensive status for API response."""
        with self._driver_lock:
            status = {
                "available": self.is_available,
                "connected": self.is_connected,
                "streaming": self.is_streaming,
                "device_id": None,
                "config": None,
            }

            if self._driver is not None and self._driver.is_connected:
                driver_status = self._driver.get_status()
                status["device_id"] = driver_status.device_id
                status["hardware"] = {
                    "temperature_c": driver_status.temperature_c,
                    "actual_freq_hz": driver_status.actual_freq_hz,
                    "actual_sample_rate_hz": driver_status.actual_sample_rate_hz,
                    "rx_path": driver_status.rx_path,
                }

            if self._last_config is not None:
                status["config"] = {
                    "center_freq_hz": self._last_config.center_freq_hz,
                    "sample_rate_hz": self._last_config.sample_rate_hz,
                    "bandwidth_hz": self._last_config.bandwidth_hz,
                    "gain_total_db": self._last_config.gain.total_db,
                }

            return status


# Module-level accessor for the singleton
def get_sdr_manager() -> SDRManager:
    """
    Get the SDR manager singleton.

    This is the preferred way to access the SDR from any component.

    Returns:
        SDRManager: The singleton instance.
    """
    return SDRManager()
