"""
GPU RF Forensics Engine - Pipeline Orchestrator

Main processing pipeline coordinating all components.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from rf_forensics.config.schema import PRESETS, RFForensicsConfig
from rf_forensics.core.async_transfer import AsyncTransferManager
from rf_forensics.core.memory_manager import PinnedMemoryManager
from rf_forensics.core.ring_buffer import GPURingBuffer
from rf_forensics.core.stream_manager import CUDAStreamManager
from rf_forensics.detection.cfar import CFARDetector
from rf_forensics.detection.peaks import Detection, PeakDetector
from rf_forensics.dsp.psd import SpectrogramBuffer, WelchPSD
from rf_forensics.ml.clustering import EmitterClusterer
from rf_forensics.ml.features import FeatureExtractor
from rf_forensics.sdr.manager import SDRManager, get_sdr_manager
from rf_forensics.sdr.usdr_driver import USDRConfig, USDRGain

# AMC imports (optional - graceful degradation if not available)
try:
    from rf_forensics.ml.amc import (
        AMCConfig,
        AMCInferenceEngine,
        CumulantFallbackClassifier,
        normalize_modulation_name,
    )

    AMC_AVAILABLE = True
except ImportError:
    AMC_AVAILABLE = False
    AMCInferenceEngine = None
    CumulantFallbackClassifier = None
    AMCConfig = None

# TorchSig adapter (optional - preferred for production quality)
try:
    from rf_forensics.ml.amc import TORCHSIG_AVAILABLE, TorchSigAdapter
except ImportError:
    TorchSigAdapter = None
    TORCHSIG_AVAILABLE = False

# Anomaly detection (optional - graceful fallback)
try:
    from rf_forensics.ml.anomaly import AnomalyDetector, create_anomaly_detector

    ANOMALY_AVAILABLE = True
except ImportError:
    ANOMALY_AVAILABLE = False
    AnomalyDetector = None
    create_anomaly_detector = None

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Pipeline operational states."""

    IDLE = "idle"
    CONFIGURING = "configuring"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class PipelineStatus:
    """Current pipeline status."""

    state: str
    uptime_seconds: float
    samples_processed: int
    detections_count: int
    current_throughput_msps: float
    gpu_memory_used_gb: float
    buffer_fill_level: float = 0.0
    last_error: str | None = None


@dataclass
class PipelineMetrics:
    """Performance metrics."""

    total_samples: int = 0
    total_detections: int = 0
    start_time: float = 0.0
    samples_per_second: float = 0.0
    processing_latency_ms: float = 0.0
    gpu_utilization: float = 0.0
    consecutive_errors: int = 0
    total_dropped_samples: int = 0


class RFForensicsPipeline:
    """
    Main processing pipeline orchestrator.

    Coordinates all processing stages:
    - Stage 0: SDR → Pinned Buffer (CPU thread)
    - Stage 1: Pinned Buffer → GPU (async transfer)
    - Stage 2: PSD Computation
    - Stage 3: CFAR Detection
    - Stage 4: Feature Extraction
    - Stage 5: Classification/Clustering
    - Stage 6: Output (WebSocket, callbacks)
    """

    def __init__(self, config: RFForensicsConfig | None = None, config_path: str | None = None):
        """
        Initialize pipeline.

        Args:
            config: Configuration object.
            config_path: Path to YAML configuration file.
        """
        if config_path:
            self._config = RFForensicsConfig.from_yaml(config_path)
        else:
            self._config = config or RFForensicsConfig()

        self._state = PipelineState.IDLE
        self._metrics = PipelineMetrics()
        self._metrics_lock = threading.Lock()  # Thread-safe metrics updates

        # Components (initialized on start)
        self._memory_manager: PinnedMemoryManager | None = None
        self._stream_manager: CUDAStreamManager | None = None
        self._ring_buffer: GPURingBuffer | None = None
        self._sdr_manager: SDRManager | None = None  # Uses unified SDRManager singleton
        self._psd: WelchPSD | None = None
        self._spectrogram: SpectrogramBuffer | None = None
        self._cfar: CFARDetector | None = None
        self._peak_detector: PeakDetector | None = None
        self._feature_extractor: FeatureExtractor | None = None
        self._clusterer: EmitterClusterer | None = None

        # AMC classification
        self._torchsig_adapter: TorchSigAdapter | None = None  # Primary: TorchSig XCiT model
        self._amc_engine: AMCInferenceEngine | None = None  # Secondary: Custom AMC
        self._amc_fallback: CumulantFallbackClassifier | None = None  # Fallback: Cumulants

        # Anomaly detection (per BACKEND_CONTRACT.md)
        self._anomaly_detector: AnomalyDetector | None = None

        # Async transfer manager for non-blocking D2H transfers
        self._async_transfers: AsyncTransferManager | None = None

        # Callbacks
        self._spectrum_callback: Callable | None = None
        self._detection_callback: Callable | None = None
        self._cluster_callback: Callable | None = None
        self._spectrogram_callback: Callable | None = None  # Optional spectrogram consumer
        self._error_callback: Callable | None = None  # Called when entering ERROR state
        self._stats_callback: Callable | None = None  # Periodic pipeline stats

        # Error threshold for transitioning to ERROR state
        self._max_consecutive_errors = 10

        # Backpressure threshold for shedding detection work
        self._backpressure_threshold = 0.8  # Skip detection above 80% fill

        # Stats broadcast interval
        self._stats_interval = 1.0  # 1 Hz
        self._last_stats_time = 0.0

        # Event loop for cross-thread callbacks
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None  # Will be set when callbacks are first registered

        # SDR throttling for backpressure
        self._throttle_sdr = False  # When True, SDR callback drops samples immediately
        self._throttle_high_watermark = 0.85  # Start throttling above 85% fill
        self._throttle_low_watermark = 0.5  # Stop throttling below 50% fill

        # Processing state
        self._processing_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # Background callback tasks (fire-and-forget, prevents GC)
        self._pending_callbacks: set[asyncio.Task] = set()
        self._max_pending_callbacks = self._config.pipeline.max_pending_callbacks

        # Detection storage (bounded deque for O(1) append with auto-eviction)
        self._recent_detections: deque[Detection] = deque(
            maxlen=self._config.pipeline.max_recent_detections
        )

    @property
    def config(self) -> RFForensicsConfig:
        return self._config

    @property
    def state(self) -> PipelineState:
        return self._state

    def _init_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")

        # Memory management
        self._memory_manager = PinnedMemoryManager(
            default_buffer_samples=self._config.buffer.samples_per_buffer
        )

        # CUDA streams
        if CUPY_AVAILABLE:
            self._stream_manager = CUDAStreamManager()

            # Async D2H transfer manager for non-blocking GPU→CPU transfers
            # Uses TRUE async memcpyAsync - overlaps D2H with compute
            # Detection pipeline uses one-frame latency (acceptable)
            # Spectrum callback uses sync path to avoid display latency
            self._async_transfers = AsyncTransferManager()
            self._async_transfers.create_channel(
                "psd", max_size=self._config.fft.fft_size, dtype=np.float32
            )
            # CFAR mask transferred as bool (memcpyAsync handles raw bytes)
            self._async_transfers.create_channel(
                "cfar", max_size=self._config.fft.fft_size, dtype=np.bool_
            )

        # Ring buffer
        self._ring_buffer = GPURingBuffer(
            num_segments=self._config.buffer.num_ring_segments,
            samples_per_segment=self._config.buffer.samples_per_buffer,
            use_gpu_buffer=CUPY_AVAILABLE,
        )

        # SDR Manager (singleton - unified driver ownership)
        self._sdr_manager = get_sdr_manager()

        # Do NOT auto-connect - wait for frontend to connect via REST API
        # Frontend workflow: /api/sdr/devices -> /api/sdr/connect -> /api/start
        if not self._sdr_manager.is_connected:
            logger.info("SDR not connected - waiting for frontend to connect via /api/sdr/connect")

        # Build USDR config from pipeline config
        gain_db = int(self._config.sdr.gain_db)
        lna_gain = min(30, gain_db)
        remaining = gain_db - lna_gain
        tia_gain = min(12, remaining)
        pga_gain = min(32, remaining - tia_gain)

        usdr_config = USDRConfig(
            center_freq_hz=int(self._config.sdr.center_freq_hz),
            sample_rate_hz=int(self._config.sdr.sample_rate_hz),
            bandwidth_hz=int(self._config.sdr.bandwidth_hz),
            gain=USDRGain(lna_db=lna_gain, tia_db=tia_gain, pga_db=pga_gain),
        )
        self._sdr_manager.configure(usdr_config)

        # DSP components
        self._psd = WelchPSD(
            fft_size=self._config.fft.fft_size,
            overlap=self._config.fft.overlap_percent / 100,
            window=self._config.fft.window_type,
            sample_rate=self._config.sdr.sample_rate_hz,
        )

        # Spectrogram buffer - only allocate if a callback is registered
        # (saves GPU memory when waterfall display not needed)
        if self._spectrogram_callback:
            self._spectrogram = SpectrogramBuffer(
                num_time_bins=256,
                fft_size=self._config.fft.fft_size,
                sample_rate=self._config.sdr.sample_rate_hz,
            )
            logger.info("Spectrogram buffer allocated (callback registered)")
        else:
            self._spectrogram = None
            logger.debug("Spectrogram buffer skipped (no callback)")

        # Detection
        self._cfar = CFARDetector(
            num_reference=self._config.cfar.num_reference_cells,
            num_guard=self._config.cfar.num_guard_cells,
            pfa=self._config.cfar.pfa,
            variant=self._config.cfar.variant,
        )

        self._peak_detector = PeakDetector()

        # ML components
        self._feature_extractor = FeatureExtractor(sample_rate=self._config.sdr.sample_rate_hz)

        self._clusterer = EmitterClusterer(
            eps=self._config.clustering.eps,
            min_samples=self._config.clustering.min_samples,
            auto_tune=self._config.clustering.auto_tune,
        )

        # AMC classification (optional) - Priority: TorchSig > Custom AMC > Cumulants
        amc_initialized = False

        # Try TorchSig first (57 modulation types, pre-trained)
        if TORCHSIG_AVAILABLE and TorchSigAdapter is not None:
            try:
                self._torchsig_adapter = TorchSigAdapter()
                if self._torchsig_adapter.initialize():
                    logger.info("TorchSig AMC adapter initialized (57 classes)")
                    amc_initialized = True
                else:
                    logger.warning("TorchSig initialization failed, trying fallbacks")
                    self._torchsig_adapter = None
            except Exception as e:
                logger.warning(f"TorchSig adapter init failed: {e}")
                self._torchsig_adapter = None

        # Try custom AMC engine if TorchSig unavailable
        if not amc_initialized and AMC_AVAILABLE:
            try:
                amc_config = AMCConfig()
                self._amc_engine = AMCInferenceEngine(config=amc_config)
                if self._amc_engine.initialize():
                    logger.info("AMC neural network engine initialized")
                    amc_initialized = True
                else:
                    logger.warning("AMC neural network failed, using cumulant fallback")
                    self._amc_engine = None
            except Exception as e:
                logger.warning(f"AMC engine init failed: {e}")
                self._amc_engine = None

        # Always have cumulant fallback ready
        if AMC_AVAILABLE and CumulantFallbackClassifier is not None:
            self._amc_fallback = CumulantFallbackClassifier()
            if not amc_initialized:
                logger.info("Using cumulant-based AMC fallback classifier")

        if not amc_initialized and self._amc_fallback is None:
            logger.info("AMC module not available, classification disabled")

        # Initialize anomaly detector (per BACKEND_CONTRACT.md - detection.anomaly_score)
        if ANOMALY_AVAILABLE and create_anomaly_detector is not None:
            try:
                self._anomaly_detector = create_anomaly_detector(prefer_neural=True)
                if self._anomaly_detector.available:
                    logger.info("Anomaly detector initialized (autoencoder)")
                else:
                    logger.info("Anomaly detector initialized (statistical fallback)")
            except Exception as e:
                logger.warning(f"Anomaly detector init failed: {e}")
                self._anomaly_detector = None
        else:
            logger.info("Anomaly detection module not available")

        logger.info("Pipeline components initialized")

    async def start(self):
        """Start the processing pipeline."""
        if self._state == PipelineState.RUNNING:
            logger.warning("Pipeline already running")
            return

        logger.info("Starting pipeline...")

        # ALWAYS cleanup first, regardless of current state
        # This guarantees no resource leaks from previous runs (including ERROR state)
        if self._state != PipelineState.IDLE:
            logger.info(f"Cleaning up from previous state ({self._state.value})")
        await self._cleanup_resources()

        # Reset to known state before configuration
        self._state = PipelineState.CONFIGURING

        try:
            # Initialize components
            self._init_components()

            # Start SDR streaming via manager (config already applied in _init_components)
            if not self._sdr_manager or not self._sdr_manager.is_connected:
                raise RuntimeError("SDR not connected - cannot start pipeline")

            started = self._sdr_manager.start_streaming(self._sdr_callback)
            if not started:
                raise RuntimeError("Failed to start SDR streaming")

            # Start processing loop
            self._stop_event.clear()
            self._metrics.start_time = time.time()
            self._processing_task = asyncio.create_task(self._processing_loop())

            self._state = PipelineState.RUNNING
            logger.info("Pipeline started")

        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            # Cleanup any partially-initialized resources to prevent GPU memory leaks
            await self._cleanup_resources()
            self._state = PipelineState.ERROR
            raise

    async def stop(self):
        """Stop the processing pipeline."""
        if self._state not in [PipelineState.RUNNING, PipelineState.PAUSED]:
            return

        logger.info("Stopping pipeline...")
        self._state = PipelineState.STOPPING

        # Stop SDR streaming first (via manager)
        if self._sdr_manager:
            self._sdr_manager.stop_streaming()

        # Stop processing loop
        self._stop_event.set()
        if self._processing_task:
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except TimeoutError:
                logger.warning("Processing task did not stop in time")
            self._processing_task = None

        # Full resource cleanup
        await self._cleanup_resources()

        self._state = PipelineState.IDLE
        logger.info("Pipeline stopped")

    async def _cleanup_resources(self):
        """Release all pipeline resources to prevent memory leaks."""
        logger.debug("Cleaning up pipeline resources...")

        # Cancel pending callbacks first (they may reference other resources)
        await self._cancel_pending_callbacks()

        # Note: SDR is owned by SDRManager singleton, don't close it here
        # Just stop streaming if active (already done in stop())
        if self._sdr_manager and self._sdr_manager.is_streaming:
            self._sdr_manager.stop_streaming()
        self._sdr_manager = None  # Release reference (singleton persists)

        # Cleanup async transfer manager
        if self._async_transfers:
            try:
                self._async_transfers.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up async transfers: {e}")
            self._async_transfers = None

        # Cleanup CUDA stream manager
        if self._stream_manager:
            try:
                self._stream_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up stream manager: {e}")
            self._stream_manager = None

        # Cleanup ring buffer
        if self._ring_buffer:
            try:
                self._ring_buffer.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up ring buffer: {e}")
            self._ring_buffer = None

        # Cleanup PSD estimator
        if self._psd:
            try:
                self._psd.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up PSD: {e}")
            self._psd = None

        # Cleanup spectrogram buffer
        if self._spectrogram:
            try:
                self._spectrogram.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up spectrogram: {e}")
            self._spectrogram = None

        # Clear other components (allow GC)
        self._cfar = None
        self._peak_detector = None
        self._feature_extractor = None
        self._clusterer = None

        # Cleanup AMC classifiers
        if self._torchsig_adapter:
            try:
                self._torchsig_adapter.shutdown()
            except Exception as e:
                logger.warning(f"Error cleaning up TorchSig adapter: {e}")
            self._torchsig_adapter = None
        if self._amc_engine:
            try:
                self._amc_engine.shutdown()
            except Exception as e:
                logger.warning(f"Error cleaning up AMC engine: {e}")
            self._amc_engine = None
        self._amc_fallback = None
        self._anomaly_detector = None

        # Clear memory manager
        if self._memory_manager:
            try:
                self._memory_manager.clear_all()
            except Exception as e:
                logger.warning(f"Error clearing memory manager: {e}")
            self._memory_manager = None

        # Clear detection storage
        self._recent_detections.clear()

        # Force GPU memory pool cleanup
        if CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                logger.warning(f"Error freeing GPU memory pool: {e}")

        logger.debug("Pipeline resources cleaned up")

    async def pause(self):
        """Pause processing (SDR keeps running)."""
        if self._state == PipelineState.RUNNING:
            self._state = PipelineState.PAUSED
            logger.info("Pipeline paused")

    async def resume(self):
        """Resume processing."""
        if self._state == PipelineState.PAUSED:
            self._state = PipelineState.RUNNING
            logger.info("Pipeline resumed")

    def _sdr_callback(self, samples: np.ndarray, timestamp: float):
        """
        Callback for SDR data - runs in SDR hardware thread.

        CRITICAL: This callback must return quickly to avoid SDR buffer overflow.
        Uses non-blocking try_get_write_segment() to never stall the SDR thread.

        Args:
            samples: IQ samples from SDR (complex64)
            timestamp: Sample timestamp from hardware
        """
        # Check throttle flag first (set by processing loop under backpressure)
        # This is a fast check that avoids ring buffer contention
        if self._throttle_sdr:
            # Track dropped samples during throttling
            with self._metrics_lock:
                self._metrics.total_dropped_samples += len(samples)
            return

        # Non-blocking attempt to get write segment
        result = self._ring_buffer.try_get_write_segment()

        if result is None:
            # No segment available - samples will be dropped
            # Report to SDRManager for metrics tracking
            if self._sdr_manager:
                self._sdr_manager.record_dropped_samples(len(samples))
            # Track in our own metrics too (thread-safe)
            with self._metrics_lock:
                self._metrics.total_dropped_samples += len(samples)
            logger.debug("Ring buffer full - dropping SDR samples")
            return

        idx, buffer = result

        # Copy samples to pinned buffer (fast memcpy to pinned memory)
        n = min(len(samples), len(buffer))
        buffer[:n] = samples[:n]

        # Mark ready (notifies consumer)
        self._ring_buffer.mark_segment_ready(idx, n, timestamp)

        # Update buffer fill level for backpressure tracking
        if self._sdr_manager and self._ring_buffer:
            fill_level = self._ring_buffer.peek_ready_count() / self._ring_buffer.num_segments
            self._sdr_manager.update_buffer_fill(fill_level)

        # Thread-safe metrics update (callback runs in SDR thread)
        with self._metrics_lock:
            self._metrics.total_samples += n

    async def _processing_loop(self):
        """Main processing loop."""
        logger.info("Processing loop started")

        # Get H2D stream for async transfers (if available)
        h2d_stream = None
        if self._stream_manager is not None:
            h2d_stream = self._stream_manager.get_stream("h2d")

        # Track previous frame's async transfer results (one-frame latency)
        prev_psd_db_np: np.ndarray | None = None
        prev_cfar_mask_np: np.ndarray | None = None

        while not self._stop_event.is_set():
            try:
                if self._state == PipelineState.PAUSED:
                    await asyncio.sleep(0.1)
                    continue

                # Check for backpressure - shed work if buffer filling up
                fill_level = 0.0
                under_pressure = False
                if self._ring_buffer:
                    fill_level = (
                        self._ring_buffer.peek_ready_count() / self._ring_buffer.num_segments
                    )
                    under_pressure = fill_level > self._backpressure_threshold

                    # SDR throttling with hysteresis to prevent oscillation
                    # Enable throttle above high watermark, disable below low watermark
                    if fill_level > self._throttle_high_watermark and not self._throttle_sdr:
                        self._throttle_sdr = True
                        logger.warning(
                            f"Backpressure: buffer {fill_level * 100:.0f}% full, "
                            f"throttling SDR input"
                        )
                    elif fill_level < self._throttle_low_watermark and self._throttle_sdr:
                        self._throttle_sdr = False
                        logger.info(
                            f"Backpressure relieved: buffer {fill_level * 100:.0f}% full, "
                            f"resuming SDR input"
                        )

                # Get data from ring buffer with async GPU transfer
                try:
                    idx, gpu_data = self._ring_buffer.get_read_segment(
                        timeout=self._config.pipeline.segment_timeout_seconds,
                        stream=h2d_stream,  # Async transfer on H2D stream
                    )
                except TimeoutError:
                    await asyncio.sleep(0.01)
                    continue

                # Synchronize H2D stream before compute (ensure transfer complete)
                if h2d_stream is not None:
                    h2d_stream.synchronize()

                start_time = time.perf_counter()

                # Ensure data is on GPU (defensive check)
                if CUPY_AVAILABLE and not hasattr(gpu_data, "__cuda_array_interface__"):
                    gpu_data = cp.asarray(gpu_data)

                # Stage 2: PSD computation
                # Note: freqs is ignored here - we use cached CPU version below
                _, psd = self._psd.compute_psd(gpu_data)

                # Convert to dB using pre-allocated floor constant (avoids temp allocation)
                psd_db = 10 * cp.log10(psd + self._psd.log_floor)

                # Update spectrogram ONLY if consumer is registered (skip overhead otherwise)
                if self._spectrogram_callback and self._spectrogram:
                    self._spectrogram.update_from_psd_db(psd_db)

                # Stage 3: CFAR detection
                cfar_mask, snr = self._cfar.detect(psd)

                # Stage 4: D2H transfers
                # Strategy: sync for spectrum (display needs current frame)
                #           async for detection (one-frame latency acceptable)
                freqs_np = self._psd.freq_axis_cpu  # Static, cached on CPU

                # Always get current frame's spectrum for display (sync path)
                if CUPY_AVAILABLE:
                    psd_db_np_current = cp.asnumpy(psd_db)
                else:
                    psd_db_np_current = psd_db

                # Under backpressure: skip detection entirely, just process spectrum
                detections_to_store = []
                if under_pressure:
                    # Shed detection work - just consume buffer and send spectrum
                    pass
                else:
                    # For detection: use async path to overlap with next frame's compute
                    if CUPY_AVAILABLE and self._async_transfers:
                        # transfer() queues async copy and returns PREVIOUS frame's result
                        psd_db_np_async = self._async_transfers.transfer("psd", psd_db)
                        # CFAR transferred as bool directly (no float32 conversion needed)
                        cfar_mask_np_async = self._async_transfers.transfer("cfar", cfar_mask)

                        # First frame returns None - use current frame (sync) this once
                        if psd_db_np_async is None or cfar_mask_np_async is None:
                            psd_db_np = psd_db_np_current
                            cfar_mask_np = cp.asnumpy(cfar_mask)
                        else:
                            # Use previous frame's data for detection (one-frame latency)
                            psd_db_np = psd_db_np_async
                            cfar_mask_np = cfar_mask_np_async
                    elif CUPY_AVAILABLE:
                        # Sync fallback (no async transfer manager)
                        psd_db_np = psd_db_np_current
                        cfar_mask_np = cp.asnumpy(cfar_mask)
                    else:
                        psd_db_np = psd_db
                        cfar_mask_np = cfar_mask

                    detections = self._peak_detector.find_peaks(psd_db_np, cfar_mask_np, freqs_np)

                    # Stage 5: Update tracking and store TRACKED detections
                    tracked = self._peak_detector.track_detections(detections)

                    # Store tracked detections (includes duty cycle, drift metadata)
                    # Fall back to raw detections if tracking failed
                    detections_to_store = tracked if tracked else detections

                    # Stage 5.5: AMC Classification (if available and not under pressure)
                    if detections_to_store and (self._amc_engine or self._amc_fallback):
                        detections_to_store = await self._classify_detections(
                            gpu_data, detections_to_store
                        )

                    self._recent_detections.extend(detections_to_store)

                    # Thread-safe metrics update
                    with self._metrics_lock:
                        self._metrics.total_detections += len(detections_to_store)

                    # Stage 6: Clustering (periodically, when enough detections)
                    if len(self._recent_detections) >= self._config.clustering.min_samples:
                        try:
                            clusters = self._run_clustering()
                            if clusters and self._cluster_callback:
                                self._fire_callback(self._cluster_callback, clusters)
                        except Exception as e:
                            import traceback

                            logger.warning(f"Clustering error: {e}\n{traceback.format_exc()}")

                # Mark segment processed
                self._ring_buffer.mark_segment_processed(idx)

                # Calculate latency (thread-safe update for status reads)
                latency = (time.perf_counter() - start_time) * 1000
                with self._metrics_lock:
                    self._metrics.processing_latency_ms = latency

                # Fire callbacks asynchronously (non-blocking)
                # Spectrum uses CURRENT frame (psd_db_np_current) to avoid display latency
                if self._spectrum_callback:
                    self._fire_callback(self._spectrum_callback, psd_db_np_current, freqs_np)

                if self._detection_callback and detections_to_store:
                    self._fire_callback(self._detection_callback, detections_to_store)

                # Spectrogram callback (if registered)
                if self._spectrogram_callback and self._spectrogram:
                    spectrogram_data = self._spectrogram.get_spectrogram()
                    if spectrogram_data is not None:
                        self._fire_callback(self._spectrogram_callback, spectrogram_data)

                # Reset error counter on success (thread-safe)
                with self._metrics_lock:
                    self._metrics.consecutive_errors = 0

                # Periodic stats broadcast (1 Hz)
                now = time.time()
                if self._stats_callback and (now - self._last_stats_time) >= self._stats_interval:
                    self._last_stats_time = now
                    stats = self.get_status()
                    self._fire_callback(self._stats_callback, stats)

                # Periodic cleanup of completed callback tasks
                self._cleanup_completed_callbacks()

                # Zero-delay yield to event loop (much faster than sleep(0.001))
                await asyncio.sleep(0)

            except Exception as e:
                import traceback

                # Thread-safe error counter update
                with self._metrics_lock:
                    self._metrics.consecutive_errors += 1
                    error_count = self._metrics.consecutive_errors

                logger.error(
                    f"Processing error ({error_count}/"
                    f"{self._max_consecutive_errors}): {e}\n{traceback.format_exc()}"
                )

                # Transition to ERROR state after too many consecutive failures
                if error_count >= self._max_consecutive_errors:
                    logger.critical("Too many consecutive errors, transitioning to ERROR state")
                    self._state = PipelineState.ERROR

                    # Set stop event so callers know to stop waiting
                    self._stop_event.set()

                    # Notify upstream via error callback
                    if self._error_callback:
                        try:
                            error_info = {
                                "state": "error",
                                "message": str(e),
                                "consecutive_errors": error_count,
                                "traceback": traceback.format_exc(),
                            }
                            self._fire_callback(self._error_callback, error_info)
                        except Exception as cb_err:
                            logger.error(f"Error callback failed: {cb_err}")

                    # Cleanup GPU resources on error
                    await self._cleanup_resources()
                    break

                await asyncio.sleep(0.1)

        # Flush any pending async transfers before exiting
        if self._async_transfers:
            self._async_transfers.flush_all()

        logger.info("Processing loop stopped")

    def _fire_callback(self, callback: Callable, *args):
        """
        Fire callback synchronously. The callback wrappers (e.g., on_detection_sync)
        handle the async dispatching to the event loop.
        """
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def _execute_callback(self, callback: Callable, *args):
        """Execute a callback (sync or async) with error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                # Run sync callback in executor to avoid blocking event loop
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, callback, *args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def _cleanup_completed_callbacks(self):
        """Remove completed tasks from pending set."""
        # The done_callback handles removal, but we can force cleanup
        # of any that slipped through
        done = {task for task in self._pending_callbacks if task.done()}
        self._pending_callbacks -= done

    async def _cancel_pending_callbacks(self):
        """Cancel all pending callbacks (called on shutdown)."""
        if not self._pending_callbacks:
            return

        logger.debug(f"Cancelling {len(self._pending_callbacks)} pending callbacks")

        for task in self._pending_callbacks:
            if not task.done():
                task.cancel()

        # Wait for cancellation with timeout
        if self._pending_callbacks:
            await asyncio.wait(
                self._pending_callbacks, timeout=1.0, return_when=asyncio.ALL_COMPLETED
            )

        self._pending_callbacks.clear()

    def _run_clustering(self) -> list[dict]:
        """
        Run clustering on recent detections.

        Uses enhanced features from detection properties + cached PSD data:
        - Spectral: center_freq, bandwidth, peak_power, snr
        - Shape: bandwidth ratios (3dB/6dB), spectral occupancy
        - Tracking: duty_cycle, frames_seen (if TrackedDetection)

        Returns list of cluster dicts for WebSocket/REST broadcast.
        """
        if (
            not self._clusterer
            or len(self._recent_detections) < self._config.clustering.min_samples
        ):
            return []

        # Extract enhanced features from detections
        sample_rate = self._config.sdr.sample_rate_hz
        center_freq = self._config.sdr.center_freq_hz

        features = []
        for det in self._recent_detections:
            # Core spectral features (normalized)
            feat = [
                det.center_freq_hz / (center_freq * 2),  # ~0.5 centered
                det.bandwidth_hz / sample_rate,  # 0-1 normalized
                (det.peak_power_db + 120) / 120,  # Shift to 0-1 range
                det.snr_db / 60,  # Assume max 60 dB SNR
            ]

            # Bandwidth shape features (3dB/6dB ratio indicates modulation type)
            bw_3db = getattr(det, "bandwidth_3db_hz", det.bandwidth_hz)
            bw_6db = getattr(det, "bandwidth_6db_hz", det.bandwidth_hz)
            if det.bandwidth_hz > 0:
                feat.append(bw_3db / det.bandwidth_hz)  # 3dB ratio
                feat.append(bw_6db / det.bandwidth_hz)  # 6dB ratio
            else:
                feat.extend([0.5, 0.7])  # Defaults

            # Spectral occupancy (bins used / FFT size)
            bin_span = getattr(det, "end_bin", 0) - getattr(det, "start_bin", 0)
            feat.append(bin_span / self._config.fft.fft_size if bin_span > 0 else 0.01)

            # Tracking features (if available from TrackedDetection)
            duty_cycle = getattr(det, "duty_cycle", 1.0)
            frames_seen = getattr(det, "frames_seen", 1)
            feat.append(duty_cycle)
            feat.append(min(frames_seen / 100.0, 1.0))  # Normalize to 0-1

            features.append(feat)

        features_array = np.array(features, dtype=np.float32)

        # Run clustering
        if CUPY_AVAILABLE:
            features_gpu = cp.asarray(features_array)
            labels = self._clusterer.fit(features_gpu)
            # Ensure labels are numpy for iteration
            if hasattr(labels, "get"):
                labels = labels.get()
        else:
            labels = self._clusterer.fit(features_array)

        # Convert cluster info to dicts for broadcast
        cluster_list = []
        for cluster_info in self._clusterer.get_all_clusters():
            # Find detections in this cluster
            cluster_detections = [
                det
                for det, label in zip(self._recent_detections, labels)
                if label == cluster_info.cluster_id
            ]

            if cluster_detections:
                # Compute cluster stats
                freqs = [d.center_freq_hz for d in cluster_detections]
                snrs = [d.snr_db for d in cluster_detections]
                bandwidths = [d.bandwidth_hz for d in cluster_detections]
                powers = [d.peak_power_db for d in cluster_detections]

                cluster_dict = {
                    "cluster_id": int(cluster_info.cluster_id),
                    "size": cluster_info.size,
                    "center_freq_hz": float(np.mean(freqs)),
                    "freq_range_hz": [float(min(freqs)), float(max(freqs))],
                    "avg_snr_db": float(np.mean(snrs)),
                    "avg_power_db": float(np.mean(powers)),
                    "avg_bandwidth_hz": float(np.mean(bandwidths)),
                    "detection_count": len(cluster_detections),
                    "label": cluster_info.label or f"Cluster_{cluster_info.cluster_id}",
                }

                # Bandwidth shape analysis (helps identify modulation type)
                bw_3db_ratios = []
                for d in cluster_detections:
                    bw_3db = getattr(d, "bandwidth_3db_hz", d.bandwidth_hz)
                    if d.bandwidth_hz > 0:
                        bw_3db_ratios.append(bw_3db / d.bandwidth_hz)
                if bw_3db_ratios:
                    cluster_dict["avg_bw_3db_ratio"] = float(np.mean(bw_3db_ratios))

                # Include tracking stats if available (TrackedDetection fields)
                duty_cycles = [getattr(d, "duty_cycle", None) for d in cluster_detections]
                duty_cycles = [dc for dc in duty_cycles if dc is not None]
                if duty_cycles:
                    cluster_dict["avg_duty_cycle"] = float(np.mean(duty_cycles))

                # Include unique track count
                track_ids = [getattr(d, "track_id", None) for d in cluster_detections]
                track_ids = [tid for tid in track_ids if tid is not None]
                if track_ids:
                    cluster_dict["unique_tracks"] = len(set(track_ids))

                # Signal type hint based on bandwidth and duty cycle
                avg_bw = np.mean(bandwidths)
                avg_duty = np.mean(duty_cycles) if duty_cycles else 1.0
                cluster_dict["signal_type_hint"] = self._classify_cluster(avg_bw, avg_duty)

                cluster_list.append(cluster_dict)

        return cluster_list

    def _classify_cluster(self, avg_bandwidth_hz: float, avg_duty_cycle: float) -> str:
        """
        Classify signal type based on bandwidth and duty cycle patterns.

        Uses heuristics from common RF signal characteristics:
        - Narrowband CW: <25 kHz, continuous (>90% duty)
        - Narrowband burst: <25 kHz, intermittent (<50% duty)
        - Voice/FM: 25-200 kHz, continuous
        - Wideband digital: 200 kHz - 5 MHz, high duty
        - Spread spectrum: >1 MHz, continuous
        - Radar/pulse: any bandwidth, very low duty (<20%)

        Args:
            avg_bandwidth_hz: Average bandwidth of cluster detections
            avg_duty_cycle: Average duty cycle (0-1) from tracking

        Returns:
            Signal type hint string
        """
        # Bandwidth thresholds
        narrowband = avg_bandwidth_hz < 25_000
        medium_band = 25_000 <= avg_bandwidth_hz < 200_000
        wideband = 200_000 <= avg_bandwidth_hz < 1_000_000
        spread = avg_bandwidth_hz >= 1_000_000

        # Duty cycle thresholds
        continuous = avg_duty_cycle > 0.90
        intermittent = 0.50 <= avg_duty_cycle <= 0.90
        burst = 0.20 <= avg_duty_cycle < 0.50
        pulse = avg_duty_cycle < 0.20

        # Classification logic
        if pulse:
            # Very low duty cycle suggests radar or pulsed systems
            if wideband or spread:
                return "radar_pulse"
            return "pulsed_beacon"

        if narrowband:
            if continuous:
                return "narrowband_cw"  # CW beacon, telemetry
            elif burst:
                return "narrowband_burst"  # Digital packet, SCADA
            return "narrowband_intermittent"

        if medium_band:
            if continuous:
                return "voice_fm"  # Analog voice, FM broadcast
            elif intermittent:
                return "digital_voice"  # DMR, P25, DPMR
            return "medium_band_burst"

        if wideband:
            if continuous:
                return "wideband_digital"  # WiFi, LTE
            return "wideband_burst"

        if spread:
            if continuous:
                return "spread_spectrum"  # CDMA, frequency hopping
            return "frequency_hopping"

        return "unknown"

    async def _classify_detections(
        self,
        gpu_data: cp.ndarray,
        detections: list[Detection],
    ) -> list[Detection]:
        """
        Classify modulation type for each detection.

        Extracts IQ segments corresponding to each detection using proper
        frequency isolation (frequency shift + lowpass filter) and classifies
        using the AMC neural network or cumulant fallback.

        Args:
            gpu_data: Full IQ buffer on GPU.
            detections: List of detections to classify.

        Returns:
            List of detections with modulation fields populated.
        """
        if not detections:
            return detections

        sample_rate = self._config.sdr.sample_rate_hz
        center_freq = self._config.sdr.center_freq_hz
        segment_len = 1024  # Matches AMC model input

        # Extract IQ segments for each detection with proper frequency isolation
        segments = []
        snr_list = []
        features_list = []
        valid_detections = []  # Track which detections have valid segments

        for det in detections:
            try:
                # Extract frequency-isolated segment for this detection
                segment = self._extract_detection_segment_gpu(
                    gpu_data, det, center_freq, sample_rate, segment_len
                )

                if segment is not None and len(segment) >= 256:
                    segments.append(segment)
                    snr_list.append(det.snr_db)
                    valid_detections.append(det)

                    # Extract features for AMC engine (if feature extractor available)
                    if self._feature_extractor:
                        try:
                            # Convert to numpy for feature extraction
                            seg_np = segment.get() if hasattr(segment, "get") else segment
                            features = self._feature_extractor.extract(seg_np)
                            features_list.append(features)
                        except Exception:
                            features_list.append(None)
                    else:
                        features_list.append(None)

            except Exception as e:
                logger.debug(f"Segment extraction failed for detection {det.detection_id}: {e}")
                continue

        if not segments:
            return detections

        try:
            # Classify using TorchSig > Neural network > Cumulant fallback
            results = None

            # Convert CuPy arrays to numpy for classification
            np_segments = [seg.get() if hasattr(seg, "get") else seg for seg in segments]

            if self._torchsig_adapter and self._torchsig_adapter.is_available:
                # TorchSig batch classification (57 modulation types)
                results = self._torchsig_adapter.classify_batch(np_segments, snr_list)
            elif self._amc_engine and self._amc_engine.is_available:
                # Custom AMC neural network - now with features!
                valid_features = [f for f in features_list if f is not None]
                results = await self._amc_engine.classify_batch(
                    segments,
                    features_list=valid_features if len(valid_features) == len(segments) else None,
                    snr_list=snr_list,
                )
            elif self._amc_fallback:
                # Cumulant-based classifier
                results = self._amc_fallback.batch_classify(np_segments, snr_list)

            if results is None:
                return detections

            # Update detections with classification results
            for det, result in zip(valid_detections, results):
                # Normalize modulation name for consistency across backends
                if AMC_AVAILABLE:
                    det.modulation_type = normalize_modulation_name(result.modulation_type)
                else:
                    det.modulation_type = result.modulation_type
                det.modulation_confidence = result.confidence

                # Normalize top-k predictions as well
                top_k = getattr(result, "top_k_predictions", [])
                if top_k and AMC_AVAILABLE:
                    det.top_k_predictions = [
                        (normalize_modulation_name(mod), conf) for mod, conf in top_k
                    ]
                else:
                    det.top_k_predictions = top_k

                # Estimate symbol rate from bandwidth (rough estimate)
                # Most modulations: symbol_rate ≈ bandwidth / (1 + excess_bandwidth)
                # Assume typical excess bandwidth of 0.35 (root raised cosine)
                det.symbol_rate = det.bandwidth_hz / 1.35

        except Exception as e:
            logger.warning(f"AMC classification error: {e}")
            # Leave modulation fields at default values

        # Compute anomaly scores for detections (per BACKEND_CONTRACT.md)
        if self._anomaly_detector and self._anomaly_detector.available:
            try:
                anomaly_scores = self._anomaly_detector.score_batch(detections)
                for det, score in zip(detections, anomaly_scores):
                    det.anomaly_score = score
            except Exception as e:
                logger.debug(f"Anomaly scoring error: {e}")
                # Leave anomaly_score at default (None)

        return detections

    def _extract_detection_segment_gpu(
        self,
        gpu_data: cp.ndarray,
        detection: Detection,
        center_freq: float,
        sample_rate: float,
        segment_len: int = 1024,
    ) -> cp.ndarray | None:
        """
        Extract IQ segment centered on detection frequency using GPU acceleration.

        This properly isolates the signal of interest by:
        1. Frequency-shifting the detection to baseband (DC)
        2. Applying a lowpass filter at the detection's bandwidth
        3. Extracting the center segment

        Args:
            gpu_data: Full IQ buffer on GPU.
            detection: Detection to extract segment for.
            center_freq: SDR center frequency in Hz.
            sample_rate: SDR sample rate in Hz.
            segment_len: Target segment length.

        Returns:
            Isolated IQ segment on GPU, or None if extraction fails.
        """
        if not CUPY_AVAILABLE:
            # Fallback to numpy
            return self._extract_detection_segment_cpu(
                gpu_data, detection, center_freq, sample_rate, segment_len
            )

        n = len(gpu_data)
        if n < segment_len:
            return None

        # Calculate frequency offset from SDR center
        freq_offset = detection.center_freq_hz - center_freq

        # Frequency shift to baseband using complex exponential
        # This moves the detection's center frequency to DC
        t = cp.arange(n, dtype=cp.float64) / sample_rate
        shift = cp.exp(-2j * cp.pi * freq_offset * t).astype(cp.complex64)
        baseband = gpu_data * shift

        # Apply lowpass filter at detection bandwidth
        # Use FFT-based filtering for GPU efficiency
        bw_hz = max(detection.bandwidth_hz, sample_rate / 100)  # Minimum 1% of sample rate
        bw_ratio = min(0.45, bw_hz / sample_rate)  # Max 45% to avoid aliasing

        filtered = self._lowpass_filter_fft_gpu(baseband, bw_ratio)

        # Extract center segment
        center = n // 2
        start = max(0, center - segment_len // 2)
        end = min(n, start + segment_len)

        return filtered[start:end]

    def _lowpass_filter_fft_gpu(self, signal: cp.ndarray, cutoff_ratio: float) -> cp.ndarray:
        """
        Apply lowpass filter using FFT on GPU.

        Args:
            signal: Complex signal on GPU.
            cutoff_ratio: Cutoff frequency as ratio of sample rate (0-0.5).

        Returns:
            Filtered signal on GPU.
        """
        n = len(signal)

        # FFT
        spectrum = cp.fft.fft(signal)

        # Create lowpass mask (brick-wall filter for simplicity)
        # In practice, a smoother transition would reduce ringing
        cutoff_bin = int(n * cutoff_ratio)

        mask = cp.zeros(n, dtype=cp.float32)
        mask[:cutoff_bin] = 1.0
        mask[-cutoff_bin:] = 1.0  # Symmetric for complex signal

        # Apply Hann window for smoother transition (reduces ringing)
        if cutoff_bin > 0:
            transition = int(cutoff_bin * 0.1)  # 10% transition band
            if transition > 1:
                hann_half = cp.hanning(transition * 2)[transition:]
                mask[cutoff_bin : cutoff_bin + transition] = hann_half
                mask[n - cutoff_bin - transition : n - cutoff_bin] = hann_half[::-1]

        # Apply filter and inverse FFT
        filtered_spectrum = spectrum * mask
        filtered = cp.fft.ifft(filtered_spectrum)

        return filtered.astype(cp.complex64)

    def _extract_detection_segment_cpu(
        self,
        data: np.ndarray,
        detection: Detection,
        center_freq: float,
        sample_rate: float,
        segment_len: int = 1024,
    ) -> np.ndarray | None:
        """CPU fallback for segment extraction."""
        n = len(data)
        if n < segment_len:
            return None

        # Frequency shift to baseband
        freq_offset = detection.center_freq_hz - center_freq
        t = np.arange(n, dtype=np.float64) / sample_rate
        shift = np.exp(-2j * np.pi * freq_offset * t).astype(np.complex64)
        baseband = data * shift

        # Simple lowpass using scipy if available, else skip
        try:
            from scipy.signal import firwin, lfilter

            bw_hz = max(detection.bandwidth_hz, sample_rate / 100)
            bw_ratio = min(0.45, bw_hz / sample_rate)

            # Design lowpass filter
            numtaps = min(101, n // 10)
            if numtaps % 2 == 0:
                numtaps += 1
            taps = firwin(numtaps, bw_ratio * 2, window="hamming")
            filtered = lfilter(taps, 1, baseband)
        except ImportError:
            filtered = baseband  # No filtering if scipy unavailable

        # Extract center segment
        center = n // 2
        start = max(0, center - segment_len // 2)
        end = min(n, start + segment_len)

        return filtered[start:end]

    def _capture_event_loop(self):
        """Capture event loop when callbacks are set from async context."""
        if self._event_loop is None:
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass  # Not in async context yet

    def set_spectrum_callback(self, callback: Callable):
        """Set callback for spectrum updates."""
        self._spectrum_callback = callback
        self._capture_event_loop()

    def set_detection_callback(self, callback: Callable):
        """Set callback for detection events."""
        self._detection_callback = callback
        self._capture_event_loop()

    def set_cluster_callback(self, callback: Callable):
        """Set callback for cluster updates."""
        self._cluster_callback = callback
        self._capture_event_loop()

    def set_spectrogram_callback(self, callback: Callable):
        """Set callback for spectrogram updates (enables spectrogram computation)."""
        self._spectrogram_callback = callback
        self._capture_event_loop()

    def set_error_callback(self, callback: Callable):
        """
        Set callback for error notifications.

        Called when pipeline transitions to ERROR state with info dict:
        {
            "state": "error",
            "message": "error description",
            "consecutive_errors": count,
            "traceback": "full traceback"
        }
        """
        self._error_callback = callback
        self._capture_event_loop()

    def set_stats_callback(self, callback: Callable):
        """
        Set callback for periodic statistics updates.

        Called at ~1 Hz with pipeline status dict from get_status().
        Useful for exposing drop rates, throughput, etc. to WebSocket clients.
        """
        self._stats_callback = callback
        self._capture_event_loop()

    def update_config(self, updates: dict[str, Any]):
        """
        Update configuration at runtime with validation.

        Uses Pydantic model validation to ensure updates are valid before applying.
        Only reconfigures components whose config sections have changed.

        Args:
            updates: Dict of section updates, e.g. {"sdr": {...}, "fft": {...}}

        Raises:
            ValueError: If updates fail validation.
        """
        from pydantic import ValidationError

        from rf_forensics.config.schema import RFForensicsConfig

        # Build update dict for Pydantic model_copy
        update_data = {}
        for section, values in updates.items():
            if hasattr(self._config, section) and values is not None:
                if isinstance(values, dict):
                    # Merge with existing section config
                    current_section = getattr(self._config, section)
                    # Handle case where section might already be a dict (corruption recovery)
                    if hasattr(current_section, "model_dump"):
                        merged = current_section.model_dump()
                    else:
                        merged = dict(current_section) if current_section else {}
                    merged.update({k: v for k, v in values.items() if v is not None})
                    update_data[section] = merged
                else:
                    update_data[section] = values

        # Validate with Pydantic before applying - reconstruct full config
        try:
            # Get current config as dict, merge updates, create new validated config
            current_dict = (
                self._config.model_dump()
                if hasattr(self._config, "model_dump")
                else dict(self._config)
            )
            for section, values in update_data.items():
                current_dict[section] = values
            new_config = RFForensicsConfig(**current_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid config update: {e}")

        # Detect which sections changed
        changed_sections = self._detect_config_changes(self._config, new_config)

        # Apply validated config
        old_config = self._config
        self._config = new_config

        # Reconfigure only changed components
        if self._state == PipelineState.RUNNING and changed_sections:
            logger.info(f"Reconfiguring components for changed sections: {changed_sections}")
            self._reconfigure_changed_components(changed_sections, old_config)

    def _detect_config_changes(self, old: RFForensicsConfig, new: RFForensicsConfig) -> set:
        """
        Detect which configuration sections changed.

        Args:
            old: Previous configuration.
            new: New configuration.

        Returns:
            Set of changed section names.
        """
        changes = set()

        if old.sdr != new.sdr:
            changes.add("sdr")
        if old.fft != new.fft:
            changes.add("fft")
        if old.cfar != new.cfar:
            changes.add("cfar")
        if old.buffer != new.buffer:
            changes.add("buffer")
        if old.clustering != new.clustering:
            changes.add("clustering")

        return changes

    def _reconfigure_changed_components(self, changes: set, old_config: RFForensicsConfig):
        """
        Reconfigure only the components affected by config changes.

        Properly cleans up old components before creating new ones.

        Args:
            changes: Set of changed section names.
            old_config: Previous configuration (for comparison).
        """
        # SDR config changed - reconfigure hardware via manager
        if "sdr" in changes:
            if self._sdr_manager:
                logger.info("Reconfiguring SDR...")
                # Build USDR config from pipeline config
                gain_db = int(self._config.sdr.gain_db)
                lna_gain = min(30, gain_db)
                remaining = gain_db - lna_gain
                tia_gain = min(12, remaining)
                pga_gain = min(32, remaining - tia_gain)

                usdr_config = USDRConfig(
                    center_freq_hz=int(self._config.sdr.center_freq_hz),
                    sample_rate_hz=int(self._config.sdr.sample_rate_hz),
                    bandwidth_hz=int(self._config.sdr.bandwidth_hz),
                    gain=USDRGain(lna_db=lna_gain, tia_db=tia_gain, pga_db=pga_gain),
                )
                self._sdr_manager.configure(usdr_config)

        # FFT config changed - recreate PSD and potentially ring buffer
        if "fft" in changes:
            logger.info("Reconfiguring FFT/PSD...")
            if self._psd:
                self._psd.cleanup()
            self._psd = WelchPSD(
                fft_size=self._config.fft.fft_size,
                overlap=self._config.fft.overlap_percent / 100,
                window=self._config.fft.window_type,
                sample_rate=self._config.sdr.sample_rate_hz,
            )

            # Check if spectrogram needs rebuild
            if self._spectrogram:
                self._spectrogram.cleanup()
                self._spectrogram = SpectrogramBuffer(
                    num_time_bins=256,
                    fft_size=self._config.fft.fft_size,
                    sample_rate=self._config.sdr.sample_rate_hz,
                )

        # Buffer config changed - may need ring buffer resize
        if "buffer" in changes:
            logger.info("Reconfiguring ring buffer...")
            old_size = old_config.buffer.samples_per_buffer
            new_size = self._config.buffer.samples_per_buffer

            if old_size != new_size and self._ring_buffer:
                # Must recreate ring buffer with new size
                self._ring_buffer.cleanup()
                self._ring_buffer = GPURingBuffer(
                    num_segments=self._config.buffer.num_ring_segments,
                    samples_per_segment=new_size,
                    use_gpu_buffer=CUPY_AVAILABLE,
                )

        # CFAR config changed - recreate detector
        if "cfar" in changes:
            logger.info("Reconfiguring CFAR detector...")
            self._cfar = CFARDetector(
                num_reference=self._config.cfar.num_reference_cells,
                num_guard=self._config.cfar.num_guard_cells,
                pfa=self._config.cfar.pfa,
                variant=self._config.cfar.variant,
            )

        # Clustering config changed - update clusterer params
        if "clustering" in changes:
            logger.info("Reconfiguring clusterer...")
            if self._clusterer:
                self._clusterer = EmitterClusterer(
                    eps=self._config.clustering.eps,
                    min_samples=self._config.clustering.min_samples,
                    auto_tune=self._config.clustering.auto_tune,
                )

    def apply_preset(self, preset_name: str):
        """Apply a configuration preset."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}")

        old_config = self._config
        self._config = PRESETS[preset_name].model_copy(deep=True)

        if self._state == PipelineState.RUNNING:
            # Detect all changes and reconfigure
            changes = self._detect_config_changes(old_config, self._config)
            if changes:
                logger.info(f"Applying preset '{preset_name}', reconfiguring: {changes}")
                self._reconfigure_changed_components(changes, old_config)

    def get_status(self) -> dict:
        """Get current pipeline status."""
        uptime = time.time() - self._metrics.start_time if self._metrics.start_time else 0

        # Thread-safe read of metrics
        with self._metrics_lock:
            total_samples = self._metrics.total_samples
            total_detections = self._metrics.total_detections
            consecutive_errors = self._metrics.consecutive_errors
            dropped_samples = self._metrics.total_dropped_samples
            latency_ms = self._metrics.processing_latency_ms

        # Calculate throughput
        throughput = total_samples / uptime / 1e6 if uptime > 0 else 0

        # GPU memory
        gpu_mem_gb = 0.0
        if CUPY_AVAILABLE:
            try:
                mem_info = cp.cuda.runtime.memGetInfo()
                used = (mem_info[1] - mem_info[0]) / (1024**3)
                gpu_mem_gb = used
            except Exception:
                pass

        # Buffer fill level
        fill_level = 0.0
        if self._ring_buffer:
            status = self._ring_buffer.get_status()
            fill_level = status.get("fill_level", 0.0)

        return {
            "state": self._state.value,
            "uptime_seconds": uptime,
            "samples_processed": total_samples,
            "detections_count": total_detections,
            "current_throughput_msps": throughput,
            "gpu_memory_used_gb": gpu_mem_gb,
            "buffer_fill_level": fill_level,
            "processing_latency_ms": latency_ms,
            "consecutive_errors": consecutive_errors,
            "dropped_samples": dropped_samples,
            "sdr_throttled": self._throttle_sdr,
        }

    def get_recent_detections(self, limit: int = 100) -> list:
        """Get recent detections."""
        return self._recent_detections[-limit:]


async def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    pipeline = RFForensicsPipeline()

    try:
        await pipeline.start()

        # Run until interrupted
        while True:
            status = pipeline.get_status()
            logger.info(
                f"Status: {status['state']}, "
                f"Throughput: {status['current_throughput_msps']:.2f} MSps, "
                f"Detections: {status['detections_count']}"
            )
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(main())
