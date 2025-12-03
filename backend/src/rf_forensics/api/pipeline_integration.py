"""
Pipeline-WebSocket Integration

Wires the RF Forensics Pipeline to the WebSocket server for real-time
frontend updates with camelCase JSON formatting.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from rf_forensics.api.websocket_server import SpectrumWebSocketServer

logger = logging.getLogger(__name__)


class PipelineWebSocketBridge:
    """
    Bridge between RFForensicsPipeline and FrontendWebSocketServer.

    Converts pipeline output (snake_case) to frontend format (camelCase)
    and broadcasts via WebSocket JSON messages.
    Also feeds detections/clusters to REST API store for GET endpoints.
    """

    def __init__(
        self,
        ws_server: SpectrumWebSocketServer,
        api_manager=None,
        center_freq_hz: int = 915_000_000,
        sample_rate_hz: int = 10_000_000,
    ):
        """
        Initialize bridge.

        Args:
            ws_server: WebSocket server instance
            api_manager: REST API manager for storing detections/clusters
            center_freq_hz: Current center frequency
            sample_rate_hz: Current sample rate
        """
        self._ws_server = ws_server
        self._api_manager = api_manager  # For REST endpoint data
        self._center_freq_hz = center_freq_hz
        self._sample_rate_hz = sample_rate_hz
        self._fft_size = 2048

        # Event loop for cross-thread callbacks (set when first needed)
        self._loop = None

        # Rate limiting
        self._last_spectrum_time = 0.0
        self._spectrum_interval = 0.05  # 20 Hz

        self._last_cluster_time = 0.0
        self._cluster_interval = 0.5  # 2 Hz

        self._last_stats_time = 0.0
        self._stats_interval = 1.0  # 1 Hz (stats don't need high rate)

        # Detection counter for IDs
        self._detection_counter = 0

        # Cluster cache
        self._clusters: list[dict] = []

    def update_sdr_config(self, center_freq_hz: int, sample_rate_hz: int, fft_size: int = None):
        """Update SDR configuration for correct frequency axis generation."""
        self._center_freq_hz = center_freq_hz
        self._sample_rate_hz = sample_rate_hz
        if fft_size:
            self._fft_size = fft_size

    async def on_spectrum(self, psd_db: np.ndarray, freqs: np.ndarray = None):
        """
        Pipeline callback for spectrum updates.

        Args:
            psd_db: Power spectral density in dB (numpy array)
            freqs: Frequency bins (optional, will generate if not provided)
        """
        # Rate limiting
        now = time.time()
        if now - self._last_spectrum_time < self._spectrum_interval:
            return
        self._last_spectrum_time = now

        # Broadcast as JSON (frontend format)
        await self._ws_server.broadcast_spectrum_json(
            psd_db=psd_db,
            center_freq_hz=self._center_freq_hz,
            sample_rate_hz=self._sample_rate_hz,
            fft_size=len(psd_db),
        )

    async def on_detection(self, detections: list):
        """
        Pipeline callback for detection events.

        Handles both raw Detection and TrackedDetection objects.
        TrackedDetection includes additional tracking metadata (track_id,
        frames_seen, duty_cycle, etc.) that is preserved when present.

        Args:
            detections: List of Detection/TrackedDetection objects or dicts
        """
        for det in detections:
            self._detection_counter += 1

            # Convert Detection object to dict if needed
            if hasattr(det, "center_freq_hz"):
                det_dict = {
                    # Core detection fields (always present)
                    "detection_id": self._detection_counter,
                    "center_freq_hz": det.center_freq_hz,
                    "bandwidth_hz": det.bandwidth_hz,
                    "peak_power_db": det.peak_power_db,
                    "snr_db": det.snr_db,
                    "modulation_type": getattr(det, "modulation_type", "Unknown"),
                    "confidence": getattr(det, "confidence", 0.0),
                    "timestamp": getattr(det, "timestamp", 0.0),
                    # Tracking fields (present in TrackedDetection)
                    "track_id": getattr(det, "track_id", None),
                    "frames_seen": getattr(det, "frames_seen", None),
                    "duty_cycle": getattr(det, "duty_cycle", None),
                    "first_seen": getattr(det, "first_seen", None),
                    "last_seen": getattr(det, "last_seen", None),
                    "frequency_drift_hz_per_s": getattr(det, "frequency_drift_hz_per_s", None),
                }
                # Remove None values to keep payload clean for raw detections
                det_dict = {k: v for k, v in det_dict.items() if v is not None}
            else:
                det_dict = dict(det)
                det_dict.setdefault("detection_id", self._detection_counter)

            # Store in API manager for REST endpoints (GET /api/detections)
            if self._api_manager:
                self._api_manager.add_detection(det_dict)

            # Send as JSON (frontend format) via WebSocket
            await self._ws_server.send_detection_json(det_dict)

    async def on_clusters(self, clusters: list):
        """
        Pipeline callback for cluster updates.

        Args:
            clusters: List of cluster dicts
        """
        # Rate limiting
        now = time.time()
        if now - self._last_cluster_time < self._cluster_interval:
            return
        self._last_cluster_time = now

        self._clusters = clusters

        # Store in API manager for REST endpoints (GET /api/clusters)
        if self._api_manager:
            self._api_manager.update_clusters(clusters)

        # Send as JSON (frontend format) via WebSocket
        await self._ws_server.send_clusters_json(clusters)

    async def on_stats(self, stats: dict):
        """
        Pipeline callback for statistics updates.

        Args:
            stats: Dict with pipeline status (from pipeline.get_status())
        """
        # Rate limiting - stats only need 1 Hz
        now = time.time()
        if now - self._last_stats_time < self._stats_interval:
            return
        self._last_stats_time = now

        # Send as JSON (frontend format) via WebSocket
        await self._ws_server.send_stats_json(stats)

    def _get_loop(self):
        """Get event loop, caching it for efficiency."""
        if self._loop is None or not self._loop.is_running():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop - callbacks will be dropped
                logger.warning("No running event loop - WebSocket callbacks will be dropped!")
                return None
        return self._loop

    def on_detection_sync(self, detections: list):
        """Synchronous wrapper for detection callback (thread-safe)."""
        loop = self._get_loop()
        if loop:
            asyncio.run_coroutine_threadsafe(self.on_detection(detections), loop)
        else:
            logger.debug(f"Dropped {len(detections)} detection(s) - no event loop")

    def on_spectrum_sync(self, psd_db: np.ndarray, freqs: np.ndarray = None):
        """Synchronous wrapper for spectrum callback (thread-safe)."""
        loop = self._get_loop()
        if loop:
            asyncio.run_coroutine_threadsafe(self.on_spectrum(psd_db, freqs), loop)
        # Note: Don't log spectrum drops - would spam at 20Hz

    def on_stats_sync(self, stats: dict):
        """Synchronous wrapper for stats callback (thread-safe)."""
        loop = self._get_loop()
        if loop:
            asyncio.run_coroutine_threadsafe(self.on_stats(stats), loop)
        else:
            logger.debug("Dropped stats update - no event loop")

    def on_clusters_sync(self, clusters: list):
        """Synchronous wrapper for cluster callback (thread-safe)."""
        loop = self._get_loop()
        if loop:
            asyncio.run_coroutine_threadsafe(self.on_clusters(clusters), loop)
        else:
            logger.debug(f"Dropped {len(clusters)} cluster(s) - no event loop")

    async def on_lora_frame(self, frame: dict):
        """
        Pipeline callback for LoRa frame decoded events.

        Args:
            frame: Dict with LoRa frame data (from LoRaFrame.to_frontend_dict())
        """
        # Send as JSON via WebSocket
        await self._ws_server.send_lora_frame_json(frame)

    def on_lora_frame_sync(self, frame: dict):
        """Synchronous wrapper for LoRa frame callback (thread-safe)."""
        loop = self._get_loop()
        if loop:
            asyncio.run_coroutine_threadsafe(self.on_lora_frame(frame), loop)
        else:
            logger.debug("Dropped LoRa frame - no event loop")


def connect_pipeline_to_websocket(
    pipeline, ws_server: SpectrumWebSocketServer, api_manager=None
) -> PipelineWebSocketBridge:
    """
    Connect an RFForensicsPipeline to a WebSocket server.

    Args:
        pipeline: RFForensicsPipeline instance
        ws_server: SpectrumWebSocketServer instance
        api_manager: Optional REST API manager for feeding detections/clusters

    Returns:
        PipelineWebSocketBridge instance
    """
    # Get config from pipeline
    center_freq = getattr(pipeline.config.sdr, "center_freq_hz", 915_000_000)
    sample_rate = getattr(pipeline.config.sdr, "sample_rate_hz", 10_000_000)
    fft_size = getattr(pipeline.config.fft, "fft_size", 2048)

    # Create bridge with both WebSocket and REST API manager
    bridge = PipelineWebSocketBridge(
        ws_server=ws_server,
        api_manager=api_manager,
        center_freq_hz=center_freq,
        sample_rate_hz=sample_rate,
    )
    bridge._fft_size = fft_size

    # Set callbacks on pipeline (all use sync wrappers for thread safety)
    pipeline.set_spectrum_callback(bridge.on_spectrum_sync)
    pipeline.set_detection_callback(bridge.on_detection_sync)
    pipeline.set_cluster_callback(bridge.on_clusters_sync)
    pipeline.set_stats_callback(bridge.on_stats_sync)

    logger.info(
        f"Pipeline connected to WebSocket server "
        f"(freq={center_freq / 1e6:.1f} MHz, rate={sample_rate / 1e6:.1f} MSps)"
    )

    return bridge


async def run_integrated_server(rest_port: int = 8000, ws_port: int = 8765, simulate: bool = False):
    """
    Run fully integrated REST + WebSocket server with pipeline.

    This is the main entry point for production deployment.
    """
    from multiprocessing import Process

    import uvicorn

    from rf_forensics.api.rest_api import RFForensicsAPI, create_rest_api
    from rf_forensics.api.websocket_server import SpectrumWebSocketServer, create_websocket_app
    from rf_forensics.pipeline.orchestrator import RFForensicsPipeline

    # Create components
    api_manager = RFForensicsAPI()
    ws_server = SpectrumWebSocketServer(port=ws_port)

    if not simulate:
        # Create and connect pipeline
        pipeline = RFForensicsPipeline()
        api_manager.set_pipeline(pipeline)
        # Pass api_manager so detections/clusters also feed REST endpoints
        bridge = connect_pipeline_to_websocket(pipeline, ws_server, api_manager)

    # Create apps
    rest_app = create_rest_api(api_manager)
    ws_app = create_websocket_app(ws_server)

    # Run servers
    def run_rest():
        uvicorn.run(rest_app, host="0.0.0.0", port=rest_port)

    def run_ws():
        uvicorn.run(ws_app, host="0.0.0.0", port=ws_port)

    rest_proc = Process(target=run_rest)
    ws_proc = Process(target=run_ws)

    rest_proc.start()
    ws_proc.start()

    try:
        if not simulate:
            # Start pipeline
            await pipeline.start()

        # Wait for processes in a non-blocking way
        # Using asyncio.to_thread() to avoid freezing the event loop
        # that the pipeline processing loop needs
        def wait_for_processes():
            """Blocking wait in a thread."""
            rest_proc.join()
            ws_proc.join()

        await asyncio.to_thread(wait_for_processes)

    except KeyboardInterrupt:
        if not simulate:
            await pipeline.stop()
        rest_proc.terminate()
        ws_proc.terminate()
