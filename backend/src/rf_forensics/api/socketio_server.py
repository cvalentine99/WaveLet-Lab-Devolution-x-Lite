"""
GPU RF Forensics Engine - Socket.IO Server

Socket.IO WebSocket server matching Devolution(x) Spectrum Detect frontend.
Uses namespaces: /spectrum, /detections, /analytics, /hardware
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

try:
    import socketio
    from aiohttp import web

    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    socketio = None
    web = None

from rf_forensics.api.frontend_adapter import FrontendMessageFormatter

logger = logging.getLogger(__name__)


class RFForensicsSocketIO:
    """
    Socket.IO server for Devolution(x) Spectrum Detect frontend.

    Namespaces (matching frontend expectations):
    - /spectrum: Real-time spectrum measurements
    - /detections: Threat detection events
    - /analytics: Performance metrics
    - /hardware: SDR device telemetry

    Events:
    - spectrum:measurement
    - detection:new, detection:update
    - analytics:metrics
    - hardware:status
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        cors_allowed_origins: str = "*",
        spectrum_rate_hz: float = 20.0,
    ):
        """
        Initialize Socket.IO server.

        Args:
            host: Bind address
            port: Port number
            cors_allowed_origins: CORS allowed origins
            spectrum_rate_hz: Target spectrum update rate
        """
        if not SOCKETIO_AVAILABLE:
            raise ImportError(
                "python-socketio and aiohttp required: pip install python-socketio aiohttp"
            )

        self._host = host
        self._port = port
        self._spectrum_rate_hz = spectrum_rate_hz
        self._min_broadcast_interval = 1.0 / spectrum_rate_hz
        self._last_spectrum_broadcast = 0.0

        # Create Socket.IO server
        self._sio = socketio.AsyncServer(
            async_mode="aiohttp",
            cors_allowed_origins=cors_allowed_origins,
            logger=False,
            engineio_logger=False,
        )

        # Create aiohttp web app
        self._app = web.Application()
        self._sio.attach(self._app)

        # Track connected clients per namespace
        self._clients: dict[str, set[str]] = {
            "/spectrum": set(),
            "/detections": set(),
            "/analytics": set(),
            "/hardware": set(),
        }

        # Detection counter for IDs
        self._detection_counter = 0

        # Register event handlers
        self._register_handlers()

    def _register_handlers(self):
        """Register Socket.IO event handlers for all namespaces."""

        # Default namespace handlers
        @self._sio.event
        async def connect(sid, environ):
            logger.info(f"[SocketIO] Client {sid} connected to default namespace")

        @self._sio.event
        async def disconnect(sid):
            logger.info(f"[SocketIO] Client {sid} disconnected from default namespace")

        # Spectrum namespace
        @self._sio.on("connect", namespace="/spectrum")
        async def spectrum_connect(sid, environ):
            self._clients["/spectrum"].add(sid)
            logger.info(
                f"[SocketIO] Client {sid} connected to /spectrum ({len(self._clients['/spectrum'])} total)"
            )

        @self._sio.on("disconnect", namespace="/spectrum")
        async def spectrum_disconnect(sid):
            self._clients["/spectrum"].discard(sid)
            logger.info(f"[SocketIO] Client {sid} disconnected from /spectrum")

        # Detections namespace
        @self._sio.on("connect", namespace="/detections")
        async def detections_connect(sid, environ):
            self._clients["/detections"].add(sid)
            logger.info(
                f"[SocketIO] Client {sid} connected to /detections ({len(self._clients['/detections'])} total)"
            )

        @self._sio.on("disconnect", namespace="/detections")
        async def detections_disconnect(sid):
            self._clients["/detections"].discard(sid)
            logger.info(f"[SocketIO] Client {sid} disconnected from /detections")

        # Analytics namespace
        @self._sio.on("connect", namespace="/analytics")
        async def analytics_connect(sid, environ):
            self._clients["/analytics"].add(sid)
            logger.info(f"[SocketIO] Client {sid} connected to /analytics")

        @self._sio.on("disconnect", namespace="/analytics")
        async def analytics_disconnect(sid):
            self._clients["/analytics"].discard(sid)
            logger.info(f"[SocketIO] Client {sid} disconnected from /analytics")

        # Hardware namespace
        @self._sio.on("connect", namespace="/hardware")
        async def hardware_connect(sid, environ):
            self._clients["/hardware"].add(sid)
            logger.info(f"[SocketIO] Client {sid} connected to /hardware")

        @self._sio.on("disconnect", namespace="/hardware")
        async def hardware_disconnect(sid):
            self._clients["/hardware"].discard(sid)
            logger.info(f"[SocketIO] Client {sid} disconnected from /hardware")

    # =========================================================================
    # Spectrum Events
    # =========================================================================

    async def emit_spectrum_measurement(
        self,
        psd_db: np.ndarray,
        center_freq_hz: float,
        sample_rate_hz: float,
        fft_size: int = None,
        device_id: str = "sdr-0",
    ) -> int:
        """
        Emit spectrum:measurement event to /spectrum namespace.

        Frontend expects:
        {
            frequency: number,
            power: number,
            timestamp: number,
            deviceId: string
        }

        For efficiency, we send the full spectrum array:
        {
            type: "spectrum",
            timestamp: number,
            centerFreqHz: number,
            sampleRateHz: number,
            fftSize: number,
            magnitudeDb: number[],
            frequencyBins: number[],
            deviceId: string
        }
        """
        # Rate limiting
        now = time.time()
        if now - self._last_spectrum_broadcast < self._min_broadcast_interval:
            return 0
        self._last_spectrum_broadcast = now

        if fft_size is None:
            fft_size = len(psd_db)

        # Format message using frontend adapter (camelCase)
        message = FrontendMessageFormatter.format_spectrum(
            psd_db=psd_db,
            center_freq_hz=center_freq_hz,
            sample_rate_hz=sample_rate_hz,
            fft_size=fft_size,
            timestamp=now,
        )
        message["deviceId"] = device_id

        # Emit to /spectrum namespace
        await self._sio.emit("spectrum:measurement", message, namespace="/spectrum")

        return len(self._clients["/spectrum"])

    # =========================================================================
    # Detection Events
    # =========================================================================

    async def emit_detection_new(self, detection: dict) -> int:
        """
        Emit detection:new event to /detections namespace.

        Frontend expects:
        {
            id: string,
            type: string,
            severity: string,
            confidence: number,
            timestamp: number,
            centerFreqHz: number,
            bandwidthHz: number,
            ...
        }
        """
        self._detection_counter += 1

        message = FrontendMessageFormatter.format_detection(
            detection_id=detection.get("detection_id", self._detection_counter),
            center_freq_hz=detection.get("center_freq_hz", 0),
            bandwidth_hz=detection.get("bandwidth_hz", 0),
            peak_power_db=detection.get("peak_power_db", -100),
            snr_db=detection.get("snr_db", 0),
            modulation_type=detection.get("modulation_type", detection.get("label", "Unknown")),
            confidence=detection.get("confidence", 0),
            duration=detection.get("duration", 0),
            timestamp=time.time(),
        )

        # Add severity based on SNR and power
        snr = detection.get("snr_db", 0)
        if snr > 25:
            message["severity"] = "high"
        elif snr > 15:
            message["severity"] = "medium"
        else:
            message["severity"] = "low"

        await self._sio.emit("detection:new", message, namespace="/detections")

        return len(self._clients["/detections"])

    async def emit_detection_update(self, detection_id: str, updates: dict) -> int:
        """
        Emit detection:update event to /detections namespace.

        Args:
            detection_id: ID of detection to update
            updates: Fields to update (status, resolution, etc.)
        """
        message = {
            "id": detection_id,
            "status": updates.get("status", "active"),
            "resolution": updates.get("resolution"),
            "timestamp": time.time(),
        }

        await self._sio.emit("detection:update", message, namespace="/detections")

        return len(self._clients["/detections"])

    async def emit_cluster_update(self, clusters: list[dict]) -> int:
        """
        Emit cluster update as detection events.

        Clusters are sent to the /detections namespace as they represent
        grouped detection activity.
        """
        message = FrontendMessageFormatter.format_clusters(clusters=clusters, timestamp=time.time())

        await self._sio.emit("cluster:update", message, namespace="/detections")

        return len(self._clients["/detections"])

    # =========================================================================
    # Analytics Events
    # =========================================================================

    async def emit_analytics_metrics(self, metrics: dict) -> int:
        """
        Emit analytics:metrics event to /analytics namespace.

        Frontend expects:
        {
            cpu: number,
            memory: number,
            detectionRate: number,
            latency: number
        }
        """
        message = {
            "cpu": metrics.get("cpu_percent", 0),
            "memory": metrics.get("memory_percent", 0),
            "detectionRate": metrics.get("detection_rate", 0),
            "latency": metrics.get("latency_ms", 0),
            "samplesProcessed": metrics.get("samples_processed", 0),
            "throughputMsps": metrics.get("throughput_msps", 0),
            "gpuUtilization": metrics.get("gpu_utilization", 0),
            "timestamp": time.time(),
        }

        await self._sio.emit("analytics:metrics", message, namespace="/analytics")

        return len(self._clients["/analytics"])

    # =========================================================================
    # Hardware Events
    # =========================================================================

    async def emit_hardware_status(self, status: dict) -> int:
        """
        Emit hardware:status event to /hardware namespace.

        Frontend expects:
        {
            deviceId: string,
            temperature: number,
            sampleRate: number,
            bufferUtilization: number
        }
        """
        message = {
            "deviceId": status.get("device_id", "sdr-0"),
            "connected": status.get("connected", False),
            "temperature": status.get("temperature_c", 0),
            "sampleRate": status.get("sample_rate_hz", 0),
            "centerFrequency": status.get("center_freq_hz", 0),
            "bufferUtilization": status.get("buffer_fill", 0),
            "streaming": status.get("streaming", False),
            "rxPath": status.get("rx_path", "LNAL"),
            "totalGainDb": status.get("total_gain_db", 0),
            "timestamp": time.time(),
        }

        await self._sio.emit("hardware:status", message, namespace="/hardware")

        return len(self._clients["/hardware"])

    # =========================================================================
    # Server Management
    # =========================================================================

    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            "spectrum_clients": len(self._clients["/spectrum"]),
            "detection_clients": len(self._clients["/detections"]),
            "analytics_clients": len(self._clients["/analytics"]),
            "hardware_clients": len(self._clients["/hardware"]),
            "total_clients": sum(len(c) for c in self._clients.values()),
            "spectrum_rate_hz": self._spectrum_rate_hz,
            "detections_sent": self._detection_counter,
        }

    @property
    def sio(self):
        """Get Socket.IO server instance."""
        return self._sio

    @property
    def app(self):
        """Get aiohttp web application."""
        return self._app

    async def start(self):
        """Start the Socket.IO server."""
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        logger.info(f"[SocketIO] Server started on {self._host}:{self._port}")
        logger.info("[SocketIO] Namespaces: /spectrum, /detections, /analytics, /hardware")


def create_socketio_server(
    host: str = "0.0.0.0", port: int = 8765, cors_origins: str = "*"
) -> RFForensicsSocketIO:
    """
    Create and configure Socket.IO server.

    Args:
        host: Bind address
        port: Port number
        cors_origins: CORS allowed origins

    Returns:
        RFForensicsSocketIO instance
    """
    return RFForensicsSocketIO(host=host, port=port, cors_allowed_origins=cors_origins)


# Standalone runner
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RF Forensics Socket.IO Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8765, help="Port number")
    parser.add_argument("--simulate", action="store_true", help="Run simulation mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    async def main():
        server = create_socketio_server(host=args.host, port=args.port)

        if args.simulate:
            # Simulation mode - generate fake data
            async def simulation_loop():
                await asyncio.sleep(2)
                logger.info("[SIM] Starting simulated data @ 20 Hz")

                center_freq = 915_000_000
                sample_rate = 2_048_000
                fft_size = 2048

                while True:
                    # Generate spectrum
                    psd_db = np.random.normal(-95, 3, fft_size).astype(np.float32)

                    # Add random signals
                    for _ in range(np.random.randint(0, 5)):
                        sig_bin = np.random.randint(100, fft_size - 100)
                        sig_power = np.random.uniform(-70, -40)
                        sig_width = np.random.randint(5, 30)
                        for i in range(-sig_width, sig_width + 1):
                            if 0 <= sig_bin + i < fft_size:
                                psd_db[sig_bin + i] = max(
                                    psd_db[sig_bin + i],
                                    sig_power * np.exp(-0.5 * (i / (sig_width / 2)) ** 2),
                                )

                    await server.emit_spectrum_measurement(
                        psd_db=psd_db,
                        center_freq_hz=center_freq,
                        sample_rate_hz=sample_rate,
                        fft_size=fft_size,
                    )

                    # Occasional detection
                    if np.random.random() < 0.1:
                        await server.emit_detection_new(
                            {
                                "detection_id": int(time.time() * 1000) % 100000,
                                "center_freq_hz": center_freq + np.random.randint(-500000, 500000),
                                "bandwidth_hz": np.random.choice([125000, 250000, 500000]),
                                "peak_power_db": np.random.uniform(-70, -40),
                                "snr_db": np.random.uniform(10, 25),
                                "modulation_type": np.random.choice(
                                    ["LoRa", "FSK", "GFSK", "WiFi", "Bluetooth"]
                                ),
                                "confidence": np.random.uniform(0.7, 0.95),
                                "duration": np.random.uniform(0.05, 0.5),
                            }
                        )

                    # Hardware status every 5 seconds
                    if int(time.time()) % 5 == 0:
                        await server.emit_hardware_status(
                            {
                                "device_id": "usdr-0",
                                "connected": True,
                                "temperature_c": 42.5 + np.random.uniform(-2, 2),
                                "sample_rate_hz": sample_rate,
                                "center_freq_hz": center_freq,
                                "buffer_fill": np.random.uniform(0.1, 0.3),
                                "streaming": True,
                            }
                        )

                    await asyncio.sleep(0.05)  # 20 Hz

            asyncio.create_task(simulation_loop())

        await server.start()

        # Keep running
        while True:
            await asyncio.sleep(1)

    asyncio.run(main())
