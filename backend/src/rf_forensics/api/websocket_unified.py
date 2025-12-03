"""
Unified WebSocket Server for RF Forensics

Single WebSocket server with path-based routing for all streams:
- /ws/spectrum - Binary spectrum frames (20-byte header + uint8)
- /ws/ble - JSON BLE packet events
- /ws/lora - JSON LoRa frame events
- /ws/iq - Binary IQ data (optional)

Per BACKEND_CONTRACT.md and frontend LiveMonitoring expectations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.server import serve as ws_serve

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

logger = logging.getLogger(__name__)


# DBm to uint8 conversion constants
DBM_MIN = -120.0
DBM_MAX = -20.0
DBM_RANGE = DBM_MAX - DBM_MIN


def dbm_to_uint8(dbm_values: np.ndarray) -> np.ndarray:
    """Convert dBm values to uint8 (0=-120dBm, 255=-20dBm)."""
    normalized = (dbm_values - DBM_MIN) / DBM_RANGE * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def pack_spectrum_frame(
    timestamp: float, center_freq_hz: float, span_hz: float, magnitude_dbm: np.ndarray
) -> bytes:
    """Pack spectrum frame (20-byte header + uint8 payload)."""
    header = struct.pack(
        "<d f f H H",
        timestamp,
        float(center_freq_hz),
        float(span_hz),
        len(magnitude_dbm),
        0,  # Reserved
    )
    payload = dbm_to_uint8(magnitude_dbm).tobytes()
    return header + payload


@dataclass
class LoRaFrame:
    """LoRa frame data structure per frontend expectations."""

    frame_id: int
    timestamp: float
    center_freq_hz: float
    spreading_factor: int  # 7-12
    bandwidth_hz: float  # 125000, 250000, 500000
    coding_rate: str  # "4/5", "4/6", "4/7", "4/8"
    payload_hex: str
    crc_valid: bool
    snr_db: float
    rssi_dbm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "lora",
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "center_freq_hz": self.center_freq_hz,
            "spreading_factor": self.spreading_factor,
            "bandwidth_hz": self.bandwidth_hz,
            "coding_rate": self.coding_rate,
            "payload_hex": self.payload_hex,
            "crc_valid": self.crc_valid,
            "snr_db": self.snr_db,
            "rssi_dbm": self.rssi_dbm,
        }


@dataclass
class BLEPacketEvent:
    """BLE packet event per frontend expectations."""

    packet_id: int
    timestamp: float
    channel: int  # 37, 38, 39
    access_address: str
    pdu_type: str
    payload_hex: str
    crc_valid: bool
    rssi_dbm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ble",
            "timestamp": self.timestamp,
            "packet_id": self.packet_id,
            "channel": self.channel,
            "access_address": self.access_address,
            "pdu_type": self.pdu_type,
            "payload_hex": self.payload_hex,
            "crc_valid": self.crc_valid,
            "rssi_dbm": self.rssi_dbm,
        }


class UnifiedWebSocketServer:
    """
    Unified WebSocket server with path-based routing.

    All streams on single port (default 8765) with different paths:
    - ws://host:8765/ws/spectrum
    - ws://host:8765/ws/ble
    - ws://host:8765/ws/lora
    - ws://host:8765/ws/iq
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required")

        self.host = host
        self.port = port

        # Clients per path
        self._spectrum_clients: set[WebSocketServerProtocol] = set()
        self._ble_clients: set[WebSocketServerProtocol] = set()
        self._lora_clients: set[WebSocketServerProtocol] = set()
        self._iq_clients: set[WebSocketServerProtocol] = set()

        self._lock = asyncio.Lock()
        self._server = None
        self._running = False

        # Counters
        self._spectrum_frames = 0
        self._ble_packets = 0
        self._lora_frames = 0
        self._iq_frames = 0

    async def start(self) -> None:
        """Start the unified WebSocket server."""
        if self._running:
            logger.warning("WebSocket server already running")
            return

        self._server = await ws_serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )

        self._running = True
        logger.info(
            f"Unified WebSocket server started on ws://{self.host}:{self.port}\n"
            f"  Endpoints: /ws/spectrum, /ws/ble, /ws/lora, /ws/iq"
        )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return

        self._running = False

        # Close all clients
        async with self._lock:
            all_clients = (
                self._spectrum_clients | self._ble_clients | self._lora_clients | self._iq_clients
            )
            if all_clients:
                await asyncio.gather(*[c.close() for c in all_clients], return_exceptions=True)
            self._spectrum_clients.clear()
            self._ble_clients.clear()
            self._lora_clients.clear()
            self._iq_clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("Unified WebSocket server stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        """Route connection based on path."""
        path = websocket.path
        client_addr = websocket.remote_address

        logger.info(f"WebSocket connection: {client_addr} -> {path}")

        # Route to appropriate handler based on path
        if path in ("/ws/spectrum", "/spectrum"):
            await self._handle_spectrum_client(websocket)
        elif path in ("/ws/ble", "/ble"):
            await self._handle_ble_client(websocket)
        elif path in ("/ws/lora", "/lora"):
            await self._handle_lora_client(websocket)
        elif path in ("/ws/iq", "/iq"):
            await self._handle_iq_client(websocket)
        else:
            logger.warning(f"Unknown WebSocket path: {path}")
            await websocket.close(1008, f"Unknown path: {path}")

    async def _handle_spectrum_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle spectrum stream client."""
        async with self._lock:
            self._spectrum_clients.add(websocket)

        try:
            async for message in websocket:
                # Spectrum is send-only, but handle any client messages
                logger.debug(
                    f"Spectrum client message: {message[:50] if isinstance(message, str) else 'binary'}"
                )
        finally:
            async with self._lock:
                self._spectrum_clients.discard(websocket)
            logger.info(f"Spectrum client disconnected: {websocket.remote_address}")

    async def _handle_ble_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle BLE stream client."""
        async with self._lock:
            self._ble_clients.add(websocket)

        try:
            async for message in websocket:
                logger.debug(
                    f"BLE client message: {message[:50] if isinstance(message, str) else 'binary'}"
                )
        finally:
            async with self._lock:
                self._ble_clients.discard(websocket)
            logger.info(f"BLE client disconnected: {websocket.remote_address}")

    async def _handle_lora_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle LoRa stream client."""
        async with self._lock:
            self._lora_clients.add(websocket)

        try:
            async for message in websocket:
                logger.debug(
                    f"LoRa client message: {message[:50] if isinstance(message, str) else 'binary'}"
                )
        finally:
            async with self._lock:
                self._lora_clients.discard(websocket)
            logger.info(f"LoRa client disconnected: {websocket.remote_address}")

    async def _handle_iq_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle IQ stream client."""
        async with self._lock:
            self._iq_clients.add(websocket)

        try:
            async for message in websocket:
                logger.debug(
                    f"IQ client message: {message[:50] if isinstance(message, str) else 'binary'}"
                )
        finally:
            async with self._lock:
                self._iq_clients.discard(websocket)
            logger.info(f"IQ client disconnected: {websocket.remote_address}")

    # =========================================================================
    # Broadcast Methods
    # =========================================================================

    async def broadcast_spectrum(
        self, timestamp: float, center_freq_hz: float, span_hz: float, magnitude_dbm: np.ndarray
    ) -> int:
        """Broadcast binary spectrum frame to /ws/spectrum clients."""
        if not self._spectrum_clients:
            return 0

        frame = pack_spectrum_frame(timestamp, center_freq_hz, span_hz, magnitude_dbm)

        async with self._lock:
            clients = list(self._spectrum_clients)

        results = await asyncio.gather(
            *[self._send_binary(c, frame) for c in clients], return_exceptions=True
        )

        self._spectrum_frames += 1
        return sum(1 for r in results if r is True)

    async def broadcast_ble(self, packet: dict[str, Any]) -> int:
        """
        Broadcast BLE packet to /ws/ble clients.

        Args:
            packet: BLE packet dict with fields per frontend expectations:
                - type: "ble"
                - timestamp, packet_id, channel, access_address
                - pdu_type, payload_hex, crc_valid, rssi_dbm
        """
        if not self._ble_clients:
            return 0

        # Ensure required fields
        if "type" not in packet:
            packet["type"] = "ble"

        message = json.dumps(packet)

        async with self._lock:
            clients = list(self._ble_clients)

        results = await asyncio.gather(
            *[self._send_text(c, message) for c in clients], return_exceptions=True
        )

        self._ble_packets += 1
        return sum(1 for r in results if r is True)

    async def broadcast_lora(self, frame: dict[str, Any]) -> int:
        """
        Broadcast LoRa frame to /ws/lora clients.

        Args:
            frame: LoRa frame dict with fields per frontend expectations:
                - type: "lora"
                - timestamp, frame_id, center_freq_hz
                - spreading_factor, bandwidth_hz, coding_rate
                - payload_hex, crc_valid, snr_db, rssi_dbm
        """
        if not self._lora_clients:
            return 0

        # Ensure required fields
        if "type" not in frame:
            frame["type"] = "lora"

        message = json.dumps(frame)

        async with self._lock:
            clients = list(self._lora_clients)

        results = await asyncio.gather(
            *[self._send_text(c, message) for c in clients], return_exceptions=True
        )

        self._lora_frames += 1
        return sum(1 for r in results if r is True)

    async def broadcast_iq(
        self, timestamp: float, center_freq_hz: float, sample_rate_hz: float, iq_data: np.ndarray
    ) -> int:
        """Broadcast binary IQ frame to /ws/iq clients."""
        if not self._iq_clients:
            return 0

        # Pack header (20 bytes per frontend contract)
        header = struct.pack(
            "<d f f H H",
            timestamp,
            float(center_freq_hz),
            float(sample_rate_hz),
            min(len(iq_data), 65535),  # u16 max
            0,  # Reserved
        )

        # Convert complex to interleaved float32 (per frontend contract)
        interleaved = np.empty(len(iq_data) * 2, dtype=np.float32)
        interleaved[0::2] = iq_data.real.astype(np.float32)
        interleaved[1::2] = iq_data.imag.astype(np.float32)

        frame = header + interleaved.tobytes()

        async with self._lock:
            clients = list(self._iq_clients)

        results = await asyncio.gather(
            *[self._send_binary(c, frame) for c in clients], return_exceptions=True
        )

        self._iq_frames += 1
        return sum(1 for r in results if r is True)

    async def _send_binary(self, client: WebSocketServerProtocol, data: bytes) -> bool:
        """Send binary data to client."""
        try:
            await client.send(data)
            return True
        except Exception:
            return False

    async def _send_text(self, client: WebSocketServerProtocol, text: str) -> bool:
        """Send text data to client."""
        try:
            await client.send(text)
            return True
        except Exception:
            return False

    # =========================================================================
    # Properties and Stats
    # =========================================================================

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def spectrum_client_count(self) -> int:
        return len(self._spectrum_clients)

    @property
    def ble_client_count(self) -> int:
        return len(self._ble_clients)

    @property
    def lora_client_count(self) -> int:
        return len(self._lora_clients)

    @property
    def iq_client_count(self) -> int:
        return len(self._iq_clients)

    @property
    def total_client_count(self) -> int:
        return (
            len(self._spectrum_clients)
            + len(self._ble_clients)
            + len(self._lora_clients)
            + len(self._iq_clients)
        )

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics."""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "clients": {
                "spectrum": self.spectrum_client_count,
                "ble": self.ble_client_count,
                "lora": self.lora_client_count,
                "iq": self.iq_client_count,
                "total": self.total_client_count,
            },
            "frames_sent": {
                "spectrum": self._spectrum_frames,
                "ble": self._ble_packets,
                "lora": self._lora_frames,
                "iq": self._iq_frames,
            },
        }


# Singleton instance
_unified_server: UnifiedWebSocketServer | None = None


async def get_unified_server(host: str = "0.0.0.0", port: int = 8765) -> UnifiedWebSocketServer:
    """Get or create the unified WebSocket server singleton."""
    global _unified_server

    if _unified_server is None:
        _unified_server = UnifiedWebSocketServer(host, port)

    if not _unified_server.is_running:
        await _unified_server.start()

    return _unified_server


async def shutdown_unified_server() -> None:
    """Shutdown the unified WebSocket server."""
    global _unified_server

    if _unified_server:
        await _unified_server.stop()
        _unified_server = None
