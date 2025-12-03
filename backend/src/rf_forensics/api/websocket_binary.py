"""
Binary WebSocket Server for Spectrum Data

Native WebSocket server providing binary spectrum frames per BACKEND_CONTRACT.md Section 2.1.

Frame format:
- 20-byte header (little-endian):
  - float64 timestamp (8 bytes)
  - float32 center_freq (4 bytes)
  - float32 span (4 bytes)
  - uint16 num_bins (2 bytes)
  - uint16 reserved (2 bytes)
- uint8 payload: magnitude values where 0=-120dBm, 255=-20dBm
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time

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
DBM_MIN = -120.0  # Maps to 0
DBM_MAX = -20.0  # Maps to 255
DBM_RANGE = DBM_MAX - DBM_MIN  # 100 dB range


def dbm_to_uint8(dbm_values: np.ndarray) -> np.ndarray:
    """
    Convert dBm values to uint8 per BACKEND_CONTRACT.md.

    Mapping: 0 = -120 dBm, 255 = -20 dBm (linear interpolation)
    Formula: value = (dBm + 120) * 255 / 100

    Args:
        dbm_values: Array of power values in dBm.

    Returns:
        Array of uint8 values.
    """
    # Linear mapping: (dBm - DBM_MIN) / DBM_RANGE * 255
    normalized = (dbm_values - DBM_MIN) / DBM_RANGE * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def uint8_to_dbm(uint8_values: np.ndarray) -> np.ndarray:
    """
    Convert uint8 values back to dBm.

    Args:
        uint8_values: Array of uint8 values.

    Returns:
        Array of power values in dBm.
    """
    return uint8_values.astype(np.float32) / 255.0 * DBM_RANGE + DBM_MIN


def pack_spectrum_header(
    timestamp: float, center_freq_hz: float, span_hz: float, num_bins: int
) -> bytes:
    """
    Pack 20-byte spectrum frame header.

    Format (little-endian):
    - float64 timestamp (8 bytes) - Unix timestamp
    - float32 center_freq (4 bytes) - Center frequency in Hz
    - float32 span (4 bytes) - Frequency span in Hz
    - uint16 num_bins (2 bytes) - Number of FFT bins
    - uint16 reserved (2 bytes) - Reserved for future use

    Args:
        timestamp: Unix timestamp.
        center_freq_hz: Center frequency in Hz.
        span_hz: Frequency span in Hz.
        num_bins: Number of FFT bins.

    Returns:
        20-byte header as bytes.
    """
    return struct.pack(
        "<d f f H H",
        timestamp,
        float(center_freq_hz),
        float(span_hz),
        int(num_bins),
        0,  # Reserved
    )


def pack_spectrum_frame(
    timestamp: float, center_freq_hz: float, span_hz: float, magnitude_dbm: np.ndarray
) -> bytes:
    """
    Pack complete spectrum frame (header + payload).

    Args:
        timestamp: Unix timestamp.
        center_freq_hz: Center frequency in Hz.
        span_hz: Frequency span in Hz.
        magnitude_dbm: Power spectrum in dBm.

    Returns:
        Complete frame as bytes.
    """
    header = pack_spectrum_header(timestamp, center_freq_hz, span_hz, len(magnitude_dbm))
    payload = dbm_to_uint8(magnitude_dbm).tobytes()
    return header + payload


class BinarySpectrumServer:
    """
    Native WebSocket server for binary spectrum data.

    Provides high-performance binary streaming of spectrum data
    to frontend clients per BACKEND_CONTRACT.md.

    Features:
    - Zero-copy binary frames
    - Multiple client support with broadcast
    - Automatic client cleanup on disconnect
    - Graceful shutdown
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """
        Initialize binary spectrum server.

        Args:
            host: Host address to bind.
            port: Port to listen on.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required for BinarySpectrumServer")

        self.host = host
        self.port = port

        # Connected clients
        self._clients: set[WebSocketServerProtocol] = set()
        self._clients_lock = asyncio.Lock()

        # Server state
        self._server = None
        self._running = False

        # Statistics
        self._frames_sent = 0
        self._bytes_sent = 0
        self._last_broadcast_time = 0.0

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            logger.warning("Binary spectrum server already running")
            return

        self._server = await ws_serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )

        self._running = True
        logger.info(
            f"Binary spectrum WebSocket server started on ws://{self.host}:{self.port}/ws/spectrum"
        )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return

        self._running = False

        # Close all client connections
        async with self._clients_lock:
            if self._clients:
                await asyncio.gather(
                    *[client.close() for client in self._clients], return_exceptions=True
                )
            self._clients.clear()

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("Binary spectrum WebSocket server stopped")

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a client connection."""
        client_addr = websocket.remote_address
        logger.info(f"Binary spectrum client connected: {client_addr}")

        async with self._clients_lock:
            self._clients.add(websocket)

        try:
            # Keep connection alive and handle any messages
            async for message in websocket:
                # Clients typically don't send messages, but handle if they do
                if isinstance(message, str):
                    logger.debug(f"Received message from {client_addr}: {message[:100]}")
                # Binary messages ignored

        except Exception as e:
            logger.debug(f"Client {client_addr} disconnected: {e}")

        finally:
            async with self._clients_lock:
                self._clients.discard(websocket)
            logger.info(f"Binary spectrum client disconnected: {client_addr}")

    async def broadcast_spectrum(
        self, timestamp: float, center_freq_hz: float, span_hz: float, magnitude_dbm: np.ndarray
    ) -> int:
        """
        Broadcast spectrum frame to all connected clients.

        Args:
            timestamp: Unix timestamp.
            center_freq_hz: Center frequency in Hz.
            span_hz: Frequency span in Hz.
            magnitude_dbm: Power spectrum in dBm.

        Returns:
            Number of clients that received the frame.
        """
        if not self._clients:
            return 0

        # Pack frame
        frame = pack_spectrum_frame(timestamp, center_freq_hz, span_hz, magnitude_dbm)

        # Broadcast to all clients
        async with self._clients_lock:
            clients = list(self._clients)

        if not clients:
            return 0

        # Send to all clients concurrently
        results = await asyncio.gather(
            *[self._send_to_client(client, frame) for client in clients], return_exceptions=True
        )

        # Count successful sends
        success_count = sum(1 for r in results if r is True)

        # Update stats
        self._frames_sent += success_count
        self._bytes_sent += len(frame) * success_count
        self._last_broadcast_time = time.time()

        return success_count

    async def _send_to_client(self, client: WebSocketServerProtocol, frame: bytes) -> bool:
        """Send frame to a single client."""
        try:
            await client.send(frame)
            return True
        except Exception:
            # Client disconnected or error
            return False

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Whether the server is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "client_count": self.client_count,
            "frames_sent": self._frames_sent,
            "bytes_sent": self._bytes_sent,
            "last_broadcast_time": self._last_broadcast_time,
        }


class BinaryIQServer:
    """
    Native WebSocket server for binary IQ data streaming.

    Similar to BinarySpectrumServer but for raw IQ samples.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        """
        Initialize binary IQ server.

        Args:
            host: Host address to bind.
            port: Port to listen on.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required for BinaryIQServer")

        self.host = host
        self.port = port
        self._clients: set[WebSocketServerProtocol] = set()
        self._clients_lock = asyncio.Lock()
        self._server = None
        self._running = False

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            return

        self._server = await ws_serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )

        self._running = True
        logger.info(f"Binary IQ WebSocket server started on ws://{self.host}:{self.port}/ws/iq")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return

        self._running = False

        async with self._clients_lock:
            if self._clients:
                await asyncio.gather(
                    *[client.close() for client in self._clients], return_exceptions=True
                )
            self._clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("Binary IQ WebSocket server stopped")

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a client connection."""
        async with self._clients_lock:
            self._clients.add(websocket)

        try:
            async for _ in websocket:
                pass  # Keep alive

        finally:
            async with self._clients_lock:
                self._clients.discard(websocket)

    async def broadcast_iq(
        self, timestamp: float, center_freq_hz: float, sample_rate_hz: float, iq_data: np.ndarray
    ) -> int:
        """
        Broadcast IQ frame to all connected clients.

        Frame format:
        - 24-byte header (little-endian):
          - float64 timestamp (8 bytes)
          - float32 center_freq (4 bytes)
          - float32 sample_rate (4 bytes)
          - uint32 num_samples (4 bytes)
          - uint32 reserved (4 bytes)
        - int16 interleaved I/Q samples

        Args:
            timestamp: Unix timestamp.
            center_freq_hz: Center frequency in Hz.
            sample_rate_hz: Sample rate in Hz.
            iq_data: Complex IQ samples.

        Returns:
            Number of clients that received the frame.
        """
        if not self._clients:
            return 0

        # Pack header
        header = struct.pack(
            "<d f f I I",
            timestamp,
            float(center_freq_hz),
            float(sample_rate_hz),
            len(iq_data),
            0,  # Reserved
        )

        # Convert complex to interleaved int16
        scale = 32767.0
        interleaved = np.empty(len(iq_data) * 2, dtype=np.int16)
        interleaved[0::2] = np.clip(iq_data.real * scale, -32768, 32767).astype(np.int16)
        interleaved[1::2] = np.clip(iq_data.imag * scale, -32768, 32767).astype(np.int16)

        frame = header + interleaved.tobytes()

        # Broadcast
        async with self._clients_lock:
            clients = list(self._clients)

        if not clients:
            return 0

        results = await asyncio.gather(
            *[self._send_to_client(client, frame) for client in clients], return_exceptions=True
        )

        return sum(1 for r in results if r is True)

    async def _send_to_client(self, client: WebSocketServerProtocol, frame: bytes) -> bool:
        """Send frame to a single client."""
        try:
            await client.send(frame)
            return True
        except Exception:
            return False

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)


class BLEWebSocketServer:
    """
    WebSocket server for BLE packet data.

    Per BACKEND_CONTRACT.md, /ws/ble endpoint sends JSON packets.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8767):
        """
        Initialize BLE WebSocket server.

        Args:
            host: Host address to bind.
            port: Port to listen on.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required for BLEWebSocketServer")

        self.host = host
        self.port = port
        self._clients: set[WebSocketServerProtocol] = set()
        self._clients_lock = asyncio.Lock()
        self._server = None
        self._running = False
        self._packet_counter = 0

    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            return

        self._server = await ws_serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10,
        )

        self._running = True
        logger.info(f"BLE WebSocket server started on ws://{self.host}:{self.port}/ws/ble")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if not self._running:
            return

        self._running = False

        async with self._clients_lock:
            if self._clients:
                await asyncio.gather(
                    *[client.close() for client in self._clients], return_exceptions=True
                )
            self._clients.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        logger.info("BLE WebSocket server stopped")

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a client connection."""
        client_addr = websocket.remote_address
        logger.info(f"BLE WebSocket client connected: {client_addr}")

        async with self._clients_lock:
            self._clients.add(websocket)

        try:
            async for message in websocket:
                # Handle subscription messages if needed
                logger.debug(
                    f"BLE client message: {message[:100] if isinstance(message, str) else 'binary'}"
                )

        finally:
            async with self._clients_lock:
                self._clients.discard(websocket)
            logger.info(f"BLE WebSocket client disconnected: {client_addr}")

    async def broadcast_packet(self, packet_dict: dict) -> int:
        """
        Broadcast BLE packet to all connected clients.

        Per BACKEND_CONTRACT.md, BLE packets are sent as JSON.

        Args:
            packet_dict: BLE packet as dictionary.

        Returns:
            Number of clients that received the packet.
        """
        if not self._clients:
            return 0

        import json

        self._packet_counter += 1

        # Add event wrapper per contract
        event = {
            "type": "ble",
            "timestamp": packet_dict.get("timestamp", time.time()),
            "packet_id": self._packet_counter,
            "data": packet_dict,
        }

        message = json.dumps(event)

        async with self._clients_lock:
            clients = list(self._clients)

        if not clients:
            return 0

        results = await asyncio.gather(
            *[self._send_to_client(client, message) for client in clients], return_exceptions=True
        )

        return sum(1 for r in results if r is True)

    async def _send_to_client(self, client: WebSocketServerProtocol, message: str) -> bool:
        """Send message to a single client."""
        try:
            await client.send(message)
            return True
        except Exception:
            return False

    @property
    def client_count(self) -> int:
        """Number of connected clients."""
        return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Whether the server is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "client_count": self.client_count,
            "packets_sent": self._packet_counter,
        }


# Singleton instances for easy access
_spectrum_server: BinarySpectrumServer | None = None
_iq_server: BinaryIQServer | None = None
_ble_server: BLEWebSocketServer | None = None


async def get_spectrum_server(host: str = "0.0.0.0", port: int = 8765) -> BinarySpectrumServer:
    """Get or create the binary spectrum server singleton."""
    global _spectrum_server

    if _spectrum_server is None:
        _spectrum_server = BinarySpectrumServer(host, port)

    if not _spectrum_server.is_running:
        await _spectrum_server.start()

    return _spectrum_server


async def get_iq_server(host: str = "0.0.0.0", port: int = 8766) -> BinaryIQServer:
    """Get or create the binary IQ server singleton."""
    global _iq_server

    if _iq_server is None:
        _iq_server = BinaryIQServer(host, port)

    if not _iq_server._running:
        await _iq_server.start()

    return _iq_server


async def get_ble_server(host: str = "0.0.0.0", port: int = 8767) -> BLEWebSocketServer:
    """Get or create the BLE WebSocket server singleton."""
    global _ble_server

    if _ble_server is None:
        _ble_server = BLEWebSocketServer(host, port)

    if not _ble_server.is_running:
        await _ble_server.start()

    return _ble_server


async def shutdown_servers() -> None:
    """Shutdown all binary WebSocket servers."""
    global _spectrum_server, _iq_server, _ble_server

    if _spectrum_server:
        await _spectrum_server.stop()
        _spectrum_server = None

    if _iq_server:
        await _iq_server.stop()
        _iq_server = None

    if _ble_server:
        await _ble_server.stop()
        _ble_server = None
