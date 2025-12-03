"""
GPU RF Forensics Engine - WebSocket Server

Binary WebSocket streaming for real-time spectral data visualization.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from rf_forensics.api.frontend_adapter import FrontendMessageFormatter


@dataclass
class SpectrumMetadata:
    """Metadata for spectrum data."""

    timestamp: float
    center_freq_hz: float
    span_hz: float
    num_bins: int


@dataclass
class IQMetadata:
    """Metadata for IQ samples."""

    timestamp: float
    center_freq_hz: float
    sample_rate_hz: float
    num_samples: int


@dataclass
class ClientConnection:
    """Tracks a WebSocket client connection."""

    websocket: Any  # WebSocket
    client_id: str
    connected_at: float
    last_message: float
    subscriptions: set[str]
    rate_limit_hz: float = 30.0


class SpectrumWebSocketServer:
    """
    Binary WebSocket server for real-time spectral streaming.

    Protocol:
    - /ws/spectrum: Binary PSD data (high rate)
    - /ws/detections: JSON detection events
    - /ws/clusters: JSON cluster updates

    Binary format for spectrum (v2 - full precision):
    - Header (28 bytes): timestamp(f64), center_freq(f64), span(f64), num_bins(u32)
    - Payload: uint8 array (quantized dB values)

    Note: Uses float64 for frequencies to maintain precision for GHz range (e.g., 2.4 GHz WiFi).
    Previous format used float32 which loses precision above ~16.7 MHz.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        max_clients: int = 10,
        spectrum_rate_hz: float = 30.0,
    ):
        """
        Initialize WebSocket server.

        Args:
            host: Bind address.
            port: Port number.
            max_clients: Maximum concurrent clients.
            spectrum_rate_hz: Target spectrum update rate.
        """
        self._host = host
        self._port = port
        self._max_clients = max_clients
        self._spectrum_rate_hz = spectrum_rate_hz

        self._clients: dict[str, ClientConnection] = {}
        self._client_counter = 0

        # Rate limiting
        self._last_spectrum_broadcast = 0.0
        self._min_broadcast_interval = 1.0 / spectrum_rate_hz

        # Pre-allocated buffers for spectrum quantization (reduces per-frame allocations)
        self._spectrum_buffer_size = 0
        self._psd_clipped: np.ndarray | None = None
        self._psd_norm: np.ndarray | None = None
        self._psd_uint8: np.ndarray | None = None

    def _ensure_spectrum_buffers(self, n_bins: int) -> None:
        """Ensure pre-allocated buffers are sized correctly."""
        if self._spectrum_buffer_size != n_bins:
            self._psd_clipped = np.empty(n_bins, dtype=np.float32)
            self._psd_norm = np.empty(n_bins, dtype=np.float32)
            self._psd_uint8 = np.empty(n_bins, dtype=np.uint8)
            self._spectrum_buffer_size = n_bins

    def create_spectrum_message_buffered(
        self,
        psd_db: np.ndarray,
        metadata: SpectrumMetadata,
        min_db: float = -120,
        max_db: float = -20,
    ) -> bytes:
        """
        Create binary spectrum message using pre-allocated buffers.

        Reduces per-frame memory allocations by reusing internal buffers.
        For use by broadcast_spectrum() in hot path.

        Args:
            psd_db: PSD array in dB.
            metadata: Spectrum metadata.
            min_db: Minimum dB value for quantization.
            max_db: Maximum dB value for quantization.

        Returns:
            Binary message bytes.
        """
        n_bins = len(psd_db)
        self._ensure_spectrum_buffers(n_bins)

        # In-place operations using pre-allocated buffers
        np.clip(psd_db, min_db, max_db, out=self._psd_clipped)
        np.subtract(self._psd_clipped, min_db, out=self._psd_norm)
        np.divide(self._psd_norm, (max_db - min_db), out=self._psd_norm)
        np.multiply(self._psd_norm, 255, out=self._psd_norm)
        self._psd_uint8[:] = self._psd_norm.astype(np.uint8)

        # Pack header (28 bytes - full precision for GHz frequencies)
        header = struct.pack(
            "<dddI",
            metadata.timestamp,
            metadata.center_freq_hz,
            metadata.span_hz,
            metadata.num_bins,
        )

        return header + self._psd_uint8.tobytes()

    @staticmethod
    def create_spectrum_message(
        psd_db: np.ndarray, metadata: SpectrumMetadata, min_db: float = -120, max_db: float = -20
    ) -> bytes:
        """
        Create binary spectrum message (static version for compatibility).

        For hot paths, prefer create_spectrum_message_buffered() instance method.

        Args:
            psd_db: PSD array in dB.
            metadata: Spectrum metadata.
            min_db: Minimum dB value for quantization.
            max_db: Maximum dB value for quantization.

        Returns:
            Binary message bytes.
        """
        # Quantize to uint8
        psd_clipped = np.clip(psd_db, min_db, max_db)
        psd_norm = (psd_clipped - min_db) / (max_db - min_db)
        psd_uint8 = (psd_norm * 255).astype(np.uint8)

        # Pack header (28 bytes - full precision for GHz frequencies)
        # Format: timestamp(d), center_freq(d), span(d), num_bins(I)
        header = struct.pack(
            "<dddI",
            metadata.timestamp,
            metadata.center_freq_hz,
            metadata.span_hz,
            metadata.num_bins,
        )

        return header + psd_uint8.tobytes()

    @staticmethod
    def parse_spectrum_message(data: bytes) -> tuple[SpectrumMetadata, np.ndarray]:
        """Parse binary spectrum message (v2 format with full precision)."""
        # Unpack header (28 bytes)
        header_size = struct.calcsize("<dddI")
        timestamp, center_freq, span, num_bins = struct.unpack("<dddI", data[:header_size])

        metadata = SpectrumMetadata(
            timestamp=timestamp, center_freq_hz=center_freq, span_hz=span, num_bins=num_bins
        )

        # Extract payload
        psd_uint8 = np.frombuffer(data[header_size:], dtype=np.uint8)

        return metadata, psd_uint8

    @staticmethod
    def create_iq_message(
        i_samples: np.ndarray, q_samples: np.ndarray, metadata: IQMetadata
    ) -> bytes:
        """
        Create binary IQ message.

        Args:
            i_samples: In-phase samples (float32).
            q_samples: Quadrature samples (float32).
            metadata: IQ metadata.

        Returns:
            Binary message bytes.

        Binary format:
        - Header (20 bytes): timestamp(f64), center_freq(f32), sample_rate(f32), num_samples(u32)
        - Payload: interleaved float32 IQ pairs [I0, Q0, I1, Q1, ...]

        Note: Uses <dffI format (uint32 for num_samples) per frontend parseIQMessage().
        """
        # Ensure float32
        i_f32 = i_samples.astype(np.float32)
        q_f32 = q_samples.astype(np.float32)

        # Pack header (20 bytes per frontend contract)
        # Format: <dffI = float64 + float32 + float32 + uint32 = 20 bytes
        header = struct.pack(
            "<dffI",
            metadata.timestamp,
            metadata.center_freq_hz,
            metadata.sample_rate_hz,
            metadata.num_samples,  # uint32, no cap needed
        )

        # Interleave IQ samples
        interleaved = np.empty(len(i_f32) * 2, dtype=np.float32)
        interleaved[0::2] = i_f32
        interleaved[1::2] = q_f32

        return header + interleaved.tobytes()

    @staticmethod
    def parse_iq_message(data: bytes) -> tuple[IQMetadata, np.ndarray, np.ndarray]:
        """Parse binary IQ message."""
        # Unpack header (20 bytes per frontend contract)
        # Format: <dffI = float64 + float32 + float32 + uint32 = 20 bytes
        header_size = struct.calcsize("<dffI")
        timestamp, center_freq, sample_rate, num_samples = struct.unpack(
            "<dffI", data[:header_size]
        )

        metadata = IQMetadata(
            timestamp=timestamp,
            center_freq_hz=center_freq,
            sample_rate_hz=sample_rate,
            num_samples=num_samples,
        )

        # Extract interleaved IQ samples
        interleaved = np.frombuffer(data[header_size:], dtype=np.float32)
        i_samples = interleaved[0::2]
        q_samples = interleaved[1::2]

        return metadata, i_samples, q_samples

    async def handle_spectrum_client(self, websocket, client_id: str):
        """Handle spectrum WebSocket connection using async iteration."""
        try:
            # Use async for to properly wait for messages without busy polling
            async for message in websocket.iter_text():
                # Handle control messages (e.g., pause, config)
                await self._handle_control_message(client_id, message)
        except WebSocketDisconnect:
            logger.debug(f"Client {client_id} disconnected from spectrum stream")
        except ConnectionResetError:
            logger.debug(f"Connection reset for spectrum client {client_id}")
        except Exception as e:
            logger.warning(f"Unexpected error handling spectrum client {client_id}: {e}")

    async def broadcast_spectrum(self, psd_db: np.ndarray, metadata: SpectrumMetadata) -> int:
        """
        Broadcast spectrum to all connected clients.

        Args:
            psd_db: PSD array in dB.
            metadata: Spectrum metadata.

        Returns:
            Number of clients sent to.
        """
        # Rate limiting
        now = time.time()
        if now - self._last_spectrum_broadcast < self._min_broadcast_interval:
            return 0

        self._last_spectrum_broadcast = now

        # Create binary message
        # Use buffered version to reduce per-frame allocations
        message = self.create_spectrum_message_buffered(psd_db, metadata)

        # Broadcast to binary spectrum subscribers only
        # NOTE: "spectrum_binary" is for high-performance binary mode
        #       "spectrum" is for JSON mode (frontend default)
        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "spectrum_binary" in client.subscriptions:
                try:
                    await client.websocket.send_bytes(message)
                    client.last_message = now
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending spectrum to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        # Remove disconnected clients
        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def broadcast_iq_samples(
        self,
        i_samples: np.ndarray,
        q_samples: np.ndarray,
        center_freq_hz: float,
        sample_rate_hz: float,
    ) -> int:
        """
        Broadcast IQ samples to all subscribed clients.

        Args:
            i_samples: In-phase samples (float32).
            q_samples: Quadrature samples (float32).
            center_freq_hz: Center frequency in Hz.
            sample_rate_hz: Sample rate in Hz.

        Returns:
            Number of clients sent to.
        """
        now = time.time()

        # Create metadata
        metadata = IQMetadata(
            timestamp=now,
            center_freq_hz=center_freq_hz,
            sample_rate_hz=sample_rate_hz,
            num_samples=len(i_samples),
        )

        # Create binary message
        message = self.create_iq_message(i_samples, q_samples, metadata)

        # Broadcast to IQ subscribers
        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "iq" in client.subscriptions:
                try:
                    await client.websocket.send_bytes(message)
                    client.last_message = now
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending IQ to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        # Remove disconnected clients
        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def broadcast_iq_json(
        self,
        i_samples: np.ndarray,
        q_samples: np.ndarray,
        center_freq_hz: float,
        sample_rate_hz: float,
        max_samples: int = 256,
    ) -> int:
        """
        Broadcast IQ samples as JSON (for frontend constellation display).

        Args:
            i_samples: In-phase samples.
            q_samples: Quadrature samples.
            center_freq_hz: Center frequency in Hz.
            sample_rate_hz: Sample rate in Hz.
            max_samples: Maximum samples to send (downsampled if needed).

        Returns:
            Number of clients sent to.
        """
        now = time.time()

        # Downsample if needed
        if len(i_samples) > max_samples:
            step = len(i_samples) // max_samples
            i_samples = i_samples[::step][:max_samples]
            q_samples = q_samples[::step][:max_samples]

        # Create JSON message
        message = {
            "type": "iq",
            "timestamp": now,
            "centerFreqHz": center_freq_hz,
            "sampleRateHz": sample_rate_hz,
            "numSamples": len(i_samples),
            "iSamples": i_samples.tolist(),
            "qSamples": q_samples.tolist(),
        }

        json_msg = json.dumps(message)

        # Broadcast to IQ subscribers (JSON mode)
        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "iq" in client.subscriptions:
                try:
                    await client.websocket.send_text(json_msg)
                    client.last_message = now
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending IQ JSON to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        # Remove disconnected clients
        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_detection(self, detection: dict) -> int:
        """Send detection event to subscribed clients."""
        message = json.dumps({"type": "detection", "data": detection, "timestamp": time.time()})

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "detections" in client.subscriptions:
                try:
                    await client.websocket.send_text(message)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending detection to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_cluster_update(self, clusters: list) -> int:
        """Send cluster update to subscribed clients."""
        message = json.dumps({"type": "clusters", "timestamp": time.time(), "data": clusters})

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "clusters" in client.subscriptions:
                try:
                    await client.websocket.send_text(message)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending cluster update to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_lora_frame(self, frame: dict) -> int:
        """Send LoRa demodulation event to subscribed clients."""
        message = json.dumps(
            {
                "type": "lora_frame",
                "timestamp": time.time(),
                "data": {
                    "frame_id": frame.get("frame_id", 0),
                    "center_freq_hz": frame.get("center_freq_hz", 0),
                    "spreading_factor": frame.get("spreading_factor", 7),
                    "bandwidth_hz": frame.get("bandwidth_hz", 125000),
                    "coding_rate": frame.get("coding_rate", "4/5"),
                    "payload_hex": frame.get("payload_hex", ""),
                    "crc_valid": frame.get("crc_valid", False),
                    "snr_db": frame.get("snr_db", 0),
                    "rssi_dbm": frame.get("rssi_dbm", -100),
                },
            }
        )

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "demodulation" in client.subscriptions:
                try:
                    await client.websocket.send_text(message)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending LoRa frame to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_ble_packet(self, packet: dict) -> int:
        """Send BLE demodulation event to subscribed clients."""
        message = json.dumps(
            {
                "type": "ble_packet",
                "timestamp": time.time(),
                "data": {
                    "packet_id": packet.get("packet_id", 0),
                    "channel": packet.get("channel", 0),
                    "access_address": packet.get("access_address", "00000000"),
                    "pdu_type": packet.get("pdu_type", "ADV_IND"),
                    "payload_hex": packet.get("payload_hex", ""),
                    "crc_valid": packet.get("crc_valid", False),
                    "rssi_dbm": packet.get("rssi_dbm", -100),
                },
            }
        )

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "demodulation" in client.subscriptions:
                try:
                    await client.websocket.send_text(message)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending BLE packet to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    # =========================================================================
    # Frontend JSON Mode (camelCase)
    # =========================================================================

    async def broadcast_spectrum_json(
        self, psd_db: np.ndarray, center_freq_hz: float, sample_rate_hz: float, fft_size: int = None
    ) -> int:
        """
        Broadcast spectrum as JSON with camelCase keys (frontend format).

        Frontend expects:
        {
            "type": "spectrum",
            "timestamp": 1733058600.123,
            "centerFreqHz": 915000000,
            "sampleRateHz": 2048000,
            "fftSize": 2048,
            "magnitudeDb": [-90.5, -89.2, ...],
            "frequencyBins": [914000000, 914001000, ...]
        }
        """
        # Rate limiting
        now = time.time()
        if now - self._last_spectrum_broadcast < self._min_broadcast_interval:
            return 0

        self._last_spectrum_broadcast = now

        if fft_size is None:
            fft_size = len(psd_db)

        # Use frontend formatter for camelCase
        message = FrontendMessageFormatter.format_spectrum(
            psd_db=psd_db,
            center_freq_hz=center_freq_hz,
            sample_rate_hz=sample_rate_hz,
            fft_size=fft_size,
            timestamp=now,
        )

        json_msg = json.dumps(message)

        # Broadcast to JSON spectrum subscribers only
        # NOTE: "spectrum" is the standard JSON format (frontend default)
        #       "spectrum_binary" is for high-performance binary mode
        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "spectrum" in client.subscriptions:
                try:
                    await client.websocket.send_text(json_msg)
                    client.last_message = now
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending spectrum JSON to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_detection_json(self, detection: dict) -> int:
        """
        Send detection as JSON with camelCase keys (frontend format).

        Frontend expects:
        {
            "type": "detection",
            "timestamp": 1733058601.456,
            "id": "det_20251201_103001_456",
            "centerFreqHz": 915200000,
            "bandwidthHz": 125000,
            ...
        }
        """
        message = FrontendMessageFormatter.format_detection(
            detection_id=detection.get("detection_id", 0),
            center_freq_hz=detection.get("center_freq_hz", 0),
            bandwidth_hz=detection.get("bandwidth_hz", 0),
            peak_power_db=detection.get("peak_power_db", -100),
            snr_db=detection.get("snr_db", 0),
            modulation_type=detection.get("modulation_type", detection.get("label", "Unknown")),
            confidence=detection.get("confidence", 0),
            duration=detection.get("duration", 0),
            timestamp=time.time(),
        )

        json_msg = json.dumps(message)

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "detections" in client.subscriptions:
                try:
                    await client.websocket.send_text(json_msg)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending detection JSON to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_clusters_json(self, clusters: list) -> int:
        """
        Send cluster update as JSON with camelCase keys (frontend format).

        Frontend expects:
        {
            "type": "cluster",
            "timestamp": 1733058602.789,
            "clusters": [
                {
                    "id": "cluster_915mhz",
                    "centerFreqHz": 915000000,
                    "freqRangeHz": [914500000, 915500000],
                    ...
                }
            ]
        }
        """
        message = FrontendMessageFormatter.format_clusters(clusters=clusters, timestamp=time.time())

        json_msg = json.dumps(message)

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "clusters" in client.subscriptions:
                try:
                    await client.websocket.send_text(json_msg)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending clusters JSON to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_stats_json(self, stats: dict) -> int:
        """
        Send pipeline statistics as JSON (frontend format).

        Frontend expects:
        {
            "type": "stats",
            "timestamp": 1733058603.123,
            "samplesProcessed": 50000000,
            "droppedSamples": 1000,
            "dropRate": 0.00002,
            "bufferFillLevel": 0.45,
            ...
        }

        Clients subscribe to "stats" channel to receive these updates.
        """
        message = FrontendMessageFormatter.format_stats(
            samples_processed=stats.get("samples_processed", 0),
            detections_count=stats.get("detections_count", 0),
            dropped_samples=stats.get("dropped_samples", 0),
            buffer_fill_level=stats.get("buffer_fill_level", 0),
            throughput_msps=stats.get("current_throughput_msps", 0),
            processing_latency_ms=stats.get("processing_latency_ms", 0),
            consecutive_errors=stats.get("consecutive_errors", 0),
            state=stats.get("state", "unknown"),
            sdr_throttled=stats.get("sdr_throttled", False),
            timestamp=time.time(),
        )

        json_msg = json.dumps(message)

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "stats" in client.subscriptions:
                try:
                    await client.websocket.send_text(json_msg)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending stats JSON to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_lora_frame_json(self, frame: dict) -> int:
        """Send LoRa frame as detection (frontend format)."""
        message = FrontendMessageFormatter.format_lora_frame(frame, time.time())
        json_msg = json.dumps(message)

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "detections" in client.subscriptions or "demodulation" in client.subscriptions:
                try:
                    await client.websocket.send_text(json_msg)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending LoRa frame JSON to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def send_ble_packet_json(self, packet: dict) -> int:
        """Send BLE packet as detection (frontend format)."""
        message = FrontendMessageFormatter.format_ble_packet(packet, time.time())
        json_msg = json.dumps(message)

        sent_count = 0
        clients_to_remove: list[str] = []
        for client_id, client in list(self._clients.items()):
            if "detections" in client.subscriptions or "demodulation" in client.subscriptions:
                try:
                    await client.websocket.send_text(json_msg)
                    sent_count += 1
                except (WebSocketDisconnect, ConnectionResetError):
                    clients_to_remove.append(client_id)
                except Exception as e:
                    logger.warning(f"Error sending BLE packet JSON to client {client_id}: {e}")
                    clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            self._clients.pop(client_id, None)

        return sent_count

    async def _handle_control_message(self, client_id: str, message: str):
        """Handle control messages from clients."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "subscribe":
                channel = data.get("channel")
                if channel and client_id in self._clients:
                    self._clients[client_id].subscriptions.add(channel)

            elif msg_type == "unsubscribe":
                channel = data.get("channel")
                if channel and client_id in self._clients:
                    self._clients[client_id].subscriptions.discard(channel)

            elif msg_type == "ping":
                # Respond with pong
                if client_id in self._clients:
                    await self._clients[client_id].websocket.send_text(
                        json.dumps({"type": "pong", "timestamp": time.time()})
                    )

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from client {client_id}: {e}")

    def register_client(self, websocket, subscriptions: set[str] | None = None) -> str:
        """Register a new client connection."""
        self._client_counter += 1
        client_id = f"client_{self._client_counter}"

        self._clients[client_id] = ClientConnection(
            websocket=websocket,
            client_id=client_id,
            connected_at=time.time(),
            last_message=time.time(),
            subscriptions=subscriptions or {"spectrum"},
            rate_limit_hz=self._spectrum_rate_hz,
        )

        return client_id

    def unregister_client(self, client_id: str):
        """Remove a client connection."""
        self._clients.pop(client_id, None)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            "num_clients": len(self._clients),
            "max_clients": self._max_clients,
            "spectrum_rate_hz": self._spectrum_rate_hz,
            "clients": [
                {
                    "id": c.client_id,
                    "connected_at": c.connected_at,
                    "subscriptions": list(c.subscriptions),
                }
                for c in self._clients.values()
            ],
        }


def create_websocket_app(server: SpectrumWebSocketServer, api_manager=None) -> FastAPI:
    """Create FastAPI application with WebSocket endpoints.

    Args:
        server: SpectrumWebSocketServer instance
        api_manager: Optional RFForensicsAPI instance for REST endpoints that need it
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available")

    app = FastAPI(title="RF Forensics WebSocket API")

    # Store API manager in app state for dependency injection (same as rest_api.py)
    if api_manager is not None:
        app.state.api_manager = api_manager

    # Use environment variable for CORS origins (default to common dev origins)
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(
        ","
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include REST API routers so frontend can use port 8765 for everything
    try:
        from rf_forensics.api.routers import (
            analysis_router,
            config_router,
            detections_router,
            recordings_router,
            sdr_router,
            system_router,
        )

        app.include_router(system_router)
        app.include_router(config_router)
        app.include_router(detections_router)
        app.include_router(recordings_router)
        app.include_router(sdr_router)
        app.include_router(analysis_router)
        logger.info("REST API routers included on WebSocket port")
    except ImportError as e:
        logger.warning(f"Could not include REST routers: {e}")

    @app.websocket("/ws/spectrum")
    async def spectrum_websocket(websocket: WebSocket):
        await websocket.accept()
        # Register with both binary and JSON subscriptions - frontend uses binary
        client_id = server.register_client(websocket, {"spectrum", "spectrum_binary"})

        try:
            await server.handle_spectrum_client(websocket, client_id)
        except WebSocketDisconnect:
            pass
        finally:
            server.unregister_client(client_id)

    @app.websocket("/ws/detections")
    async def detections_websocket(websocket: WebSocket):
        await websocket.accept()
        client_id = server.register_client(websocket, {"detections"})

        try:
            # Use async for to properly wait for client messages/disconnects
            async for message in websocket.iter_text():
                await server._handle_control_message(client_id, message)
        except WebSocketDisconnect:
            pass
        finally:
            server.unregister_client(client_id)

    @app.websocket("/ws/clusters")
    async def clusters_websocket(websocket: WebSocket):
        await websocket.accept()
        client_id = server.register_client(websocket, {"clusters"})

        try:
            # Use async for to properly wait for client messages/disconnects
            async for message in websocket.iter_text():
                await server._handle_control_message(client_id, message)
        except WebSocketDisconnect:
            pass
        finally:
            server.unregister_client(client_id)

    @app.websocket("/ws/demodulation")
    async def demodulation_websocket(websocket: WebSocket):
        """WebSocket endpoint for demodulation events (LoRa/BLE)."""
        await websocket.accept()
        client_id = server.register_client(websocket, {"demodulation"})

        try:
            # Use async for to properly wait for client messages/disconnects
            async for message in websocket.iter_text():
                await server._handle_control_message(client_id, message)
        except WebSocketDisconnect:
            pass
        finally:
            server.unregister_client(client_id)

    @app.websocket("/ws/iq")
    async def iq_websocket(websocket: WebSocket):
        """WebSocket endpoint for IQ sample streaming (constellation display)."""
        await websocket.accept()
        client_id = server.register_client(websocket, {"iq"})

        try:
            # Use async for to properly wait for client messages/disconnects
            async for message in websocket.iter_text():
                await server._handle_control_message(client_id, message)
        except WebSocketDisconnect:
            pass
        finally:
            server.unregister_client(client_id)

    return app
