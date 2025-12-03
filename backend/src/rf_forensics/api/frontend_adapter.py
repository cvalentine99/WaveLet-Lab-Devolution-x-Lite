"""
Frontend Adapter Layer

Transforms backend snake_case data to frontend camelCase format.
Keeps internal backend optimized while providing clean API for JavaScript frontends.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

import numpy as np

# =============================================================================
# Case Conversion Utilities
# =============================================================================


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def transform_keys(obj: Any, converter: Callable[[str], str]) -> Any:
    """Recursively transform dictionary keys using converter function."""
    if isinstance(obj, dict):
        return {converter(k): transform_keys(v, converter) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [transform_keys(item, converter) for item in obj]
    return obj


def to_camel_case(data: dict) -> dict:
    """Transform all keys in nested dict from snake_case to camelCase."""
    return transform_keys(data, snake_to_camel)


def to_snake_case(data: dict) -> dict:
    """Transform all keys in nested dict from camelCase to snake_case."""
    return transform_keys(data, camel_to_snake)


# =============================================================================
# Frontend Message Formatters
# =============================================================================


class FrontendMessageFormatter:
    """
    Formats backend data into frontend-expected JSON structures.

    All output uses camelCase keys per JavaScript conventions.
    """

    @staticmethod
    def format_spectrum(
        psd_db: np.ndarray,
        center_freq_hz: float,
        sample_rate_hz: float,
        fft_size: int,
        timestamp: float = None,
    ) -> dict[str, Any]:
        """
        Format spectrum data for frontend.

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
        if timestamp is None:
            timestamp = time.time()

        # Generate frequency bins
        freq_start = center_freq_hz - sample_rate_hz / 2
        freq_step = sample_rate_hz / fft_size
        frequency_bins = [freq_start + i * freq_step for i in range(fft_size)]

        # Convert numpy array to list if needed
        magnitude_db = psd_db.tolist() if isinstance(psd_db, np.ndarray) else list(psd_db)

        return {
            "type": "spectrum",
            "timestamp": timestamp,
            "centerFreqHz": int(center_freq_hz),
            "sampleRateHz": int(sample_rate_hz),
            "fftSize": fft_size,
            "magnitudeDb": magnitude_db,
            "frequencyBins": frequency_bins,
        }

    @staticmethod
    def format_detection(
        detection_id: int,
        center_freq_hz: float,
        bandwidth_hz: float,
        peak_power_db: float,
        snr_db: float,
        timestamp: float = None,
        modulation_type: str = "Unknown",
        modulation_confidence: float = 0.0,
        duration: float = 0.0,
        bandwidth_3db_hz: float = None,
        bandwidth_6db_hz: float = None,
        start_bin: int = None,
        end_bin: int = None,
        top_k_predictions: list = None,
        # New fields per BACKEND_CONTRACT.md
        track_id: str = None,
        duty_cycle: float = None,
        frames_seen: int = None,
        cluster_id: int = None,
        anomaly_score: float = None,
        symbol_rate: float = None,
    ) -> dict[str, Any]:
        """
        Format detection for frontend per BACKEND_CONTRACT.md.

        Frontend expects (Section 2 - Detection Stream):
        {
            "type": "detection",
            "timestamp": 1701590400.123,
            "data": {
                "detection_id": 12345,
                "center_freq_hz": 915125000,
                "bandwidth_hz": 125000,
                "bandwidth_3db_hz": 100000,
                "bandwidth_6db_hz": 150000,
                "peak_power_db": -45.2,
                "snr_db": 18.5,
                "start_bin": 512,
                "end_bin": 520,
                "modulation_type": "LoRa",
                "modulation_confidence": 0.92,
                "top_k_predictions": [...],
                "track_id": "trk-abc123",
                "duty_cycle": 0.15,
                "frames_seen": 42,
                "cluster_id": 3,
                "anomaly_score": 0.12,
                "symbol_rate": 50000
            }
        }

        Note: Backend uses snake_case, frontend converts to camelCase (Section 3).
        """
        if timestamp is None:
            timestamp = time.time()

        # Build data payload with snake_case per contract
        data = {
            "detection_id": detection_id,
            "center_freq_hz": int(center_freq_hz),
            "bandwidth_hz": int(bandwidth_hz),
            "peak_power_db": round(peak_power_db, 2),
            "snr_db": round(snr_db, 2),
            "modulation_type": modulation_type,
            "modulation_confidence": round(modulation_confidence, 4),
        }

        # Bandwidth shape fields
        if bandwidth_3db_hz is not None:
            data["bandwidth_3db_hz"] = int(bandwidth_3db_hz)
        if bandwidth_6db_hz is not None:
            data["bandwidth_6db_hz"] = int(bandwidth_6db_hz)

        # Bin indices
        if start_bin is not None:
            data["start_bin"] = start_bin
        if end_bin is not None:
            data["end_bin"] = end_bin

        # Top-K predictions for AMC
        if top_k_predictions:
            data["top_k_predictions"] = [
                {"modulation_type": mod, "confidence": round(conf, 4)}
                for mod, conf in top_k_predictions
            ]

        # Tracking fields (from TrackedDetection)
        if track_id is not None:
            data["track_id"] = track_id
        if duty_cycle is not None:
            data["duty_cycle"] = round(duty_cycle, 3)
        if frames_seen is not None:
            data["frames_seen"] = frames_seen

        # Clustering & anomaly fields (HIGH priority per contract)
        if cluster_id is not None:
            data["cluster_id"] = cluster_id
        if anomaly_score is not None:
            data["anomaly_score"] = round(anomaly_score, 3)
        if symbol_rate is not None:
            data["symbol_rate"] = int(symbol_rate)

        return {"type": "detection", "timestamp": timestamp, "data": data}

    @staticmethod
    def format_cluster(
        cluster_id: int,
        center_freq_hz: float,
        freq_range_hz: tuple = None,
        signal_count: int = 0,
        avg_power_db: float = -100.0,
        dominant_modulation: str = "Unknown",
        activity: float = 0.0,
        avg_snr_db: float = 0.0,
        label: str = "",
        # New fields per BACKEND_CONTRACT.md Section 2.3
        dominant_frequency_hz: float = None,
        avg_bandwidth_hz: float = None,
        detection_count: int = None,
        signal_type_hint: str = None,
        avg_duty_cycle: float = None,
        unique_tracks: int = None,
        avg_bw_3db_ratio: float = None,
    ) -> dict[str, Any]:
        """
        Format single cluster for frontend per BACKEND_CONTRACT.md.

        Frontend expects (Section 2.3 - Cluster Stream):
        {
            "cluster_id": 3,
            "size": 42,
            "center_freq_hz": 915125000,
            "dominant_frequency_hz": 915125000,
            "freq_range_hz": [915000000, 915250000],
            "avg_snr_db": 15.2,
            "avg_power_db": -48.5,
            "avg_bandwidth_hz": 125000,
            "detection_count": 156,
            "label": "LoRa Gateway",
            "signal_type_hint": "LoRa",
            "avg_duty_cycle": 0.12,
            "unique_tracks": 3,
            "avg_bw_3db_ratio": 0.8
        }

        Note: Backend uses snake_case per contract Section 3.
        """
        # Default freq_range if not provided
        if freq_range_hz is None:
            freq_range_hz = (center_freq_hz, center_freq_hz)

        # Use detection_count if provided, else signal_count
        count = detection_count if detection_count is not None else signal_count

        result = {
            "cluster_id": cluster_id,
            "size": count,
            "center_freq_hz": int(center_freq_hz),
            "dominant_frequency_hz": int(dominant_frequency_hz or center_freq_hz),
            "freq_range_hz": [int(freq_range_hz[0]), int(freq_range_hz[1])],
            "avg_snr_db": round(avg_snr_db, 2),
            "avg_power_db": round(avg_power_db, 2),
            "avg_bandwidth_hz": int(avg_bandwidth_hz) if avg_bandwidth_hz else 0,
            "detection_count": count,
            "label": label or f"Cluster_{cluster_id}",
        }

        # ML classification hint (CRITICAL per contract)
        if signal_type_hint:
            result["signal_type_hint"] = signal_type_hint
        elif dominant_modulation and dominant_modulation != "Unknown":
            result["signal_type_hint"] = dominant_modulation

        # Tracking metrics
        if avg_duty_cycle is not None:
            result["avg_duty_cycle"] = round(avg_duty_cycle, 3)
        if unique_tracks is not None:
            result["unique_tracks"] = unique_tracks
        if avg_bw_3db_ratio is not None:
            result["avg_bw_3db_ratio"] = round(avg_bw_3db_ratio, 3)

        return result

    @staticmethod
    def format_stats(
        samples_processed: int,
        detections_count: int,
        dropped_samples: int,
        buffer_fill_level: float,
        throughput_msps: float,
        processing_latency_ms: float,
        consecutive_errors: int = 0,
        state: str = "running",
        sdr_throttled: bool = False,
        timestamp: float = None,
    ) -> dict[str, Any]:
        """
        Format pipeline statistics for frontend.

        Frontend expects:
        {
            "type": "stats",
            "timestamp": 1733058603.123,
            "samplesProcessed": 50000000,
            "detectionsCount": 42,
            "droppedSamples": 1000,
            "dropRate": 0.00002,
            "bufferFillLevel": 0.45,
            "throughputMsps": 48.5,
            "processingLatencyMs": 2.3,
            "consecutiveErrors": 0,
            "state": "running",
            "sdrThrottled": false
        }
        """
        if timestamp is None:
            timestamp = time.time()

        # Calculate drop rate
        total = samples_processed + dropped_samples
        drop_rate = dropped_samples / total if total > 0 else 0.0

        return {
            "type": "stats",
            "timestamp": timestamp,
            "samplesProcessed": samples_processed,
            "detectionsCount": detections_count,
            "droppedSamples": dropped_samples,
            "dropRate": round(drop_rate, 6),
            "bufferFillLevel": round(buffer_fill_level, 3),
            "throughputMsps": round(throughput_msps, 2),
            "processingLatencyMs": round(processing_latency_ms, 2),
            "consecutiveErrors": consecutive_errors,
            "state": state,
            "sdrThrottled": sdr_throttled,
        }

    @staticmethod
    def format_clusters(clusters: list[dict], timestamp: float = None) -> dict[str, Any]:
        """
        Format cluster update message for frontend per BACKEND_CONTRACT.md.

        Frontend expects (Section 2.3 - Cluster Stream):
        {
            "type": "clusters",  # Note: plural
            "timestamp": 1701590400.123,
            "data": [...]  # Nested under "data" key
        }
        """
        if timestamp is None:
            timestamp = time.time()

        formatted_clusters = []
        for i, c in enumerate(clusters):
            # Handle both snake_case backend format and pre-formatted
            if "cluster_id" in c:
                formatted_clusters.append(
                    FrontendMessageFormatter.format_cluster(
                        cluster_id=c["cluster_id"],
                        center_freq_hz=c.get("dominant_frequency_hz", c.get("center_freq_hz", 0)),
                        freq_range_hz=c.get("freq_range_hz", (0, 0)),
                        signal_count=c.get("size", c.get("signal_count", 0)),
                        avg_power_db=c.get("avg_power_db", -100),
                        dominant_modulation=c.get("label", c.get("dominant_modulation", "Unknown")),
                        activity=c.get("activity", 0),
                        avg_snr_db=c.get("avg_snr_db", 0),
                        label=c.get("label", ""),
                        # New fields per contract
                        dominant_frequency_hz=c.get("dominant_frequency_hz"),
                        avg_bandwidth_hz=c.get("avg_bandwidth_hz"),
                        detection_count=c.get("detection_count"),
                        signal_type_hint=c.get("signal_type_hint"),
                        avg_duty_cycle=c.get("avg_duty_cycle"),
                        unique_tracks=c.get("unique_tracks"),
                        avg_bw_3db_ratio=c.get("avg_bw_3db_ratio"),
                    )
                )
            else:
                # Already in camelCase or different format
                formatted_clusters.append(c)

        return {
            "type": "clusters",  # Plural per contract
            "timestamp": timestamp,
            "data": formatted_clusters,  # Nested under "data" per contract
        }

    @staticmethod
    def format_lora_frame(frame: dict, timestamp: float = None) -> dict[str, Any]:
        """Format LoRa frame for frontend demodulation display."""
        if timestamp is None:
            timestamp = time.time()

        return {
            "type": "detection",  # Frontend treats as detection
            "timestamp": timestamp,
            "id": f"lora_{frame.get('frame_id', 0)}",
            "centerFreqHz": frame.get("center_freq_hz", 0),
            "bandwidthHz": frame.get("bandwidth_hz", 125000),
            "peakPowerDb": frame.get("rssi_dbm", -100),
            "snrDb": frame.get("snr_db", 0),
            "modulationType": "LoRa",
            "confidence": 1.0 if frame.get("crc_valid", False) else 0.5,
            "duration": 0,
            # LoRa-specific fields
            "spreadingFactor": frame.get("spreading_factor", 7),
            "codingRate": frame.get("coding_rate", "4/5"),
            "payloadHex": frame.get("payload_hex", ""),
            "crcValid": frame.get("crc_valid", False),
        }

    @staticmethod
    def format_ble_packet(packet: dict, timestamp: float = None) -> dict[str, Any]:
        """Format BLE packet for frontend demodulation display."""
        if timestamp is None:
            timestamp = time.time()

        # Map BLE channel to frequency
        channel = packet.get("channel", 37)
        if channel == 37:
            freq_hz = 2402000000
        elif channel == 38:
            freq_hz = 2426000000
        elif channel == 39:
            freq_hz = 2480000000
        else:
            freq_hz = 2402000000 + channel * 2000000

        return {
            "type": "detection",
            "timestamp": timestamp,
            "id": f"ble_{packet.get('packet_id', 0)}",
            "centerFreqHz": freq_hz,
            "bandwidthHz": 2000000,  # BLE channel width
            "peakPowerDb": packet.get("rssi_dbm", -100),
            "snrDb": 15,  # Estimated
            "modulationType": "Bluetooth",
            "confidence": 1.0 if packet.get("crc_valid", False) else 0.5,
            "duration": 0,
            # BLE-specific fields
            "channel": channel,
            "accessAddress": packet.get("access_address", ""),
            "pduType": packet.get("pdu_type", ""),
            "payloadHex": packet.get("payload_hex", ""),
            "crcValid": packet.get("crc_valid", False),
        }


# =============================================================================
# REST API Response Adapter
# =============================================================================


class RESTResponseAdapter:
    """
    Adapts backend REST responses to frontend-expected format.
    """

    @staticmethod
    def health_response(
        gpu_available: bool = True, gpu_name: str = "NVIDIA GPU", backend_version: str = "1.0.0"
    ) -> dict[str, Any]:
        """
        Format health check response.

        Frontend expects:
        {
            "status": "ok",
            "gpu_available": true,
            "gpu_name": "NVIDIA RTX 4090",
            "backend_version": "1.0.0",
            "timestamp": "2025-12-01T10:30:00Z"
        }
        """
        from datetime import datetime

        return {
            "status": "ok" if gpu_available else "error",
            "gpu_available": gpu_available,  # Keep underscore per frontend spec
            "gpu_name": gpu_name,
            "backend_version": backend_version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    @staticmethod
    def sdr_config_response(
        config: dict, success: bool = True, message: str = ""
    ) -> dict[str, Any]:
        """
        Format SDR config response.

        Frontend expects:
        {
            "success": true,
            "message": "SDR configured successfully",
            "appliedConfig": {
                "centerFreqHz": 915000000,
                "sampleRateHz": 2048000,
                "gainDb": 40,
                "bandwidth": 2000000
            }
        }
        """
        # Convert snake_case config to camelCase
        applied_config = {
            "centerFreqHz": config.get("center_freq_hz", 915000000),
            "sampleRateHz": config.get("sample_rate_hz", 10000000),
            "gainDb": config.get("gain_db", 40),
            "bandwidth": config.get("bandwidth_hz", 10000000),
        }

        return {
            "success": success,
            "message": message
            or ("SDR configured successfully" if success else "Configuration failed"),
            "appliedConfig": applied_config,
        }

    @staticmethod
    def monitoring_start_response(session_id: str, success: bool = True) -> dict[str, Any]:
        """
        Format monitoring start response.

        Frontend expects:
        {
            "success": true,
            "message": "Monitoring started",
            "sessionId": "session_abc123"
        }
        """
        return {
            "success": success,
            "message": "Monitoring started" if success else "Failed to start monitoring",
            "sessionId": session_id,
        }

    @staticmethod
    def monitoring_stop_response(success: bool = True) -> dict[str, Any]:
        """Format monitoring stop response."""
        return {
            "success": success,
            "message": "Monitoring stopped" if success else "Failed to stop monitoring",
        }

    @staticmethod
    def recording_start_response(
        recording_id: str, filepath: str, success: bool = True
    ) -> dict[str, Any]:
        """
        Format recording start response.

        Frontend expects:
        {
            "success": true,
            "recordingId": "rec_xyz789",
            "filepath": "/data/recordings/capture_915mhz.iq"
        }
        """
        return {"success": success, "recordingId": recording_id, "filepath": filepath}

    @staticmethod
    def recording_stop_response(
        filepath: str, file_size: int, duration: float, success: bool = True
    ) -> dict[str, Any]:
        """
        Format recording stop response.

        Frontend expects:
        {
            "success": true,
            "filepath": "/data/recordings/capture_915mhz.iq",
            "fileSize": 125829120,
            "duration": 60.5
        }
        """
        return {
            "success": success,
            "filepath": filepath,
            "fileSize": file_size,
            "duration": duration,
        }

    @staticmethod
    def error_response(error_code: str, message: str) -> dict[str, Any]:
        """
        Format error response.

        Frontend expects:
        {
            "success": false,
            "error": "error_code",
            "message": "Human-readable error description"
        }
        """
        return {"success": False, "error": error_code, "message": message}


# =============================================================================
# Request Adapter (camelCase -> snake_case)
# =============================================================================


class RESTRequestAdapter:
    """
    Adapts frontend camelCase requests to backend snake_case format.
    """

    @staticmethod
    def parse_sdr_config(request: dict) -> dict:
        """
        Parse SDR config from frontend format.

        Frontend sends:
        {
            "centerFreqHz": 915000000,
            "sampleRateHz": 2048000,
            "gainDb": 40,
            "bandwidth": 2000000
        }

        Backend needs:
        {
            "center_freq_hz": 915000000,
            "sample_rate_hz": 2048000,
            "gain_db": 40,
            "bandwidth_hz": 2000000
        }
        """
        return {
            "center_freq_hz": request.get("centerFreqHz", request.get("center_freq_hz")),
            "sample_rate_hz": request.get("sampleRateHz", request.get("sample_rate_hz")),
            "gain_db": request.get("gainDb", request.get("gain_db")),
            "bandwidth_hz": request.get(
                "bandwidth", request.get("bandwidthHz", request.get("bandwidth_hz"))
            ),
        }

    @staticmethod
    def parse_monitoring_start(request: dict) -> dict:
        """
        Parse monitoring start request.

        Frontend sends:
        {
            "mode": "spectrum",
            "fftSize": 2048,
            "detectionThreshold": -80
        }
        """
        return {
            "mode": request.get("mode", "both"),
            "fft_size": request.get("fftSize", request.get("fft_size", 2048)),
            "detection_threshold": request.get(
                "detectionThreshold", request.get("detection_threshold", -80)
            ),
        }

    @staticmethod
    def parse_recording_start(request: dict) -> dict:
        """
        Parse recording start request.

        Frontend sends:
        {
            "filename": "capture_915mhz.iq",
            "format": "complex_float32",
            "duration": 60
        }

        Backend needs:
        {
            "name": "capture_915mhz",
            "description": "",
            "duration_seconds": 60
        }
        """
        filename = request.get("filename", "recording")
        # Extract name from filename
        name = filename.rsplit(".", 1)[0] if "." in filename else filename

        return {
            "name": name,
            "description": f"Format: {request.get('format', 'complex_float32')}",
            "duration_seconds": request.get("duration", None),
        }

    @staticmethod
    def parse_recording_stop(request: dict) -> dict:
        """Parse recording stop request."""
        return {"recording_id": request.get("recordingId", request.get("recording_id"))}


# =============================================================================
# Convenience function for generic transformation
# =============================================================================


def adapt_response(data: dict, to_frontend: bool = True) -> dict:
    """
    Generic adapter for converting between frontend and backend formats.

    Args:
        data: Dictionary to transform
        to_frontend: If True, convert to camelCase. If False, convert to snake_case.

    Returns:
        Transformed dictionary
    """
    if to_frontend:
        return to_camel_case(data)
    return to_snake_case(data)
