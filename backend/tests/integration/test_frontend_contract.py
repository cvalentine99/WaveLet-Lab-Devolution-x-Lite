"""
Frontend Contract Compliance Tests

Tests that backend outputs conform to BACKEND_CONTRACT.md specifications.
These tests validate the format of WebSocket events and REST responses
to ensure frontend compatibility.
"""

import pytest
import struct
import time
import numpy as np

from rf_forensics.api.contract_validator import (
    validate_detection_event,
    validate_clusters_event,
    validate_spectrum_frame,
    validate_iq_frame,
    validate_ble_packet,
    validate_lora_frame,
    validate_status_response,
    parse_spectrum_frame,
    parse_iq_frame,
    SPECTRUM_HEADER_SIZE,
    IQ_HEADER_SIZE,
)


# ============================================================================
# Detection Event Tests
# ============================================================================

class TestDetectionEventContract:
    """Test detection event format compliance."""

    def test_valid_detection_event(self):
        """Valid detection event should pass validation."""
        event = {
            "type": "detection",
            "timestamp": time.time(),
            "data": {
                "detection_id": 1,
                "center_freq_hz": 915e6,
                "bandwidth_hz": 500000,
                "peak_power_db": -50.0,
                "snr_db": 25.0,
                "timestamp": time.time(),
                "modulation_type": "LoRa",
                "modulation_confidence": 0.85,
            }
        }

        errors = validate_detection_event(event)
        assert errors == [], f"Valid event should have no errors: {errors}"

    def test_detection_with_optional_fields(self):
        """Detection with optional fields should be valid."""
        event = {
            "type": "detection",
            "timestamp": time.time(),
            "data": {
                "detection_id": 42,
                "center_freq_hz": 2.4e9,
                "bandwidth_hz": 1000000,
                "peak_power_db": -45.0,
                "snr_db": 30.0,
                "timestamp": time.time(),
                "modulation_type": "BLE",
                "modulation_confidence": 0.92,
                # Optional fields
                "bandwidth_3db_hz": 800000,
                "bandwidth_6db_hz": 1200000,
                "cluster_id": 5,
                "anomaly_score": 0.15,
                "track_id": 123,
                "duty_cycle": 0.75,
            }
        }

        errors = validate_detection_event(event)
        assert errors == [], f"Event with optional fields should be valid: {errors}"

    def test_detection_missing_data_wrapper(self):
        """Detection without data wrapper should fail."""
        event = {
            "type": "detection",
            "timestamp": time.time(),
            "detection_id": 1,  # Flat, not nested
            "center_freq_hz": 915e6,
        }

        errors = validate_detection_event(event)
        assert any("data" in e for e in errors), "Should report missing 'data' wrapper"

    def test_detection_missing_required_fields(self):
        """Detection missing required fields should fail."""
        event = {
            "type": "detection",
            "timestamp": time.time(),
            "data": {
                "detection_id": 1,
                # Missing: center_freq_hz, bandwidth_hz, etc.
            }
        }

        errors = validate_detection_event(event)
        assert len(errors) > 0, "Missing fields should cause errors"
        assert any("center_freq_hz" in e for e in errors)

    def test_detection_wrong_type(self):
        """Detection with wrong type should fail."""
        event = {
            "type": "detect",  # Wrong - should be "detection"
            "timestamp": time.time(),
            "data": {"detection_id": 1}
        }

        errors = validate_detection_event(event)
        assert any("type" in e for e in errors)

    def test_anomaly_score_range(self):
        """Anomaly score must be 0.0-1.0."""
        event = {
            "type": "detection",
            "timestamp": time.time(),
            "data": {
                "detection_id": 1,
                "center_freq_hz": 915e6,
                "bandwidth_hz": 500000,
                "peak_power_db": -50.0,
                "snr_db": 25.0,
                "timestamp": time.time(),
                "modulation_type": "LoRa",
                "modulation_confidence": 0.85,
                "anomaly_score": 1.5,  # Invalid - out of range
            }
        }

        errors = validate_detection_event(event)
        assert any("anomaly_score" in e for e in errors)


# ============================================================================
# Cluster Event Tests
# ============================================================================

class TestClustersEventContract:
    """Test clusters event format compliance."""

    def test_valid_clusters_event(self):
        """Valid clusters event should pass validation."""
        event = {
            "type": "clusters",  # Plural!
            "timestamp": time.time(),
            "data": [
                {
                    "cluster_id": 0,
                    "size": 15,
                    "dominant_frequency_hz": 915.5e6,
                    "avg_snr_db": 22.5,
                },
                {
                    "cluster_id": 1,
                    "size": 8,
                    "dominant_frequency_hz": 868.1e6,
                    "avg_snr_db": 18.2,
                }
            ]
        }

        errors = validate_clusters_event(event)
        assert errors == [], f"Valid event should have no errors: {errors}"

    def test_clusters_with_optional_fields(self):
        """Clusters with optional fields should be valid."""
        event = {
            "type": "clusters",
            "timestamp": time.time(),
            "data": [
                {
                    "cluster_id": 0,
                    "size": 15,
                    "dominant_frequency_hz": 915.5e6,
                    "avg_snr_db": 22.5,
                    # Optional
                    "label": "LoRa Gateway",
                    "signal_type_hint": "LoRa",
                    "freq_range_hz": [915.0e6, 916.0e6],
                    "avg_power_db": -52.3,
                    "unique_tracks": 3,
                }
            ]
        }

        errors = validate_clusters_event(event)
        assert errors == [], f"Event with optional fields should be valid: {errors}"

    def test_clusters_singular_type_fails(self):
        """Type 'cluster' (singular) should fail - must be 'clusters'."""
        event = {
            "type": "cluster",  # Wrong - should be "clusters"
            "timestamp": time.time(),
            "data": []
        }

        errors = validate_clusters_event(event)
        assert any("clusters" in e and "plural" in e for e in errors)

    def test_clusters_missing_data_wrapper(self):
        """Clusters without data wrapper should fail."""
        event = {
            "type": "clusters",
            "timestamp": time.time(),
            "clusters": []  # Wrong - should be "data"
        }

        errors = validate_clusters_event(event)
        assert any("data" in e for e in errors)

    def test_freq_range_format(self):
        """freq_range_hz must be [min, max] list."""
        event = {
            "type": "clusters",
            "timestamp": time.time(),
            "data": [
                {
                    "cluster_id": 0,
                    "size": 5,
                    "dominant_frequency_hz": 915e6,
                    "avg_snr_db": 20.0,
                    "freq_range_hz": 915e6,  # Wrong - should be list
                }
            ]
        }

        errors = validate_clusters_event(event)
        assert any("freq_range_hz" in e for e in errors)


# ============================================================================
# Binary Spectrum Frame Tests
# ============================================================================

class TestSpectrumFrameContract:
    """Test binary spectrum frame format compliance (v2 - full precision)."""

    def test_valid_spectrum_frame(self):
        """Valid spectrum frame should pass validation."""
        timestamp = time.time()
        center_freq = 915e6
        span = 10e6
        num_bins = 1024

        # Pack header (28 bytes - v2 format with float64 frequencies)
        header = struct.pack('<dddI', timestamp, center_freq, span, num_bins)

        # Create payload
        payload = np.random.randint(0, 256, num_bins, dtype=np.uint8).tobytes()

        frame = header + payload

        errors = validate_spectrum_frame(frame)
        assert errors == [], f"Valid frame should have no errors: {errors}"

    def test_spectrum_header_size(self):
        """Spectrum header must be exactly 28 bytes (v2 format)."""
        assert SPECTRUM_HEADER_SIZE == 28, "Spectrum header must be 28 bytes per v2 contract"

    def test_spectrum_frame_too_short(self):
        """Frame shorter than header should fail."""
        short_frame = b'\x00' * 20  # Old 20-byte format should be rejected

        errors = validate_spectrum_frame(short_frame)
        assert len(errors) > 0
        assert any("too short" in e for e in errors)

    def test_spectrum_payload_size_mismatch(self):
        """Payload size not matching header should fail."""
        timestamp = time.time()
        center_freq = 915e6
        span = 10e6
        num_bins = 1024

        header = struct.pack('<dddI', timestamp, center_freq, span, num_bins)
        payload = b'\x00' * 512  # Wrong size - should be 1024

        frame = header + payload

        errors = validate_spectrum_frame(frame)
        assert any("payload size mismatch" in e for e in errors)

    def test_parse_spectrum_frame(self):
        """Parse should extract correct values."""
        timestamp = 1234567890.123
        center_freq = 915e6
        span = 10e6
        num_bins = 256

        header = struct.pack('<dddI', timestamp, center_freq, span, num_bins)
        payload = np.arange(num_bins, dtype=np.uint8).tobytes()
        frame = header + payload

        result = parse_spectrum_frame(frame)
        assert result is not None
        assert abs(result["timestamp"] - timestamp) < 0.001
        assert result["center_freq_hz"] == center_freq
        assert result["span_hz"] == span
        assert result["num_bins"] == num_bins
        assert len(result["magnitude_uint8"]) == num_bins

    def test_ghz_frequency_precision(self):
        """v2 format should preserve GHz frequency precision."""
        timestamp = time.time()
        # WiFi channel 6: exactly 2437000000 Hz
        center_freq = 2437000000.0
        span = 20000000.0
        num_bins = 512

        header = struct.pack('<dddI', timestamp, center_freq, span, num_bins)
        payload = b'\x80' * num_bins
        frame = header + payload

        result = parse_spectrum_frame(frame)
        assert result is not None
        # With float64, exact Hz precision should be preserved
        assert result["center_freq_hz"] == center_freq
        assert result["span_hz"] == span


# ============================================================================
# Binary IQ Frame Tests
# ============================================================================

class TestIQFrameContract:
    """Test binary IQ frame format compliance."""

    def test_valid_iq_frame(self):
        """Valid IQ frame should pass validation."""
        timestamp = time.time()
        center_freq = 915e6
        sample_rate = 10e6
        num_samples = 256

        # Pack header (20 bytes)
        header = struct.pack('<dffHH', timestamp, center_freq, sample_rate, num_samples, 0)

        # Create payload (interleaved float32)
        i_samples = np.random.randn(num_samples).astype(np.float32)
        q_samples = np.random.randn(num_samples).astype(np.float32)
        interleaved = np.empty(num_samples * 2, dtype=np.float32)
        interleaved[0::2] = i_samples
        interleaved[1::2] = q_samples

        frame = header + interleaved.tobytes()

        errors = validate_iq_frame(frame)
        assert errors == [], f"Valid frame should have no errors: {errors}"

    def test_iq_header_size(self):
        """IQ header must be exactly 20 bytes."""
        assert IQ_HEADER_SIZE == 20, "IQ header must be 20 bytes per contract"

    def test_iq_payload_float32(self):
        """IQ payload must be float32, not int16."""
        timestamp = time.time()
        center_freq = 915e6
        sample_rate = 10e6
        num_samples = 256

        header = struct.pack('<dffHH', timestamp, center_freq, sample_rate, num_samples, 0)

        # Correct: float32 payload (8 bytes per sample pair)
        payload_float32 = np.zeros(num_samples * 2, dtype=np.float32).tobytes()
        frame_correct = header + payload_float32

        errors = validate_iq_frame(frame_correct)
        assert errors == [], "float32 payload should be valid"

        # Wrong: int16 payload would be wrong size (4 bytes per sample pair)
        payload_int16 = np.zeros(num_samples * 2, dtype=np.int16).tobytes()
        frame_wrong = header + payload_int16

        errors = validate_iq_frame(frame_wrong)
        assert any("payload size mismatch" in e for e in errors), "int16 payload should fail validation"

    def test_parse_iq_frame(self):
        """Parse should extract correct values."""
        timestamp = 1234567890.123
        center_freq = 915e6
        sample_rate = 10e6
        num_samples = 128

        header = struct.pack('<dffHH', timestamp, center_freq, sample_rate, num_samples, 0)

        i_samples = np.sin(np.linspace(0, 2*np.pi, num_samples)).astype(np.float32)
        q_samples = np.cos(np.linspace(0, 2*np.pi, num_samples)).astype(np.float32)
        interleaved = np.empty(num_samples * 2, dtype=np.float32)
        interleaved[0::2] = i_samples
        interleaved[1::2] = q_samples

        frame = header + interleaved.tobytes()

        result = parse_iq_frame(frame)
        assert result is not None
        assert abs(result["timestamp"] - timestamp) < 0.001
        assert result["center_freq_hz"] == center_freq
        assert result["sample_rate_hz"] == sample_rate
        assert result["num_samples"] == num_samples
        assert np.allclose(result["i_samples"], i_samples)
        assert np.allclose(result["q_samples"], q_samples)


# ============================================================================
# BLE Packet Tests
# ============================================================================

class TestBLEPacketContract:
    """Test BLE packet format compliance."""

    def test_valid_ble_packet(self):
        """Valid BLE packet should pass validation."""
        packet = {
            "type": "ble",
            "timestamp": time.time(),
            "packet_id": 1,
            "channel": 37,
            "access_address": "0x8E89BED6",
            "pdu_type": "ADV_IND",
            "payload_hex": "0201060303AAFE",
            "crc_valid": True,
            "rssi_dbm": -65.0,
        }

        errors = validate_ble_packet(packet)
        assert errors == [], f"Valid packet should have no errors: {errors}"

    def test_ble_invalid_channel(self):
        """BLE channel must be 37, 38, or 39."""
        packet = {
            "type": "ble",
            "timestamp": time.time(),
            "packet_id": 1,
            "channel": 36,  # Invalid - must be 37, 38, or 39
            "access_address": "0x8E89BED6",
            "pdu_type": "ADV_IND",
            "payload_hex": "0201060303AAFE",
            "crc_valid": True,
            "rssi_dbm": -65.0,
        }

        errors = validate_ble_packet(packet)
        assert any("channel" in e for e in errors)


# ============================================================================
# LoRa Frame Tests
# ============================================================================

class TestLoRaFrameContract:
    """Test LoRa frame format compliance."""

    def test_valid_lora_frame(self):
        """Valid LoRa frame should pass validation."""
        frame = {
            "type": "lora",
            "timestamp": time.time(),
            "frame_id": 1,
            "center_freq_hz": 915e6,
            "spreading_factor": 7,
            "bandwidth_hz": 125000,
            "coding_rate": "4/5",
            "payload_hex": "48656C6C6F",
            "crc_valid": True,
            "snr_db": 8.5,
            "rssi_dbm": -95.0,
        }

        errors = validate_lora_frame(frame)
        assert errors == [], f"Valid frame should have no errors: {errors}"

    def test_lora_invalid_spreading_factor(self):
        """LoRa SF must be 7-12."""
        frame = {
            "type": "lora",
            "timestamp": time.time(),
            "frame_id": 1,
            "center_freq_hz": 915e6,
            "spreading_factor": 6,  # Invalid - must be 7-12
            "bandwidth_hz": 125000,
            "coding_rate": "4/5",
            "payload_hex": "48656C6C6F",
            "crc_valid": True,
            "snr_db": 8.5,
            "rssi_dbm": -95.0,
        }

        errors = validate_lora_frame(frame)
        assert any("spreading_factor" in e for e in errors)

    def test_lora_invalid_bandwidth(self):
        """LoRa bandwidth must be 125k, 250k, or 500k."""
        frame = {
            "type": "lora",
            "timestamp": time.time(),
            "frame_id": 1,
            "center_freq_hz": 915e6,
            "spreading_factor": 7,
            "bandwidth_hz": 100000,  # Invalid
            "coding_rate": "4/5",
            "payload_hex": "48656C6C6F",
            "crc_valid": True,
            "snr_db": 8.5,
            "rssi_dbm": -95.0,
        }

        errors = validate_lora_frame(frame)
        assert any("bandwidth_hz" in e for e in errors)

    def test_lora_invalid_coding_rate(self):
        """LoRa coding rate must be 4/5, 4/6, 4/7, or 4/8."""
        frame = {
            "type": "lora",
            "timestamp": time.time(),
            "frame_id": 1,
            "center_freq_hz": 915e6,
            "spreading_factor": 7,
            "bandwidth_hz": 125000,
            "coding_rate": "4/9",  # Invalid
            "payload_hex": "48656C6C6F",
            "crc_valid": True,
            "snr_db": 8.5,
            "rssi_dbm": -95.0,
        }

        errors = validate_lora_frame(frame)
        assert any("coding_rate" in e for e in errors)


# ============================================================================
# Status Response Tests
# ============================================================================

class TestStatusResponseContract:
    """Test status REST response format compliance."""

    def test_valid_status_response(self):
        """Valid status response should pass validation."""
        status = {
            "state": "running",
            "uptime_seconds": 3600.5,
            "samples_processed": 1000000000,
            "detections_count": 150,
            "current_throughput_msps": 9.8,
            "gpu_memory_used_gb": 2.5,
            "buffer_fill_level": 0.75,
            "processing_latency_ms": 12.3,
            "consecutive_errors": 0,
            "dropped_samples": 0,
            "sdr_throttled": False,
            "gpu_utilization_percent": 85.0,
        }

        errors = validate_status_response(status)
        assert errors == [], f"Valid status should have no errors: {errors}"

    def test_status_invalid_state(self):
        """Status state must be valid enum value."""
        status = {
            "state": "starting",  # Invalid - not in enum
            "uptime_seconds": 0,
            "samples_processed": 0,
            "detections_count": 0,
            "current_throughput_msps": 0,
            "gpu_memory_used_gb": 0,
            "buffer_fill_level": 0,
            "processing_latency_ms": 0,
            "consecutive_errors": 0,
            "dropped_samples": 0,
            "sdr_throttled": False,
            "gpu_utilization_percent": 0,
        }

        errors = validate_status_response(status)
        assert any("state" in e for e in errors)

    def test_status_buffer_fill_range(self):
        """Buffer fill level must be 0.0-1.0."""
        status = {
            "state": "running",
            "uptime_seconds": 100,
            "samples_processed": 1000,
            "detections_count": 5,
            "current_throughput_msps": 10.0,
            "gpu_memory_used_gb": 1.0,
            "buffer_fill_level": 1.5,  # Invalid - must be 0-1
            "processing_latency_ms": 10.0,
            "consecutive_errors": 0,
            "dropped_samples": 0,
            "sdr_throttled": False,
            "gpu_utilization_percent": 50.0,
        }

        errors = validate_status_response(status)
        assert any("buffer_fill_level" in e for e in errors)


# ============================================================================
# Integration with Frontend Adapter
# ============================================================================

class TestFrontendAdapterIntegration:
    """Test that frontend adapter produces compliant output."""

    def test_detection_adapter_output(self):
        """FrontendMessageFormatter detection output should be compliant."""
        try:
            from rf_forensics.api.frontend_adapter import FrontendMessageFormatter

            formatter = FrontendMessageFormatter()

            # Create test detection
            detection = {
                "detection_id": 1,
                "center_freq_hz": 915e6,
                "bandwidth_hz": 500000,
                "peak_power_db": -50.0,
                "snr_db": 25.0,
                "modulation_type": "LoRa",
                "modulation_confidence": 0.85,
            }

            event = formatter.format_detection(detection)

            errors = validate_detection_event(event)
            assert errors == [], f"Adapter output should be compliant: {errors}"

        except ImportError:
            pytest.skip("Frontend adapter not available")

    def test_clusters_adapter_output(self):
        """FrontendMessageFormatter clusters output should be compliant."""
        try:
            from rf_forensics.api.frontend_adapter import FrontendMessageFormatter

            formatter = FrontendMessageFormatter()

            clusters = [
                {
                    "cluster_id": 0,
                    "size": 10,
                    "dominant_frequency_hz": 915e6,
                    "avg_snr_db": 20.0,
                }
            ]

            event = formatter.format_clusters(clusters)

            errors = validate_clusters_event(event)
            assert errors == [], f"Adapter output should be compliant: {errors}"

        except ImportError:
            pytest.skip("Frontend adapter not available")
