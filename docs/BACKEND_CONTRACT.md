# Backend Integration Contract

**Frontend Version:** 2025-12-03
**Purpose:** Complete specification for backend ↔ frontend data exchange

---

## 1. REST API Endpoints

### System Control

| Endpoint | Method | Request Body | Response | Notes |
|----------|--------|--------------|----------|-------|
| `/api/start` | POST | None | `{ success: bool }` | Start acquisition |
| `/api/stop` | POST | None | `{ success: bool }` | Stop acquisition |
| `/api/pause` | POST | None | `{ success: bool }` | Pause streaming |
| `/api/resume` | POST | None | `{ success: bool }` | Resume streaming |

### System Status

**GET `/api/status`**

```json
{
  "state": "idle" | "configuring" | "running" | "paused" | "error",
  "uptime_seconds": 3600,
  "samples_processed": 1000000000,
  "detections_count": 1542,
  "current_throughput_msps": 40.5,
  "gpu_memory_used_gb": 8.2,
  "buffer_fill_level": 0.35,
  "processing_latency_ms": 4.2
}
```

### Configuration

**GET `/api/config`**

```json
{
  "sdr": {
    "device_type": "usdr",
    "center_freq_hz": 915000000,
    "sample_rate_hz": 10000000,
    "bandwidth_hz": 10000000,
    "gain": {
      "lna_db": 15,
      "tia_db": 9,
      "pga_db": 12
    },
    "rx_path": "LNAL",
    "devboard": {
      "lna_enable": true,
      "pa_enable": false,
      "attenuator_db": 0,
      "vctcxo_dac": 32768,
      "gps_enable": false,
      "osc_enable": true,
      "loopback_enable": false,
      "uart_enable": false
    }
  },
  "pipeline": {
    "fft_size": 2048,
    "window_type": "hann",
    "overlap": 0.5,
    "averaging_count": 4
  },
  "cfar": {
    "ref_cells": 32,
    "guard_cells": 4,
    "pfa": 0.000001,
    "variant": "CA"
  }
}
```

**POST `/api/config`** - Partial update, same schema

### SDR Device Management

**GET `/api/sdr/devices`**

```json
{
  "devices": [
    {
      "id": "usdr:0",
      "model": "uSDR DevBoard",
      "serial": "ABC123",
      "status": "available" | "connected" | "in_use"
    }
  ]
}
```

**POST `/api/sdr/connect`**

```json
// Request
{ "device_id": "usdr:0" }

// Response
{ "success": true, "device_id": "usdr:0" }
```

**POST `/api/sdr/disconnect`**

```json
{ "success": true }
```

**GET `/api/sdr/status`**

```json
{
  "connected": true,
  "device_id": "usdr:0",
  "temperature_c": 45.2,
  "actual_freq_hz": 915000000,
  "actual_sample_rate_hz": 10000000,
  "actual_bandwidth_hz": 10000000,
  "rx_path": "LNAL",
  "streaming": true
}
```

**GET `/api/sdr/metrics`**

```json
{
  "overflow": {
    "total": 0,
    "rate_per_sec": 0.0,
    "last_timestamp": null
  },
  "samples": {
    "total_received": 500000000,
    "total_dropped": 1200,
    "drop_rate_percent": 0.00024
  },
  "hardware": {
    "temperature_c": 45.2,
    "pll_locked": true,
    "actual_sample_rate_hz": 10000000,
    "actual_freq_hz": 915000000
  },
  "streaming": {
    "uptime_seconds": 3600,
    "reconnect_count": 0,
    "last_error": null
  },
  "backpressure": {
    "events": 0,
    "buffer_fill_percent": 35.2
  }
}
```

**GET `/api/sdr/health`**

```json
{
  "status": "healthy" | "degraded" | "unhealthy",
  "connected": {},
  "streaming": true,
  "warnings": ["High temperature detected"],
  "metrics_summary": {
    "overflow_rate": 0.0,
    "drop_rate_percent": 0.00024,
    "buffer_fill_percent": 35.2,
    "temperature_c": 45.2,
    "uptime_seconds": 3600
  }
}
```

**GET `/api/sdr/capabilities`**

```json
{
  "freq_range_hz": { "min": 1000000, "max": 3800000000 },
  "sample_rate_range_hz": { "min": 1000000, "max": 65000000 },
  "bandwidth_range_hz": { "min": 500000, "max": 40000000 },
  "gain_range_db": { "min": 0, "max": 74 },
  "rx_paths": ["LNAH", "LNAL", "LNAW"],
  "supported_formats": ["cf32_le", "cs16_le"],
  "device": {
    "id": "usdr:0",
    "model": "uSDR DevBoard",
    "serial": "ABC123",
    "firmware_version": "1.2.3"
  },
  "features": {
    "supports_gpudirect": true,
    "supports_timestamps": true,
    "max_channels": 2
  }
}
```

**POST `/api/sdr/config`** - Direct SDR config update

```json
// Request
{
  "center_freq_hz": 915000000,
  "sample_rate_hz": 10000000,
  "bandwidth_hz": 10000000
}
```

**POST `/api/sdr/gain`**

```json
// Request
{
  "lna_db": 15,
  "tia_db": 9,
  "pga_db": 12
}
```

**POST `/api/sdr/rx_path`**

```json
// Request
{ "rx_path": "LNAL" }  // "LNAH" | "LNAL" | "LNAW"
```

**POST `/api/sdr/frequency`**

```json
// Request
{ "center_freq_hz": 915000000 }
```

### Duplexer Bands

**GET `/api/sdr/bands`**

```json
{
  "bands": [
    {
      "name": "B1_RX",
      "category": "cellular" | "tx_only" | "rx_only" | "tdd",
      "description": "LTE Band 1 Downlink",
      "freq_range_mhz": [2110, 2170],
      "pa_enable": false,
      "lna_enable": true
    }
  ]
}
```

**POST `/api/sdr/band`**

```json
// Request
{ "band": "B1_RX" }
```

### Recordings

**POST `/api/recordings/start`**

```json
// Request
{
  "name": "capture-2025-12-03",
  "description": "ISM band monitoring",
  "duration_seconds": 60  // optional, null = manual stop
}

// Response
{
  "recording_id": "rec-abc123",
  "started_at": "2025-12-03T10:00:00Z"
}
```

**POST `/api/recordings/stop`**

```json
// Request
{ "recording_id": "rec-abc123" }

// Response
{
  "recording_id": "rec-abc123",
  "file_size_bytes": 480000000,
  "duration_seconds": 60,
  "num_samples": 600000000
}
```

**GET `/api/recordings`**

```json
{
  "recordings": [
    {
      "id": "rec-abc123",
      "name": "capture-2025-12-03",
      "description": "ISM band monitoring",
      "center_freq_hz": 915000000,
      "sample_rate_hz": 10000000,
      "num_samples": 600000000,
      "duration_seconds": 60,
      "file_size_bytes": 480000000,
      "created_at": "2025-12-03T10:00:00Z",
      "status": "stopped",
      "sigmf_meta_path": "/recordings/capture.sigmf-meta",
      "sigmf_data_path": "/recordings/capture.sigmf-data"
    }
  ]
}
```

**GET `/api/recordings/{id}/download`** - Returns ZIP file

**DELETE `/api/recordings/{id}`**

### Detection Export

**POST `/api/detections/export?format=json`**

Returns: `application/json` blob of all detections

---

## 2. WebSocket Endpoints

### Spectrum Stream

**Endpoint:** `ws://host:8765/ws/spectrum`
**Protocol:** Binary

#### Header (20 bytes, little-endian)

| Offset | Type | Field | Description |
|--------|------|-------|-------------|
| 0 | float64 | timestamp | Unix timestamp (seconds) |
| 8 | float32 | center_freq | Center frequency (Hz) |
| 12 | float32 | span | Frequency span (Hz) |
| 16 | uint16 | num_bins | Number of FFT bins |
| 18 | uint16 | reserved | Padding |

#### Payload

- **Offset:** 20 bytes
- **Format:** `uint8[num_bins]`
- **Conversion:** `dBm = -120 + (value / 255) * 100`
- **Range:** 0 → -120 dBm, 255 → -20 dBm

---

### Detection Stream

**Endpoint:** `ws://host:8765/ws/detections`
**Protocol:** JSON

```json
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

    // AMC Classification (from ML model)
    "modulation_type": "LoRa",
    "modulation_confidence": 0.92,
    "top_k_predictions": [
      { "modulation_type": "LoRa", "confidence": 0.92 },
      { "modulation_type": "FSK", "confidence": 0.05 },
      { "modulation_type": "GFSK", "confidence": 0.03 }
    ],

    // Tracking (from signal tracker)
    "track_id": "trk-abc123",
    "duty_cycle": 0.15,
    "frames_seen": 42,

    // Clustering & Anomaly (NEW - required for ML dashboard)
    "cluster_id": 3,
    "anomaly_score": 0.12,
    "symbol_rate": 50000
  }
}
```

**CRITICAL FIELDS FOR ML FEATURES:**
- `cluster_id` - Links detection to DBSCAN cluster
- `anomaly_score` - Autoencoder anomaly score (0.0-1.0, >0.5 = anomalous)
- `modulation_type` / `modulation_confidence` / `top_k_predictions` - AMC results

---

### Cluster Stream

**Endpoint:** `ws://host:8765/ws/clusters`
**Protocol:** JSON

```json
{
  "type": "clusters",
  "timestamp": 1701590400.123,
  "data": [
    {
      "cluster_id": 3,
      "size": 42,
      "center_freq_hz": 915125000,
      "dominant_frequency_hz": 915125000,  // Legacy alias
      "freq_range_hz": [915000000, 915250000],
      "avg_snr_db": 15.2,
      "avg_power_db": -48.5,
      "avg_bandwidth_hz": 125000,
      "detection_count": 156,
      "label": "LoRa Gateway",

      // ML Classification Hint (NEW - for dashboard)
      "signal_type_hint": "LoRa",

      // Tracking metrics
      "avg_duty_cycle": 0.12,
      "unique_tracks": 3,
      "avg_bw_3db_ratio": 0.8
    }
  ]
}
```

**CRITICAL FIELD:**
- `signal_type_hint` - Backend's classification guess for the cluster

---

### IQ Stream (Constellation)

**Endpoint:** `ws://host:8765/ws/iq`
**Protocol:** Binary

#### Header (20 bytes, little-endian)

| Offset | Type | Field |
|--------|------|-------|
| 0 | float64 | timestamp |
| 8 | float32 | center_freq_hz |
| 12 | float32 | sample_rate_hz |
| 16 | uint32 | num_samples |

#### Payload

- **Format:** Interleaved float32 I/Q pairs
- **Size:** `num_samples * 8` bytes

---

### LoRa Decoder Stream

**Endpoint:** `ws://host:8765/ws/lora`
**Protocol:** JSON

```json
{
  "type": "lora",
  "timestamp": 1701590400.123,
  "frame_id": 1,
  "center_freq_hz": 915000000,
  "spreading_factor": 7,
  "bandwidth_hz": 125000,
  "coding_rate": "4/5",
  "payload_hex": "48656c6c6f",
  "crc_valid": true,
  "snr_db": 12.5,
  "rssi_dbm": -85.0
}
```

---

### BLE Decoder Stream (MISSING - needs implementation)

**Endpoint:** `ws://host:8765/ws/ble`
**Protocol:** JSON

```json
{
  "type": "ble",
  "timestamp": 1701590400.123,
  "packet_id": 1,
  "channel": 37,
  "access_address": "8E89BED6",
  "pdu_type": "ADV_IND",
  "payload_hex": "0201061aff4c00...",
  "crc_valid": true,
  "rssi_dbm": -72.0
}
```

---

## 3. Field Name Conventions

| Frontend (TypeScript) | Backend (Python) | Notes |
|-----------------------|------------------|-------|
| `centerFreqHz` | `center_freq_hz` | All frequencies in Hz |
| `sampleRateHz` | `sample_rate_hz` | |
| `bandwidthHz` | `bandwidth_hz` | |
| `peakPowerDb` | `peak_power_db` | dBm |
| `snrDb` | `snr_db` | dB |
| `lnaDb` | `lna_db` | 0-30 dB |
| `tiaDb` | `tia_db` | 0-12 dB (steps of 3) |
| `pgaDb` | `pga_db` | 0-32 dB |
| `modulationType` | `modulation_type` | String |
| `clusterId` | `cluster_id` | Integer |
| `anomalyScore` | `anomaly_score` | 0.0-1.0 |
| `signalTypeHint` | `signal_type_hint` | String |

---

## 4. What Frontend NEEDS from Backend

### Currently Missing/Broken

| Feature | What Backend Must Send | Priority |
|---------|----------------------|----------|
| `cluster_id` in detections | Add to detection events when clustering assigns | HIGH |
| `anomaly_score` in detections | Add autoencoder output (0.0-1.0) | HIGH |
| `signal_type_hint` in clusters | Add classification hint per cluster | MEDIUM |
| `/ws/ble` endpoint | Implement BLE decoder stream | MEDIUM |
| GPU utilization % | Add to `/api/status` response | LOW |

### Frontend Expectations

1. **Timestamps:** Unix seconds (float64), frontend multiplies by 1000 for JS Date
2. **Frequencies:** Always Hz (never MHz)
3. **Power:** Always dBm or dB (never linear)
4. **Anomaly scores:** 0.0 = normal, 1.0 = highly anomalous
5. **Confidence:** 0.0-1.0 (frontend displays as percentage)

---

## 5. Error Handling

All REST endpoints should return errors as:

```json
{
  "error": true,
  "message": "Human-readable error message",
  "code": "ERROR_CODE"
}
```

HTTP status codes:
- `400` - Bad request (invalid parameters)
- `404` - Not found (device/recording doesn't exist)
- `409` - Conflict (already running, already connected)
- `500` - Internal server error
- `503` - Service unavailable (SDR not connected)

---

## 6. Connection Info

Frontend expects backend at:
- **REST API:** `http://localhost:8000` (env: `VITE_BACKEND_URL`)
- **WebSocket:** `ws://localhost:8765` (env: `VITE_BACKEND_WS_URL`)

Both are configurable via environment variables.
