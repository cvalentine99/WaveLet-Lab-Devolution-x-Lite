# Backend API Specification

Backend is ready for frontend integration. All APIs aligned with your BACKEND_INTEGRATION_GUIDE.md.

---

## Connection Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| REST API | `http://localhost:8000` | Control & config |
| WebSocket | `ws://localhost:8765/ws/spectrum` | Binary PSD @ 30Hz |
| WebSocket | `ws://localhost:8765/ws/detections` | JSON detection events |
| WebSocket | `ws://localhost:8765/ws/clusters` | JSON cluster updates @ 1Hz |
| WebSocket | `ws://localhost:8765/ws/demodulation` | JSON LoRa/BLE packets |

---

## REST API

### POST /api/start
```json
{ "success": true }
```

### POST /api/stop
```json
{ "success": true }
```

### GET /api/status
```json
{
  "state": "idle" | "configuring" | "running" | "paused" | "error",
  "uptime_seconds": 123.45,
  "samples_processed": 1000000,
  "detections_count": 42,
  "current_throughput_msps": 10.5,
  "gpu_memory_used_gb": 2.1,
  "buffer_fill_level": 0.75,
  "processing_latency_ms": 12.3
}
```

### GET /api/config
### POST /api/config
```json
{
  "sdr": {
    "center_freq_hz": 915000000,
    "sample_rate_hz": 10000000,
    "gain_db": 40.0
  },
  "pipeline": {
    "fft_size": 1024,
    "window_type": "hann",
    "overlap": 0.5
  },
  "cfar": {
    "pfa": 0.000001,
    "ref_cells": 32,
    "guard_cells": 4
  },
  "clustering": {
    "enabled": true,
    "eps": 0.5,
    "min_samples": 5
  }
}
```

**Note:** `overlap` is 0.0-1.0, not percent. `ref_cells` not `num_reference_cells`.

---

## SigMF Recording API

Records IQ samples to SigMF format (`.sigmf-meta` + `.sigmf-data`).

### POST /api/recordings/start
```json
// Request
{
  "name": "915 MHz LoRa Capture",
  "description": "ISM band monitoring session",
  "duration_seconds": 60  // Optional auto-stop
}

// Response
{
  "recording_id": "rec_abc123",
  "status": "recording"
}
```

### POST /api/recordings/stop
```json
// Request
{ "recording_id": "rec_abc123" }

// Response
{
  "recording_id": "rec_abc123",
  "status": "stopped",
  "file_size_bytes": 80000000,
  "num_samples": 10000000
}
```

### GET /api/recordings
```json
{
  "recordings": [
    {
      "id": "rec_abc123",
      "name": "915 MHz LoRa Capture",
      "description": "ISM band monitoring session",
      "center_freq_hz": 915000000,
      "sample_rate_hz": 10000000,
      "num_samples": 10000000,
      "duration_seconds": 1.0,
      "file_size_bytes": 80000000,
      "created_at": "2025-12-01T12:00:00Z",
      "status": "stopped",
      "sigmf_meta_path": "/data/recordings/rec_abc123.sigmf-meta",
      "sigmf_data_path": "/data/recordings/rec_abc123.sigmf-data"
    }
  ]
}
```

### GET /api/recordings/{recording_id}
Returns single recording metadata.

### GET /api/recordings/{recording_id}/download
Returns ZIP file containing `.sigmf-meta` + `.sigmf-data`.

### DELETE /api/recordings/{recording_id}
```json
{ "success": true }
```

### SigMF Format Details
- **Datatype:** `cf32_le` (complex float32, little-endian)
- **Bytes per sample:** 8 (4 I + 4 Q interleaved)
- **Files:** `{recording_id}.sigmf-meta` (JSON) + `{recording_id}.sigmf-data` (binary)

### POST /api/detections/export?format=json
Returns downloadable JSON file with all detections.

---

## Binary Spectrum Protocol (ws://localhost:8765/ws/spectrum)

```
Header (16 bytes, little-endian):
┌─────────────┬─────────────┬──────────┬──────────┬──────────┐
│ timestamp   │ center_freq │ span     │ num_bins │ reserved │
│ float64     │ float32     │ float32  │ uint16   │ uint16   │
│ 8 bytes     │ 4 bytes     │ 4 bytes  │ 2 bytes  │ 2 bytes  │
└─────────────┴─────────────┴──────────┴──────────┴──────────┘

Payload: uint8[num_bins]
  - 0 = -120 dBm
  - 255 = -20 dBm
  - Linear interpolation between
```

---

## WebSocket JSON Events

### /ws/detections
```json
{
  "type": "detection",
  "timestamp": 1699123456.789,
  "data": {
    "detection_id": 1,
    "center_freq_hz": 915000000,
    "bandwidth_hz": 125000,
    "bandwidth_3db_hz": 100000,
    "bandwidth_6db_hz": 150000,
    "peak_power_db": -45.2,
    "snr_db": 15.3,
    "start_bin": 450,
    "end_bin": 550
  }
}
```

### /ws/clusters
```json
{
  "type": "clusters",
  "timestamp": 1699123456.789,
  "data": [
    {
      "cluster_id": 1,
      "size": 15,
      "dominant_frequency_hz": 915000000,
      "avg_snr_db": 12.5,
      "label": "LoRa"
    }
  ]
}
```

### /ws/demodulation (LoRa)
```json
{
  "type": "lora_frame",
  "timestamp": 1699123456.789,
  "data": {
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
}
```

### /ws/demodulation (BLE)
```json
{
  "type": "ble_packet",
  "timestamp": 1699123456.789,
  "data": {
    "packet_id": 1,
    "channel": 37,
    "access_address": "8E89BED6",
    "pdu_type": "ADV_IND",
    "payload_hex": "0201061aff4c00...",
    "crc_valid": true,
    "rssi_dbm": -72.0
  }
}
```

**PDU Types:** `ADV_IND`, `ADV_DIRECT_IND`, `ADV_NONCONN_IND`, `SCAN_REQ`, `SCAN_RSP`, `CONNECT_IND`, `ADV_SCAN_IND`

---

## SDR Device Management API (uSDR DevBoard)

New endpoints for controlling the uSDR DevBoard hardware.

### GET /api/sdr/devices
Discover available SDR devices.
```json
{
  "devices": [
    {
      "id": "usb://0",
      "model": "uSDR DevBoard",
      "serial": "ABC123",
      "status": "available"
    }
  ]
}
```

### POST /api/sdr/connect
```json
// Request
{ "device_id": "usb://0" }

// Response
{ "success": true, "device_id": "usb://0" }
```

### POST /api/sdr/disconnect
```json
{ "success": true }
```

### GET /api/sdr/status
Real-time SDR hardware status.
```json
{
  "connected": true,
  "device_id": "usb://0",
  "temperature_c": 42.5,
  "actual_freq_hz": 915000000,
  "actual_sample_rate_hz": 10000000,
  "actual_bandwidth_hz": 10000000,
  "rx_path": "LNAL",
  "streaming": true
}
```

### GET /api/sdr/config
Get full SDR configuration.
```json
{
  "device_type": "usdr",
  "center_freq_hz": 915000000,
  "sample_rate_hz": 10000000,
  "bandwidth_hz": 10000000,
  "gain": {
    "lna_db": 15,
    "tia_db": 9,
    "pga_db": 12,
    "total_db": 36
  },
  "rx_path": "LNAL",
  "devboard": {
    "lna_enable": true,
    "attenuator_db": 0,
    "vctcxo_dac": 32768
  }
}
```

### POST /api/sdr/config
Update SDR configuration (same schema as GET response).

### POST /api/sdr/frequency
Quick frequency change.
```json
// Request query param: freq_hz=915000000

// Response
{ "success": true, "freq_hz": 915000000 }
```

### POST /api/sdr/gain
Quick gain adjustment.
```json
// Request query params: lna_db=15&tia_db=9&pga_db=12

// Response
{
  "success": true,
  "gain": {
    "lna_db": 15,
    "tia_db": 9,
    "pga_db": 12,
    "total_db": 36
  }
}
```

### POST /api/sdr/rx_path
Set RX antenna path.
```json
// Request query param: path=LNAL

// Response
{ "success": true, "rx_path": "LNAL" }
```

### RX Path Options
| Path | Frequency Range | Description |
|------|-----------------|-------------|
| LNAH | 1.5 - 3.8 GHz | High band |
| LNAL | 0.3 - 2.2 GHz | Low band (default) |
| LNAW | Wideband | Full range |

### Gain Stages (LMS6002D)
| Stage | Range | Description |
|-------|-------|-------------|
| LNA | 0-30 dB | Low Noise Amplifier (1 dB steps) |
| TIA | 0, 3, 9, 12 dB | Trans-Impedance Amplifier |
| PGA | 0-32 dB | Programmable Gain Amplifier |

---

## Duplexer Band Selection API

The DevBoard has a duplexer with configurable filter paths for different frequency bands.

### GET /api/sdr/bands
List available duplexer bands. Optional query: `?category=cellular`
```json
{
  "bands": [
    {
      "name": "band2",
      "aliases": ["pcs", "gsm1900"],
      "category": "cellular",
      "freq_range_mhz": [1850, 1990],
      "description": "PCS / GSM 1900",
      "pa_enable": true,
      "lna_enable": true,
      "trx_mode": "TRX_BAND2",
      "rx_filter": "RX_LPF1200",
      "tx_filter": "TX_LPF400"
    }
  ],
  "categories": ["cellular", "tx_only", "rx_only", "tdd"]
}
```

### GET /api/sdr/bands/{band_name}
Get details for a specific band (accepts name or alias).

### POST /api/sdr/bands
Set duplexer band configuration.
```json
// Request
{ "band": "gsm900" }

// Response
{
  "success": true,
  "band": "band8",
  "category": "cellular",
  "description": "GSM 900",
  "freq_range_mhz": [880, 960],
  "pa_enable": true,
  "lna_enable": true
}
```

### Available Duplexer Bands

#### Cellular FDD Bands (Duplexer Active)
| Band | Aliases | Frequency Range | Description |
|------|---------|-----------------|-------------|
| band2 | pcs, gsm1900 | 1850-1990 MHz | PCS / GSM 1900 |
| band3 | dcs, gsm1800 | 1710-1880 MHz | DCS / GSM 1800 |
| band5 | gsm850 | 824-894 MHz | GSM 850 |
| band7 | imte | 2500-2690 MHz | IMT-E / LTE Band 7 |
| band8 | gsm900 | 880-960 MHz | GSM 900 |

#### TX-Only Paths (Duplexer Bypass)
| Band | Frequency Range | TX Filter |
|------|-----------------|-----------|
| txlpf400 | 0-400 MHz | TX_LPF400 |
| txlpf1200 | 0-1200 MHz | TX_LPF1200 |
| txlpf2100 | 0-2100 MHz | TX_LPF2100 |
| txlpf4200 | 0-4200 MHz | TX_BYPASS |

#### RX-Only Paths (Duplexer Bypass)
| Band | Frequency Range | RX Filter |
|------|-----------------|-----------|
| rxlpf1200 | 0-1200 MHz | RX_LPF1200 |
| rxlpf2100 | 0-2100 MHz | RX_LPF2100 |
| rxbpf2100_3000 | 2100-3000 MHz | RX_BPF2100_3000 |
| rxbpf3000_4200 | 3000-4200 MHz | RX_BPF3000_4200 |

#### TDD / Half-Duplex Modes
| Band | Frequency Range | Description |
|------|-----------------|-------------|
| trx0_400 | 0-400 MHz | TDD 0-400 MHz |
| trx400_1200 | 400-1200 MHz | TDD 400-1200 MHz |
| trx1200_2100 | 1200-2100 MHz | TDD 1.2-2.1 GHz |
| trx2100_3000 | 2100-3000 MHz | TDD 2.1-3.0 GHz |
| trx3000_4200 | 3000-4200 MHz | TDD 3.0-4.2 GHz |

### Frontend UI: Band Selection Dialog
```
┌─────────────────────────────────────────────────────────────┐
│ Duplexer Band Selection                               [X]   │
├─────────────────────────────────────────────────────────────┤
│ Category: [Cellular ▼]                                      │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ○ Band 2 (PCS/GSM 1900)      1850-1990 MHz             │ │
│ │ ○ Band 3 (DCS/GSM 1800)      1710-1880 MHz             │ │
│ │ ○ Band 5 (GSM 850)           824-894 MHz               │ │
│ │ ○ Band 7 (IMT-E)             2500-2690 MHz             │ │
│ │ ● Band 8 (GSM 900)           880-960 MHz    ◄ Active   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Selected: band8 (GSM 900)                                   │
│ PA: Enabled  |  LNA: Enabled                                │
│                                                             │
│                              [Cancel]  [Apply]              │
└─────────────────────────────────────────────────────────────┘
```

---

## CORS

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: *
Access-Control-Allow-Headers: *
```

---

## Quick Start

```bash
# Start backend
cd /home/cvalentine/GPU\ Forensics/rf_forensics
python -m uvicorn api.rest_api:app --host 0.0.0.0 --port 8000

# Test system
curl http://localhost:8000/api/status
curl http://localhost:8000/api/config
curl -X POST http://localhost:8000/api/start

# Test recordings
curl -X POST http://localhost:8000/api/recordings/start \
  -H "Content-Type: application/json" \
  -d '{"name": "Test", "description": "915 MHz test"}'

curl -X POST http://localhost:8000/api/recordings/stop \
  -H "Content-Type: application/json" \
  -d '{"recording_id": "rec_abc123"}'

curl http://localhost:8000/api/recordings
curl http://localhost:8000/api/recordings/rec_abc123/download -o recording.zip

# Test SDR
curl http://localhost:8000/api/sdr/devices
curl http://localhost:8000/api/sdr/status

curl -X POST http://localhost:8000/api/sdr/connect \
  -H "Content-Type: application/json" \
  -d '{"device_id": "usb://0"}'

curl -X POST "http://localhost:8000/api/sdr/frequency?freq_hz=915000000"
curl -X POST "http://localhost:8000/api/sdr/gain?lna_db=15&tia_db=9&pga_db=12"
curl -X POST "http://localhost:8000/api/sdr/rx_path?path=LNAL"

# Test duplexer bands
curl http://localhost:8000/api/sdr/bands
curl "http://localhost:8000/api/sdr/bands?category=cellular"
curl http://localhost:8000/api/sdr/bands/gsm900

curl -X POST http://localhost:8000/api/sdr/bands \
  -H "Content-Type: application/json" \
  -d '{"band": "band8"}'
```
