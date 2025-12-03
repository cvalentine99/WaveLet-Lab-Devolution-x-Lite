# GPU RF Forensics Engine - Backend Integration Spec

**For: Devolution(x) Spectrum Detect Frontend**
**Backend Version:** 1.0
**Updated:** December 1, 2025

---

## Overview

This document specifies how the GPU RF Forensics Engine backend integrates with the Devolution(x) Spectrum Detect frontend. The backend provides:

1. **REST API** (FastAPI on port 8000) - Configuration, control, status
2. **Socket.IO Server** (port 8765) - Real-time streaming data

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start servers (simulation mode)
python scripts/run_socketio_server.py --simulate

# Start servers (real hardware)
python scripts/run_socketio_server.py
```

**Endpoints:**
- REST API: `http://localhost:8000`
- Socket.IO: `http://localhost:8765`

---

## Socket.IO Integration

The backend uses Socket.IO to stream real-time data to the frontend. Connect using:

```typescript
import { io } from 'socket.io-client'

// Spectrum data
const spectrumSocket = io('http://localhost:8765/spectrum')
spectrumSocket.on('spectrum:measurement', (data) => {
  // Handle spectrum data
})

// Detection events
const detectSocket = io('http://localhost:8765/detections')
detectSocket.on('detection:new', (detection) => {
  // Handle new detection
})
```

### Namespaces

| Namespace | Purpose | Events |
|-----------|---------|--------|
| `/spectrum` | Real-time spectrum data | `spectrum:measurement` |
| `/detections` | Threat detection events | `detection:new`, `detection:update`, `cluster:update` |
| `/analytics` | Performance metrics | `analytics:metrics` |
| `/hardware` | SDR device telemetry | `hardware:status` |

### Event Schemas

#### `spectrum:measurement`
```json
{
  "type": "spectrum",
  "timestamp": 1733058600.123,
  "centerFreqHz": 915000000,
  "sampleRateHz": 2048000,
  "fftSize": 2048,
  "magnitudeDb": [-90.5, -89.2, ...],
  "frequencyBins": [914000000, 914001000, ...],
  "deviceId": "usdr-0"
}
```

#### `detection:new`
```json
{
  "type": "detection",
  "timestamp": 1733058601.456,
  "id": "det_20251201_103001_456",
  "centerFreqHz": 915200000,
  "bandwidthHz": 125000,
  "peakPowerDb": -45.2,
  "snrDb": 18.5,
  "modulationType": "LoRa",
  "confidence": 0.92,
  "duration": 0.250,
  "severity": "high"
}
```

#### `detection:update`
```json
{
  "id": "det_20251201_103001_456",
  "status": "resolved",
  "resolution": "false_positive",
  "timestamp": 1733058700.123
}
```

#### `cluster:update`
```json
{
  "type": "cluster",
  "timestamp": 1733058602.789,
  "clusters": [{
    "id": "cluster_915mhz",
    "centerFreqHz": 915000000,
    "freqRangeHz": [914500000, 915500000],
    "signalCount": 12,
    "avgPowerDb": -52.3,
    "dominantModulation": "LoRa",
    "activity": 0.65
  }]
}
```

#### `analytics:metrics`
```json
{
  "cpu": 25.5,
  "memory": 45.2,
  "detectionRate": 1.5,
  "latency": 12.3,
  "samplesProcessed": 1048576,
  "throughputMsps": 10.24,
  "gpuUtilization": 55.0,
  "timestamp": 1733058603.000
}
```

#### `hardware:status`
```json
{
  "deviceId": "usdr-0",
  "connected": true,
  "temperature": 42.5,
  "sampleRate": 10000000,
  "centerFrequency": 915000000,
  "bufferUtilization": 0.25,
  "streaming": true,
  "rxPath": "LNAL",
  "totalGainDb": 36,
  "timestamp": 1733058604.000
}
```

---

## REST API Endpoints

All REST responses use **camelCase** field names for frontend compatibility.

### Health Check

**GET /health**
```json
{
  "status": "ok",
  "gpuAvailable": true,
  "gpuName": "NVIDIA RTX 4090",
  "backendVersion": "1.0.0",
  "timestamp": 1733058600.123
}
```

### SDR Configuration

**POST /api/sdr/config**
```json
// Request
{
  "centerFreqHz": 915000000,
  "sampleRateHz": 10000000,
  "gainDb": 40.0,
  "bandwidthHz": 10000000
}

// Response
{
  "success": true,
  "config": {
    "centerFreqHz": 915000000,
    "sampleRateHz": 10000000,
    "gainDb": 40.0,
    "bandwidthHz": 10000000
  }
}
```

**GET /api/sdr/devices**
```json
{
  "devices": [{
    "id": "usb://0",
    "model": "uSDR DevBoard",
    "serial": "ABC123",
    "status": "available"
  }]
}
```

**POST /api/sdr/connect**
```json
// Request
{ "deviceId": "usb://0" }

// Response
{ "success": true, "deviceId": "usb://0" }
```

**GET /api/sdr/status**
```json
{
  "connected": true,
  "deviceId": "usb://0",
  "temperatureC": 42.5,
  "actualFreqHz": 915000000,
  "actualSampleRateHz": 10000000,
  "actualBandwidthHz": 10000000,
  "rxPath": "LNAL",
  "streaming": true
}
```

**GET /api/sdr/bands**
```json
{
  "bands": [
    {
      "name": "band2",
      "aliases": ["pcs", "gsm1900"],
      "category": "cellular",
      "freqRangeMhz": [1850, 1990],
      "description": "PCS / GSM 1900"
    },
    ...
  ]
}
```

### Monitoring Control

**POST /api/monitoring/start**
```json
// Request (optional)
{ "preset": "wideband_survey" }

// Response
{
  "success": true,
  "message": "Monitoring started",
  "sessionId": "session_000001"
}
```

**POST /api/monitoring/stop**
```json
{
  "success": true,
  "message": "Monitoring stopped"
}
```

### Recording Control

**POST /api/recording/start**
```json
// Request
{
  "filename": "capture_915mhz",
  "format": "sigmf",
  "maxDurationSec": 300
}

// Response
{
  "success": true,
  "recordingId": "rec_20251201_103000",
  "filename": "capture_915mhz.sigmf"
}
```

**POST /api/recording/stop**
```json
{
  "success": true,
  "recordingId": "rec_20251201_103000",
  "filepath": "/recordings/capture_915mhz.sigmf",
  "durationSec": 45.2,
  "sizeMb": 128.5
}
```

---

## Frontend Connection Example

```typescript
// client/src/lib/rfBackend.ts

import { io, Socket } from 'socket.io-client'

const BACKEND_URL = 'http://localhost:8765'
const REST_URL = 'http://localhost:8000'

// Socket.IO connections
export const spectrumSocket: Socket = io(`${BACKEND_URL}/spectrum`)
export const detectionsSocket: Socket = io(`${BACKEND_URL}/detections`)
export const analyticsSocket: Socket = io(`${BACKEND_URL}/analytics`)
export const hardwareSocket: Socket = io(`${BACKEND_URL}/hardware`)

// Event handlers
spectrumSocket.on('spectrum:measurement', (data) => {
  // Update spectrum visualization
  updateWaterfall(data.magnitudeDb, data.frequencyBins)
})

detectionsSocket.on('detection:new', (detection) => {
  // Add to detection list, show alert
  addDetection(detection)
  if (detection.severity === 'high') {
    showAlert(detection)
  }
})

analyticsSocket.on('analytics:metrics', (metrics) => {
  // Update dashboard metrics
  updateDashboard(metrics)
})

hardwareSocket.on('hardware:status', (status) => {
  // Update hardware panel
  updateHardwareStatus(status)
})

// REST API client
export const api = {
  async getHealth() {
    const res = await fetch(`${REST_URL}/health`)
    return res.json()
  },

  async configureSDR(config: SDRConfig) {
    const res = await fetch(`${REST_URL}/api/sdr/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    })
    return res.json()
  },

  async startMonitoring(preset?: string) {
    const res = await fetch(`${REST_URL}/api/monitoring/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset })
    })
    return res.json()
  },

  async stopMonitoring() {
    const res = await fetch(`${REST_URL}/api/monitoring/stop`, {
      method: 'POST'
    })
    return res.json()
  }
}
```

---

## TypeScript Interfaces

```typescript
// Spectrum measurement
interface SpectrumMeasurement {
  type: 'spectrum'
  timestamp: number
  centerFreqHz: number
  sampleRateHz: number
  fftSize: number
  magnitudeDb: number[]
  frequencyBins: number[]
  deviceId: string
}

// Detection event
interface Detection {
  type: 'detection'
  timestamp: number
  id: string
  centerFreqHz: number
  bandwidthHz: number
  peakPowerDb: number
  snrDb: number
  modulationType: string
  confidence: number
  duration: number
  severity: 'low' | 'medium' | 'high'
}

// Hardware status
interface HardwareStatus {
  deviceId: string
  connected: boolean
  temperature: number
  sampleRate: number
  centerFrequency: number
  bufferUtilization: number
  streaming: boolean
  rxPath: string
  totalGainDb: number
  timestamp: number
}

// Analytics metrics
interface AnalyticsMetrics {
  cpu: number
  memory: number
  detectionRate: number
  latency: number
  samplesProcessed: number
  throughputMsps: number
  gpuUtilization: number
  timestamp: number
}

// SDR config
interface SDRConfig {
  centerFreqHz: number
  sampleRateHz: number
  gainDb: number
  bandwidthHz?: number
}
```

---

## Hardware: uSDR DevBoard

The backend supports the uSDR DevBoard (Wavelet Lab):

- **RFIC:** LMS6002D
- **Frequency:** 1 MHz – 3.8 GHz
- **Sample Rate:** up to 65 MSPS
- **Bandwidth:** 0.5–40 MHz
- **Channels:** 2x MIMO

### RX Path Options
| Path | Frequency Range | Use Case |
|------|-----------------|----------|
| LNAH | 1.5-3.8 GHz | High band cellular, WiFi 5GHz |
| LNAL | 0.3-2.2 GHz | Low band cellular, LoRa, WiFi 2.4GHz |
| LNAW | Wideband | Wideband scanning |

### Duplexer Bands
The backend supports 18 predefined duplexer bands including:
- Cellular: Band 2, 3, 5, 7, 8
- TDD: Band 38, 40
- RX-only and TX-only bands

---

## Architecture Notes

```
┌─────────────────────────────────────────────────────────────┐
│                 Devolution(x) Frontend                       │
│                 (React + Socket.IO Client)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌───────────────┐            ┌────────────────────┐
│   REST API    │            │  Socket.IO Server  │
│   (FastAPI)   │            │   (python-socketio)│
│   Port 8000   │            │     Port 8765      │
└───────┬───────┘            └─────────┬──────────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   RF Forensics Pipeline      │
        │   (GPU-Accelerated)          │
        ├──────────────────────────────┤
        │ • SDR Acquisition (uSDR)     │
        │ • FFT/PSD Computation        │
        │ • CFAR Detection             │
        │ • Signal Classification      │
        │ • Clustering                 │
        └──────────────────────────────┘
```

---

## Data Flow

1. **SDR → Pipeline**: Raw IQ samples acquired from uSDR at configured rate
2. **Pipeline → Socket.IO**:
   - Spectrum data → `/spectrum` namespace (20 Hz)
   - Detections → `/detections` namespace (event-driven)
   - Metrics → `/analytics` namespace (1 Hz)
3. **Frontend → REST API**: Configuration changes, control commands
4. **REST API → Pipeline**: Apply new configuration, start/stop

---

## CORS Configuration

The backend is configured to accept requests from any origin (`*`). For production, restrict this to your frontend domain.

---

## Testing

### Test Socket.IO Connection
```javascript
// In browser console
const io = require('socket.io-client')
const socket = io('http://localhost:8765/spectrum')
socket.on('spectrum:measurement', (d) => console.log('Spectrum:', d.fftSize, 'bins'))
```

### Test REST API
```bash
# Health check
curl http://localhost:8000/health

# Configure SDR
curl -X POST http://localhost:8000/api/sdr/config \
  -H "Content-Type: application/json" \
  -d '{"centerFreqHz": 915000000, "sampleRateHz": 10000000}'

# Start monitoring
curl -X POST http://localhost:8000/api/monitoring/start
```

---

## Troubleshooting

### Socket.IO Connection Issues
- Ensure `python-socketio` and `aiohttp` are installed
- Check CORS settings if connecting from different origin
- Verify port 8765 is not blocked by firewall

### No Spectrum Data
- Check if simulation mode is enabled (`--simulate`)
- Verify SDR hardware is connected (non-simulate mode)
- Check `/hardware` namespace for device status

### Detection Events Not Firing
- Detection threshold may be too high
- Check noise floor in spectrum data
- Verify CFAR parameters in configuration
