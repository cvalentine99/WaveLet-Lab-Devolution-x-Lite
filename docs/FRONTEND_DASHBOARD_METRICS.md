# Frontend Dashboard Metrics Reference

This document describes ALL metrics available from the backend for dashboard display.

---

## Quick Reference: Endpoints

| Endpoint | Purpose | Poll Interval |
|----------|---------|---------------|
| `GET /health` | System health check | 5-10s |
| `GET /api/status` | Pipeline state & GPU metrics | 1s |
| `GET /api/sdr/status` | SDR hardware status | 1s |
| `GET /api/sdr/metrics` | Detailed SDR metrics | 1s |
| `GET /api/sdr/health` | SDR health with warnings | 2-5s |
| `GET /api/detections` | Signal detections list | On-demand or via WS |
| `GET /api/clusters` | Clustered signals | On-demand or via WS |

---

## 1. System Health (`GET /health`)

**Use for:** Top-level health indicator, GPU availability check

```typescript
interface HealthResponse {
  status: "ok" | "error";
  gpu_available: boolean;
  gpu_name: string;           // e.g., "NVIDIA GeForce RTX 4090"
  backend_version: string;    // e.g., "1.0.0"
  timestamp: string;          // ISO 8601 timestamp
}
```

**Example Response:**
```json
{
  "status": "ok",
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "backend_version": "1.0.0",
  "timestamp": "2025-12-03T15:06:34.494576Z"
}
```

**Dashboard Widget:** Health indicator (green/red), GPU name badge

---

## 2. Pipeline Status (`GET /api/status`)

**Use for:** Main dashboard metrics panel, GPU utilization, throughput gauges

```typescript
interface PipelineStatus {
  // Pipeline State
  state: "idle" | "running" | "paused" | "error";
  uptime_seconds: number;
  consecutive_errors: number;

  // Processing Metrics
  samples_processed: number;        // Total samples since start
  detections_count: number;         // Total detections found
  current_throughput_msps: number;  // Current throughput in MSPS
  processing_latency_ms: number;    // Per-frame latency in ms

  // GPU Metrics
  gpu_memory_used_gb: number;       // GPU VRAM in use
  gpu_utilization_percent: number;  // GPU compute utilization (0-100)

  // Buffer/Flow Metrics
  buffer_fill_level: number;        // Ring buffer fill (0.0-1.0)
  dropped_samples: number;          // Samples dropped due to backpressure
  sdr_throttled: boolean;           // True if SDR is being throttled
}
```

**Example Response:**
```json
{
  "state": "running",
  "uptime_seconds": 3600.5,
  "samples_processed": 36000000000,
  "detections_count": 15234,
  "current_throughput_msps": 10.0,
  "gpu_memory_used_gb": 1.32,
  "buffer_fill_level": 0.25,
  "processing_latency_ms": 1.05,
  "consecutive_errors": 0,
  "dropped_samples": 0,
  "sdr_throttled": false,
  "gpu_utilization_percent": 45.0
}
```

### Dashboard Widgets

| Metric | Widget Type | Display |
|--------|-------------|---------|
| `state` | Status badge | "Running" (green), "Idle" (gray), "Error" (red) |
| `uptime_seconds` | Timer | "01:23:45" format |
| `samples_processed` | Counter | "36.0 Gsamples" |
| `detections_count` | Counter | "15,234 signals" |
| `current_throughput_msps` | Gauge | 0-20 MSPS scale |
| `processing_latency_ms` | Gauge | 0-50ms scale (green < 5ms) |
| `gpu_memory_used_gb` | Progress bar | X.XX / 24 GB (for RTX 4090) |
| `gpu_utilization_percent` | Gauge | 0-100% |
| `buffer_fill_level` | Progress bar | 0-100% (yellow > 50%, red > 80%) |
| `dropped_samples` | Counter | Show if > 0 (warning) |
| `sdr_throttled` | Indicator | Show warning if true |

---

## 3. SDR Status (`GET /api/sdr/status`)

**Use for:** SDR connection status, hardware readings

```typescript
interface SDRStatus {
  connected: boolean;
  device_id: string | null;        // e.g., "bus=pci,device=usdr0"
  temperature_c: number;           // Hardware temperature
  actual_freq_hz: number;          // Actual tuned frequency
  actual_sample_rate_hz: number;   // Actual sample rate
  actual_bandwidth_hz: number;     // Actual bandwidth
  rx_path: string;                 // "LNAH" | "LNAL" | "LNAW"
  streaming: boolean;              // Is actively streaming
}
```

**Example Response:**
```json
{
  "connected": true,
  "device_id": "bus=pci,device=usdr0",
  "temperature_c": 48.2,
  "actual_freq_hz": 915000000,
  "actual_sample_rate_hz": 10000000,
  "actual_bandwidth_hz": 10000000,
  "rx_path": "LNAL",
  "streaming": true
}
```

### Dashboard Widgets

| Metric | Widget Type | Display |
|--------|-------------|---------|
| `connected` | Indicator | Green dot = connected |
| `device_id` | Label | Device name |
| `temperature_c` | Gauge | 0-80°C (yellow > 60, red > 70) |
| `actual_freq_hz` | Display | "915.000 MHz" |
| `actual_sample_rate_hz` | Display | "10.0 MSPS" |
| `rx_path` | Badge | "LNAH" / "LNAL" / "LNAW" |
| `streaming` | Indicator | Pulsing green when active |

---

## 4. SDR Metrics (`GET /api/sdr/metrics`)

**Use for:** Detailed performance metrics, debugging, advanced dashboard

```typescript
interface SDRMetrics {
  overflow: {
    total: number;                 // Total overflow events
    rate_per_sec: number;          // Current overflow rate
    last_timestamp: number | null; // Last overflow time
  };
  samples: {
    total_received: number;        // Total samples from SDR
    total_dropped: number;         // Dropped due to backpressure
    drop_rate_percent: number;     // Current drop rate
  };
  hardware: {
    temperature_c: number;
    pll_locked: boolean;           // PLL lock status
    actual_sample_rate_hz: number;
    actual_freq_hz: number;
  };
  streaming: {
    uptime_seconds: number;        // Time since stream started
    reconnect_count: number;       // Auto-reconnection count
    last_error: string | null;     // Last error message
  };
  backpressure: {
    events: number;                // Backpressure event count
    buffer_fill_percent: number;   // Current buffer fill
  };
}
```

**Example Response:**
```json
{
  "overflow": {
    "total": 0,
    "rate_per_sec": 0.0,
    "last_timestamp": null
  },
  "samples": {
    "total_received": 40230715392,
    "total_dropped": 1545469952,
    "drop_rate_percent": 3.84
  },
  "hardware": {
    "temperature_c": 48.2,
    "pll_locked": true,
    "actual_sample_rate_hz": 10000000,
    "actual_freq_hz": 915000000
  },
  "streaming": {
    "uptime_seconds": 3600.5,
    "reconnect_count": 0,
    "last_error": null
  },
  "backpressure": {
    "events": 0,
    "buffer_fill_percent": 25.0
  }
}
```

### Dashboard Widgets (Advanced Panel)

| Metric | Widget | Warning Threshold |
|--------|--------|-------------------|
| `overflow.rate_per_sec` | Sparkline | > 1.0/s = warning |
| `samples.drop_rate_percent` | Gauge | > 1% = yellow, > 5% = red |
| `hardware.pll_locked` | Indicator | false = red warning |
| `streaming.reconnect_count` | Counter | > 0 = show |
| `backpressure.events` | Counter | > 0 = show |
| `backpressure.buffer_fill_percent` | Progress | > 75% = yellow, > 90% = red |

---

## 5. SDR Health (`GET /api/sdr/health`)

**Use for:** Quick health summary with warnings

```typescript
interface SDRHealth {
  status: "healthy" | "degraded" | "disconnected" | "unavailable";
  connected: boolean;
  streaming: boolean;
  warnings: string[];              // List of warning messages
  metrics_summary: {
    overflow_rate: number;
    drop_rate_percent: number;
    buffer_fill_percent: number;
    temperature_c: number;
    uptime_seconds: number;
  };
}
```

**Example Response:**
```json
{
  "status": "degraded",
  "connected": true,
  "streaming": true,
  "warnings": [
    "High drop rate: 3.8%",
    "Buffer near full: 87.5%"
  ],
  "metrics_summary": {
    "overflow_rate": 0.0,
    "drop_rate_percent": 3.84,
    "buffer_fill_percent": 87.5,
    "temperature_c": 48.2,
    "uptime_seconds": 3600.5
  }
}
```

### Status Color Mapping

| Status | Color | Icon |
|--------|-------|------|
| `healthy` | Green | ✓ |
| `degraded` | Yellow/Orange | ⚠ |
| `disconnected` | Gray | ○ |
| `unavailable` | Red | ✗ |

---

## 6. Detections (`GET /api/detections`)

**Use for:** Signal list table, detection history

**Query Parameters:**
- `limit` - Max results (default 100)
- `offset` - Pagination offset
- `min_snr` - Filter by minimum SNR
- `modulation` - Filter by modulation type

```typescript
interface Detection {
  detection_id: number;
  center_freq_hz: number;
  bandwidth_hz: number;
  peak_power_db: number;
  snr_db: number;
  modulation_type: string;         // "FM", "AM", "BPSK", "QPSK", "Unknown"
  confidence: number;              // 0.0 - 1.0
  timestamp: number;               // Unix timestamp
  track_id: string;                // Tracking ID for persistence
  frames_seen: number;             // How many frames signal detected
  duty_cycle: number;              // 0.0 - 1.0
  first_seen: number;              // First detection timestamp
  last_seen: number;               // Last detection timestamp
  frequency_drift_hz_per_s: number; // Frequency drift rate
}

interface DetectionsResponse {
  total: number;
  offset: number;
  limit: number;
  detections: Detection[];
}
```

**Example Response:**
```json
{
  "total": 15234,
  "offset": 0,
  "limit": 100,
  "detections": [
    {
      "detection_id": 12345,
      "center_freq_hz": 915234567.0,
      "bandwidth_hz": 25000.0,
      "peak_power_db": -45.2,
      "snr_db": 25.5,
      "modulation_type": "FM",
      "confidence": 0.92,
      "timestamp": 1733234567.123,
      "track_id": "trk-000042",
      "frames_seen": 150,
      "duty_cycle": 0.85,
      "first_seen": 1733234500.0,
      "last_seen": 1733234567.123,
      "frequency_drift_hz_per_s": 2.5
    }
  ]
}
```

### Table Columns

| Column | Field | Format |
|--------|-------|--------|
| ID | `detection_id` | #12345 |
| Frequency | `center_freq_hz` | 915.235 MHz |
| Bandwidth | `bandwidth_hz` | 25 kHz |
| Power | `peak_power_db` | -45.2 dBm |
| SNR | `snr_db` | 25.5 dB |
| Modulation | `modulation_type` | FM |
| Confidence | `confidence` | 92% (progress bar) |
| Duration | `last_seen - first_seen` | 1m 7s |
| Duty | `duty_cycle` | 85% |

---

## 7. Clusters (`GET /api/clusters`)

**Use for:** Grouped signal analysis, anomaly detection

```typescript
interface Cluster {
  cluster_id: number;
  size: number;                    // Number of detections in cluster
  center_freq_hz: number;          // Cluster center frequency
  freq_range_hz: [number, number]; // [min, max] frequency range
  avg_snr_db: number;
  avg_power_db: number;
  avg_bandwidth_hz: number;
  detection_count: number;
  label: string;                   // Cluster label
  avg_duty_cycle: number;
  unique_tracks: number;           // Number of unique signals
  signal_type_hint: string;        // "narrowband_cw", "wideband", "pulsed"
}

interface ClustersResponse {
  clusters: Cluster[];
}
```

**Example Response:**
```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "size": 956,
      "center_freq_hz": 915000000.0,
      "freq_range_hz": [914500000.0, 915500000.0],
      "avg_snr_db": 22.5,
      "avg_power_db": -52.3,
      "avg_bandwidth_hz": 25000.0,
      "detection_count": 956,
      "label": "ISM_915",
      "avg_duty_cycle": 0.75,
      "unique_tracks": 12,
      "signal_type_hint": "narrowband_cw"
    }
  ]
}
```

---

## 8. WebSocket Real-Time Metrics

For real-time dashboard updates, use WebSockets instead of polling:

### Spectrum Data (`ws://localhost:8765/ws/spectrum`)

**Binary format** - 20-byte header + float32 array:

```typescript
// Parse spectrum binary data
function parseSpectrumData(buffer: ArrayBuffer) {
  const view = new DataView(buffer);

  const header = {
    timestamp: view.getFloat64(0, true),      // 8 bytes
    center_freq_hz: view.getFloat32(8, true), // 4 bytes
    span_hz: view.getFloat32(12, true),       // 4 bytes
    num_bins: view.getUint16(16, true),       // 2 bytes
    flags: view.getUint16(18, true),          // 2 bytes
  };

  // Power values in dBFS (float32 array)
  const powerData = new Float32Array(buffer.slice(20));

  return { header, powerData };
}
```

### Detections Stream (`ws://localhost:8765/ws/detections`)

**JSON format:**

```typescript
interface DetectionMessage {
  type: "detection";
  data: Detection;  // Same as REST detection object
}
```

### Clusters Stream (`ws://localhost:8765/ws/clusters`)

**JSON format:**

```typescript
interface ClusterMessage {
  type: "cluster";
  data: Cluster;  // Same as REST cluster object
}
```

---

## 9. Frontend Implementation Example

```typescript
// Dashboard metrics hook
import { useState, useEffect } from 'react';

interface DashboardMetrics {
  pipeline: PipelineStatus | null;
  sdrHealth: SDRHealth | null;
  systemHealth: HealthResponse | null;
}

export function useDashboardMetrics(pollInterval = 1000) {
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    pipeline: null,
    sdrHealth: null,
    systemHealth: null,
  });

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const [pipeline, sdrHealth, systemHealth] = await Promise.all([
          fetch('http://localhost:8000/api/status').then(r => r.json()),
          fetch('http://localhost:8000/api/sdr/health').then(r => r.json()),
          fetch('http://localhost:8000/health').then(r => r.json()),
        ]);

        setMetrics({ pipeline, sdrHealth, systemHealth });
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, pollInterval);

    return () => clearInterval(interval);
  }, [pollInterval]);

  return metrics;
}

// Usage in component
function Dashboard() {
  const { pipeline, sdrHealth, systemHealth } = useDashboardMetrics(1000);

  if (!pipeline) return <Loading />;

  return (
    <div className="dashboard">
      {/* System Health */}
      <HealthBadge status={systemHealth?.status} gpu={systemHealth?.gpu_name} />

      {/* Pipeline State */}
      <StateBadge state={pipeline.state} />
      <UptimeDisplay seconds={pipeline.uptime_seconds} />

      {/* Throughput Gauge */}
      <Gauge
        value={pipeline.current_throughput_msps}
        max={20}
        unit="MSPS"
        label="Throughput"
      />

      {/* GPU Metrics */}
      <ProgressBar
        value={pipeline.gpu_memory_used_gb}
        max={24}
        label="GPU Memory"
        unit="GB"
      />
      <Gauge
        value={pipeline.gpu_utilization_percent}
        max={100}
        unit="%"
        label="GPU Util"
      />

      {/* Processing */}
      <Counter value={pipeline.samples_processed} label="Samples" />
      <Counter value={pipeline.detections_count} label="Detections" />
      <Gauge
        value={pipeline.processing_latency_ms}
        max={50}
        unit="ms"
        label="Latency"
        thresholds={{ green: 5, yellow: 20, red: 50 }}
      />

      {/* SDR Health */}
      <SDRHealthPanel
        status={sdrHealth?.status}
        warnings={sdrHealth?.warnings}
        temperature={sdrHealth?.metrics_summary.temperature_c}
      />

      {/* Buffer Status */}
      <ProgressBar
        value={pipeline.buffer_fill_level * 100}
        max={100}
        label="Buffer"
        thresholds={{ yellow: 50, red: 80 }}
      />

      {/* Warnings */}
      {pipeline.dropped_samples > 0 && (
        <Warning>Dropped {pipeline.dropped_samples} samples</Warning>
      )}
      {pipeline.sdr_throttled && (
        <Warning>SDR throttled - reduce sample rate</Warning>
      )}
    </div>
  );
}
```

---

## 10. Metric Thresholds Reference

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| `processing_latency_ms` | < 5 | 5-20 | > 20 |
| `buffer_fill_level` | < 0.5 | 0.5-0.8 | > 0.8 |
| `drop_rate_percent` | 0 | 0.1-1.0 | > 1.0 |
| `overflow_rate_per_sec` | 0 | 0.1-1.0 | > 1.0 |
| `temperature_c` | < 50 | 50-70 | > 70 |
| `gpu_utilization_percent` | < 80 | 80-95 | > 95 |
| `consecutive_errors` | 0 | 1-5 | > 5 |

---

## 11. Environment Variables

The frontend should use these environment variables:

```bash
VITE_BACKEND_URL=http://localhost:8000
VITE_BACKEND_WS_URL=ws://localhost:8765
```

```typescript
// In your API client
const API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
const WS_BASE = import.meta.env.VITE_BACKEND_WS_URL || 'ws://localhost:8765';
```
