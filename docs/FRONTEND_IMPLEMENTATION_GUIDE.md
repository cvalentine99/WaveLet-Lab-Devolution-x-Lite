# RF Forensics Engine - Frontend Implementation Guide

## For React + Tailwind CSS Developers

This guide provides complete specifications for building the web GUI that interfaces with the GPU RF Forensics Engine backend.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack Recommendations](#tech-stack-recommendations)
3. [WebSocket Protocol](#websocket-protocol)
4. [REST API Reference](#rest-api-reference)
5. [UI Components Specification](#ui-components-specification)
6. [Real-Time Visualization](#real-time-visualization)
7. [State Management](#state-management)
8. [Example Code](#example-code)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     React Frontend                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Controls   │  │  Spectrum   │  │   Waterfall Display     │  │
│  │   Panel     │  │   Display   │  │   (WebGL/Canvas)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Detections  │  │  Clusters   │  │   Demodulation View     │  │
│  │   Table     │  │    View     │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    WebSocket + REST API Layer                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              GPU RF Forensics Engine (Python Backend)            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ WS:8765  │  │ REST:8000│  │   SDR    │  │  GPU Processing  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Connection Points

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| WebSocket - Spectrum | 8765 | `ws://` | Binary PSD streaming (30 Hz) |
| WebSocket - Detections | 8765 | `ws://` | JSON detection events |
| WebSocket - Clusters | 8765 | `ws://` | JSON cluster updates |
| REST API | 8000 | `http://` | Control & configuration |

---

## Tech Stack Recommendations

### Required
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "tailwindcss": "^3.4.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "recharts": "^2.10.0"
  }
}
```

### Recommended Additions
```json
{
  "dependencies": {
    "three": "^0.160.0",
    "@react-three/fiber": "^8.15.0",
    "regl": "^2.1.0",
    "d3-scale": "^4.0.0",
    "d3-scale-chromatic": "^3.0.0",
    "lucide-react": "^0.300.0",
    "@radix-ui/react-slider": "^1.1.0",
    "@radix-ui/react-select": "^2.0.0",
    "@radix-ui/react-tabs": "^1.0.0"
  }
}
```

---

## WebSocket Protocol

### Endpoint: `/ws/spectrum` (Binary)

High-rate binary PSD data for spectrum and waterfall displays.

#### Binary Message Format

```
┌────────────────────────────────────────────────────────────┐
│                    Header (16 bytes)                        │
├──────────────┬──────────────┬────────────┬────────┬────────┤
│ timestamp    │ center_freq  │ span       │num_bins│reserved│
│ (float64)    │ (float32)    │ (float32)  │(uint16)│(uint16)│
│ 8 bytes      │ 4 bytes      │ 4 bytes    │2 bytes │2 bytes │
├──────────────┴──────────────┴────────────┴────────┴────────┤
│                    Payload (num_bins bytes)                 │
│              uint8 array - quantized dB values              │
│              0 = -120 dBm, 255 = -20 dBm                    │
└────────────────────────────────────────────────────────────┘
```

#### JavaScript Parser

```javascript
function parseSpectrumMessage(arrayBuffer) {
  const view = new DataView(arrayBuffer);

  // Parse header (little-endian)
  const timestamp = view.getFloat64(0, true);
  const centerFreq = view.getFloat32(8, true);
  const span = view.getFloat32(12, true);
  const numBins = view.getUint16(14, true);

  // Parse payload
  const payload = new Uint8Array(arrayBuffer, 16, numBins);

  // Convert uint8 to dB values
  const psdDb = new Float32Array(numBins);
  for (let i = 0; i < numBins; i++) {
    psdDb[i] = (payload[i] / 255) * 100 - 120; // Map 0-255 to -120 to -20 dB
  }

  return {
    timestamp,
    centerFreq,
    span,
    numBins,
    psdDb,
    // Generate frequency axis
    frequencies: Array.from({ length: numBins }, (_, i) =>
      centerFreq - span/2 + (i / numBins) * span
    )
  };
}
```

### Endpoint: `/ws/detections` (JSON)

Detection events pushed when signals are found.

```typescript
interface DetectionEvent {
  type: "detection";
  timestamp: number;
  data: {
    detection_id: number;
    center_freq_hz: number;
    bandwidth_hz: number;
    bandwidth_3db_hz: number;
    bandwidth_6db_hz: number;
    peak_power_db: number;
    snr_db: number;
    start_bin: number;
    end_bin: number;
  };
}
```

### Endpoint: `/ws/clusters` (JSON)

Cluster updates pushed periodically.

```typescript
interface ClusterUpdate {
  type: "clusters";
  timestamp: number;
  data: Array<{
    cluster_id: number;
    size: number;
    dominant_frequency_hz: number;
    avg_snr_db: number;
    label: string;
  }>;
}
```

### WebSocket Connection Manager

```typescript
// hooks/useWebSocket.ts
import { useEffect, useRef, useCallback } from 'react';

interface WebSocketConfig {
  url: string;
  onBinaryMessage?: (data: ArrayBuffer) => void;
  onJsonMessage?: (data: any) => void;
  reconnectInterval?: number;
}

export function useWebSocket({
  url,
  onBinaryMessage,
  onJsonMessage,
  reconnectInterval = 3000
}: WebSocketConfig) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();

  const connect = useCallback(() => {
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      console.log(`Connected to ${url}`);
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        onBinaryMessage?.(event.data);
      } else {
        try {
          const json = JSON.parse(event.data);
          onJsonMessage?.(json);
        } catch (e) {
          console.error('Failed to parse JSON:', e);
        }
      }
    };

    ws.onclose = () => {
      console.log(`Disconnected from ${url}`);
      reconnectTimeoutRef.current = window.setTimeout(connect, reconnectInterval);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    wsRef.current = ws;
  }, [url, onBinaryMessage, onJsonMessage, reconnectInterval]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((data: string | ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    }
  }, []);

  return { send };
}
```

---

## REST API Reference

Base URL: `http://localhost:8000`

### System Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/start` | Start acquisition |
| `POST` | `/api/stop` | Stop acquisition |
| `POST` | `/api/pause` | Pause processing |
| `POST` | `/api/resume` | Resume processing |
| `GET` | `/api/status` | Get system status |

#### Status Response
```typescript
interface SystemStatus {
  state: "idle" | "configuring" | "running" | "paused" | "error";
  uptime_seconds: number;
  samples_processed: number;
  detections_count: number;
  current_throughput_msps: number;
  gpu_memory_used_gb: number;
  buffer_fill_level: number;
  processing_latency_ms: number;
}
```

### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config` | Get current config |
| `PUT` | `/api/config` | Update config |
| `GET` | `/api/config/presets` | List presets |
| `POST` | `/api/config/presets/{name}` | Apply preset |

#### Configuration Schema
```typescript
interface Config {
  sdr: {
    device_type: "usrp" | "hackrf" | "usdr";
    center_freq_hz: number;  // 1e6 to 6e9
    sample_rate_hz: number;  // 1e3 to 61.44e6
    bandwidth_hz: number;
    gain_db: number;         // 0 to 76
  };
  pipeline: {
    fft_size: number;        // Power of 2: 64-65536
    window_type: "hann" | "hamming" | "blackman" | "kaiser" | "flattop";
    overlap_percent: number; // 0-90
    averaging_count: number; // 1-1000
  };
  cfar: {
    num_reference_cells: number;
    num_guard_cells: number;
    pfa: number;             // 1e-12 to 0.1
    variant: "CA" | "GO" | "SO" | "OS";
  };
}
```

### Detections

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/detections` | List detections |
| `GET` | `/api/detections/{id}` | Get detection details |
| `POST` | `/api/detections/export` | Export to file |

Query parameters for `/api/detections`:
- `limit` (int, max 1000)
- `offset` (int)
- `min_snr_db` (float)

### Clusters

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/clusters` | List clusters |
| `GET` | `/api/clusters/{id}` | Get cluster details |
| `POST` | `/api/clusters/{id}/label` | Label a cluster |

### API Client

```typescript
// lib/api.ts
const API_BASE = 'http://localhost:8000';

export const api = {
  // System control
  start: () => fetch(`${API_BASE}/api/start`, { method: 'POST' }),
  stop: () => fetch(`${API_BASE}/api/stop`, { method: 'POST' }),
  pause: () => fetch(`${API_BASE}/api/pause`, { method: 'POST' }),
  resume: () => fetch(`${API_BASE}/api/resume`, { method: 'POST' }),

  getStatus: () =>
    fetch(`${API_BASE}/api/status`).then(r => r.json()),

  // Configuration
  getConfig: () =>
    fetch(`${API_BASE}/api/config`).then(r => r.json()),

  updateConfig: (config: Partial<Config>) =>
    fetch(`${API_BASE}/api/config`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    }),

  getPresets: () =>
    fetch(`${API_BASE}/api/config/presets`).then(r => r.json()),

  applyPreset: (name: string) =>
    fetch(`${API_BASE}/api/config/presets/${name}`, { method: 'POST' }),

  // Detections
  getDetections: (params?: { limit?: number; offset?: number; min_snr_db?: number }) => {
    const query = new URLSearchParams(params as any).toString();
    return fetch(`${API_BASE}/api/detections?${query}`).then(r => r.json());
  },

  // Clusters
  getClusters: () =>
    fetch(`${API_BASE}/api/clusters`).then(r => r.json()),

  labelCluster: (id: number, label: string) =>
    fetch(`${API_BASE}/api/clusters/${id}/label?label=${encodeURIComponent(label)}`, {
      method: 'POST'
    })
};
```

---

## UI Components Specification

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│ Header: Status Bar                                                   │
│ [State: Running] [915.0 MHz] [10 MSps] [GPU: 2.1 GB] [Latency: 5ms] │
├────────────────────┬────────────────────────────────────────────────┤
│                    │                                                 │
│  Control Panel     │  Main Display Area                             │
│  (Left Sidebar)    │                                                 │
│                    │  ┌─────────────────────────────────────────┐   │
│  ┌──────────────┐  │  │         Spectrum Display                │   │
│  │ SDR Config   │  │  │         (Line Chart)                    │   │
│  │ - Frequency  │  │  └─────────────────────────────────────────┘   │
│  │ - Sample Rate│  │                                                 │
│  │ - Gain       │  │  ┌─────────────────────────────────────────┐   │
│  └──────────────┘  │  │         Waterfall Display               │   │
│                    │  │         (WebGL Canvas)                  │   │
│  ┌──────────────┐  │  │                                         │   │
│  │ FFT Config   │  │  │         ████████████████████            │   │
│  │ - FFT Size   │  │  │         ██░░████░░██████████            │   │
│  │ - Window     │  │  │         ████████████████████            │   │
│  │ - Averaging  │  │  └─────────────────────────────────────────┘   │
│  └──────────────┘  │                                                 │
│                    ├────────────────────────────────────────────────┤
│  ┌──────────────┐  │  Bottom Panel (Tabs)                           │
│  │ CFAR Config  │  │  ┌──────┬──────────┬─────────┬───────────────┐ │
│  │ - Pfa        │  │  │Detect│ Clusters │ Demod   │ Constellation │ │
│  │ - Ref Cells  │  │  └──────┴──────────┴─────────┴───────────────┘ │
│  └──────────────┘  │                                                 │
│                    │  [Detection Table / Cluster View / etc.]       │
│  [Start] [Stop]    │                                                 │
│                    │                                                 │
└────────────────────┴────────────────────────────────────────────────┘
```

### Component Specifications

#### 1. Status Bar

```tsx
// components/StatusBar.tsx
interface StatusBarProps {
  status: SystemStatus;
}

export function StatusBar({ status }: StatusBarProps) {
  const stateColors = {
    idle: 'bg-gray-500',
    running: 'bg-green-500',
    paused: 'bg-yellow-500',
    error: 'bg-red-500',
    configuring: 'bg-blue-500'
  };

  return (
    <div className="h-10 bg-gray-900 border-b border-gray-700 flex items-center px-4 gap-6 text-sm">
      {/* State indicator */}
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${stateColors[status.state]}`} />
        <span className="text-gray-300 capitalize">{status.state}</span>
      </div>

      {/* Metrics */}
      <div className="flex items-center gap-4 text-gray-400">
        <span>{(status.current_throughput_msps).toFixed(1)} MSps</span>
        <span>GPU: {status.gpu_memory_used_gb.toFixed(1)} GB</span>
        <span>Latency: {status.processing_latency_ms.toFixed(1)} ms</span>
        <span>Detections: {status.detections_count}</span>
        <span>Buffer: {(status.buffer_fill_level * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}
```

#### 2. Control Panel

```tsx
// components/ControlPanel.tsx
interface ControlPanelProps {
  config: Config;
  onConfigChange: (config: Partial<Config>) => void;
  onStart: () => void;
  onStop: () => void;
  isRunning: boolean;
}

export function ControlPanel({ config, onConfigChange, onStart, onStop, isRunning }: ControlPanelProps) {
  return (
    <div className="w-72 bg-gray-900 border-r border-gray-700 p-4 space-y-6 overflow-y-auto">
      {/* SDR Configuration */}
      <section>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">SDR Configuration</h3>

        {/* Center Frequency */}
        <div className="mb-4">
          <label className="block text-xs text-gray-500 mb-1">Center Frequency</label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              value={config.sdr.center_freq_hz / 1e6}
              onChange={(e) => onConfigChange({
                sdr: { ...config.sdr, center_freq_hz: parseFloat(e.target.value) * 1e6 }
              })}
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white"
              step="0.1"
            />
            <span className="text-gray-500 text-sm">MHz</span>
          </div>
        </div>

        {/* Sample Rate */}
        <div className="mb-4">
          <label className="block text-xs text-gray-500 mb-1">Sample Rate</label>
          <select
            value={config.sdr.sample_rate_hz}
            onChange={(e) => onConfigChange({
              sdr: { ...config.sdr, sample_rate_hz: parseFloat(e.target.value) }
            })}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white"
          >
            <option value={1e6}>1 MSps</option>
            <option value={2e6}>2 MSps</option>
            <option value={5e6}>5 MSps</option>
            <option value={10e6}>10 MSps</option>
            <option value={20e6}>20 MSps</option>
          </select>
        </div>

        {/* Gain Slider */}
        <div className="mb-4">
          <label className="block text-xs text-gray-500 mb-1">
            Gain: {config.sdr.gain_db} dB
          </label>
          <input
            type="range"
            min={0}
            max={76}
            value={config.sdr.gain_db}
            onChange={(e) => onConfigChange({
              sdr: { ...config.sdr, gain_db: parseFloat(e.target.value) }
            })}
            className="w-full accent-blue-500"
          />
        </div>
      </section>

      {/* FFT Configuration */}
      <section>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">FFT Settings</h3>

        <div className="mb-4">
          <label className="block text-xs text-gray-500 mb-1">FFT Size</label>
          <select
            value={config.pipeline.fft_size}
            onChange={(e) => onConfigChange({
              pipeline: { ...config.pipeline, fft_size: parseInt(e.target.value) }
            })}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white"
          >
            {[256, 512, 1024, 2048, 4096, 8192].map(size => (
              <option key={size} value={size}>{size}</option>
            ))}
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-xs text-gray-500 mb-1">Window</label>
          <select
            value={config.pipeline.window_type}
            onChange={(e) => onConfigChange({
              pipeline: { ...config.pipeline, window_type: e.target.value }
            })}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-white"
          >
            <option value="hann">Hann</option>
            <option value="hamming">Hamming</option>
            <option value="blackman">Blackman</option>
            <option value="kaiser">Kaiser</option>
            <option value="flattop">Flat Top</option>
          </select>
        </div>
      </section>

      {/* CFAR Configuration */}
      <section>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Detection (CFAR)</h3>

        <div className="mb-4">
          <label className="block text-xs text-gray-500 mb-1">
            Pfa: {config.cfar.pfa.toExponential(1)}
          </label>
          <input
            type="range"
            min={-12}
            max={-1}
            step={1}
            value={Math.log10(config.cfar.pfa)}
            onChange={(e) => onConfigChange({
              cfar: { ...config.cfar, pfa: Math.pow(10, parseFloat(e.target.value)) }
            })}
            className="w-full accent-blue-500"
          />
        </div>
      </section>

      {/* Control Buttons */}
      <div className="flex gap-2 pt-4 border-t border-gray-700">
        {!isRunning ? (
          <button
            onClick={onStart}
            className="flex-1 bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded font-medium transition-colors"
          >
            Start
          </button>
        ) : (
          <button
            onClick={onStop}
            className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded font-medium transition-colors"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
```

#### 3. Spectrum Display (Line Chart)

```tsx
// components/SpectrumDisplay.tsx
import { LineChart, Line, XAxis, YAxis, ReferenceLine, ResponsiveContainer } from 'recharts';

interface SpectrumDisplayProps {
  frequencies: number[];  // Hz
  psdDb: number[];        // dB
  detections?: Detection[];
  threshold?: number[];   // CFAR threshold
}

export function SpectrumDisplay({ frequencies, psdDb, detections, threshold }: SpectrumDisplayProps) {
  // Prepare data for recharts
  const data = frequencies.map((freq, i) => ({
    freq: freq / 1e6,  // Convert to MHz
    psd: psdDb[i],
    threshold: threshold?.[i]
  }));

  // Format frequency for display
  const formatFreq = (value: number) => `${value.toFixed(2)} MHz`;

  return (
    <div className="h-64 bg-gray-950 rounded border border-gray-800 p-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 40 }}>
          <XAxis
            dataKey="freq"
            stroke="#6b7280"
            fontSize={10}
            tickFormatter={formatFreq}
          />
          <YAxis
            domain={[-120, -20]}
            stroke="#6b7280"
            fontSize={10}
            tickFormatter={(v) => `${v} dB`}
          />

          {/* CFAR threshold */}
          {threshold && (
            <Line
              type="monotone"
              dataKey="threshold"
              stroke="#f59e0b"
              strokeWidth={1}
              dot={false}
              strokeDasharray="4 2"
            />
          )}

          {/* PSD trace */}
          <Line
            type="monotone"
            dataKey="psd"
            stroke="#3b82f6"
            strokeWidth={1}
            dot={false}
            isAnimationActive={false}
          />

          {/* Detection markers */}
          {detections?.map(det => (
            <ReferenceLine
              key={det.detection_id}
              x={det.center_freq_hz / 1e6}
              stroke="#ef4444"
              strokeWidth={2}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

#### 4. Waterfall Display (WebGL)

**This is the most performance-critical component.** Use WebGL or Canvas2D with careful optimization.

```tsx
// components/WaterfallDisplay.tsx
import { useRef, useEffect, useCallback } from 'react';

interface WaterfallDisplayProps {
  width: number;
  height: number;
  numFreqBins: number;
  numTimeLines: number;
  colormap?: 'viridis' | 'plasma' | 'jet' | 'turbo';
}

// Viridis colormap (256 colors)
const VIRIDIS = generateViridisColormap();

export function WaterfallDisplay({
  width,
  height,
  numFreqBins = 1024,
  numTimeLines = 256,
  colormap = 'viridis'
}: WaterfallDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const textureRef = useRef<WebGLTexture | null>(null);
  const dataRef = useRef<Uint8Array>(new Uint8Array(numFreqBins * numTimeLines));
  const writeLineRef = useRef(0);

  // Initialize WebGL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', {
      antialias: false,
      preserveDrawingBuffer: false
    });
    if (!gl) {
      console.error('WebGL not supported');
      return;
    }
    glRef.current = gl;

    // Create shader program
    const program = createShaderProgram(gl);
    gl.useProgram(program);

    // Create texture for waterfall data
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    textureRef.current = texture;

    // Create colormap texture
    const colormapTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, colormapTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, VIRIDIS);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Set uniforms
    gl.uniform1i(gl.getUniformLocation(program, 'u_data'), 0);
    gl.uniform1i(gl.getUniformLocation(program, 'u_colormap'), 1);

    // Setup fullscreen quad
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    return () => {
      gl.deleteTexture(texture);
      gl.deleteTexture(colormapTexture);
    };
  }, [numFreqBins, numTimeLines]);

  // Add new spectrum line
  const addLine = useCallback((psdUint8: Uint8Array) => {
    const offset = writeLineRef.current * numFreqBins;
    dataRef.current.set(psdUint8, offset);
    writeLineRef.current = (writeLineRef.current + 1) % numTimeLines;

    // Update texture
    const gl = glRef.current;
    if (!gl || !textureRef.current) return;

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textureRef.current);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.LUMINANCE,
      numFreqBins, numTimeLines, 0,
      gl.LUMINANCE, gl.UNSIGNED_BYTE, dataRef.current
    );

    // Render
    gl.viewport(0, 0, width, height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }, [numFreqBins, numTimeLines, width, height]);

  // Expose addLine method
  useEffect(() => {
    (window as any).__waterfallAddLine = addLine;
    return () => { delete (window as any).__waterfallAddLine; };
  }, [addLine]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="bg-black rounded border border-gray-800"
    />
  );
}

// Shader creation helper
function createShaderProgram(gl: WebGLRenderingContext): WebGLProgram {
  const vsSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;
    void main() {
      gl_Position = vec4(a_position, 0, 1);
      v_texCoord = (a_position + 1.0) / 2.0;
    }
  `;

  const fsSource = `
    precision mediump float;
    varying vec2 v_texCoord;
    uniform sampler2D u_data;
    uniform sampler2D u_colormap;
    void main() {
      float value = texture2D(u_data, v_texCoord).r;
      gl_FragColor = texture2D(u_colormap, vec2(value, 0.5));
    }
  `;

  const vs = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vs, vsSource);
  gl.compileShader(vs);

  const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fs, fsSource);
  gl.compileShader(fs);

  const program = gl.createProgram()!;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);

  return program;
}

// Generate Viridis colormap
function generateViridisColormap(): Uint8Array {
  const colors = new Uint8Array(256 * 3);
  // Viridis color data (simplified - use full table in production)
  const viridis = [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37]
  ];

  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const idx = Math.min(Math.floor(t * (viridis.length - 1)), viridis.length - 2);
    const frac = t * (viridis.length - 1) - idx;

    for (let c = 0; c < 3; c++) {
      colors[i * 3 + c] = Math.round(
        viridis[idx][c] * (1 - frac) + viridis[idx + 1][c] * frac
      );
    }
  }

  return colors;
}
```

#### 5. Detections Table

```tsx
// components/DetectionsTable.tsx
interface Detection {
  detection_id: number;
  center_freq_hz: number;
  bandwidth_hz: number;
  peak_power_db: number;
  snr_db: number;
  timestamp: number;
}

interface DetectionsTableProps {
  detections: Detection[];
  onSelectDetection?: (detection: Detection) => void;
}

export function DetectionsTable({ detections, onSelectDetection }: DetectionsTableProps) {
  const formatFreq = (hz: number) => {
    if (hz >= 1e9) return `${(hz / 1e9).toFixed(3)} GHz`;
    if (hz >= 1e6) return `${(hz / 1e6).toFixed(3)} MHz`;
    return `${(hz / 1e3).toFixed(1)} kHz`;
  };

  const formatBandwidth = (hz: number) => {
    if (hz >= 1e6) return `${(hz / 1e6).toFixed(1)} MHz`;
    return `${(hz / 1e3).toFixed(1)} kHz`;
  };

  return (
    <div className="overflow-auto max-h-64">
      <table className="w-full text-sm">
        <thead className="bg-gray-800 sticky top-0">
          <tr>
            <th className="px-3 py-2 text-left text-gray-400 font-medium">ID</th>
            <th className="px-3 py-2 text-left text-gray-400 font-medium">Frequency</th>
            <th className="px-3 py-2 text-left text-gray-400 font-medium">Bandwidth</th>
            <th className="px-3 py-2 text-left text-gray-400 font-medium">Power</th>
            <th className="px-3 py-2 text-left text-gray-400 font-medium">SNR</th>
            <th className="px-3 py-2 text-left text-gray-400 font-medium">Time</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-800">
          {detections.map(det => (
            <tr
              key={det.detection_id}
              onClick={() => onSelectDetection?.(det)}
              className="hover:bg-gray-800/50 cursor-pointer transition-colors"
            >
              <td className="px-3 py-2 text-gray-300">{det.detection_id}</td>
              <td className="px-3 py-2 text-blue-400 font-mono">
                {formatFreq(det.center_freq_hz)}
              </td>
              <td className="px-3 py-2 text-gray-400">
                {formatBandwidth(det.bandwidth_hz)}
              </td>
              <td className="px-3 py-2 text-gray-300">
                {det.peak_power_db.toFixed(1)} dBm
              </td>
              <td className={`px-3 py-2 font-medium ${
                det.snr_db > 20 ? 'text-green-400' :
                det.snr_db > 10 ? 'text-yellow-400' : 'text-gray-400'
              }`}>
                {det.snr_db.toFixed(1)} dB
              </td>
              <td className="px-3 py-2 text-gray-500 text-xs">
                {new Date(det.timestamp * 1000).toLocaleTimeString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {detections.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No detections yet
        </div>
      )}
    </div>
  );
}
```

---

## State Management

Use **Zustand** for simple, performant state management:

```typescript
// store/useStore.ts
import { create } from 'zustand';

interface SpectrumData {
  frequencies: number[];
  psdDb: Float32Array;
  timestamp: number;
}

interface RFStore {
  // Connection state
  isConnected: boolean;
  setConnected: (connected: boolean) => void;

  // System status
  status: SystemStatus | null;
  setStatus: (status: SystemStatus) => void;

  // Configuration
  config: Config | null;
  setConfig: (config: Config) => void;
  updateConfig: (partial: Partial<Config>) => void;

  // Spectrum data (keep minimal for performance)
  spectrum: SpectrumData | null;
  setSpectrum: (data: SpectrumData) => void;

  // Detections
  detections: Detection[];
  addDetection: (detection: Detection) => void;
  clearDetections: () => void;

  // Clusters
  clusters: Cluster[];
  setClusters: (clusters: Cluster[]) => void;

  // UI state
  selectedDetection: Detection | null;
  setSelectedDetection: (detection: Detection | null) => void;

  activeTab: 'detections' | 'clusters' | 'demod' | 'constellation';
  setActiveTab: (tab: string) => void;
}

export const useStore = create<RFStore>((set, get) => ({
  // Connection
  isConnected: false,
  setConnected: (connected) => set({ isConnected: connected }),

  // Status
  status: null,
  setStatus: (status) => set({ status }),

  // Config
  config: null,
  setConfig: (config) => set({ config }),
  updateConfig: (partial) => set((state) => ({
    config: state.config ? { ...state.config, ...partial } : null
  })),

  // Spectrum (high-frequency updates)
  spectrum: null,
  setSpectrum: (data) => set({ spectrum: data }),

  // Detections (keep last 500)
  detections: [],
  addDetection: (detection) => set((state) => ({
    detections: [...state.detections.slice(-499), detection]
  })),
  clearDetections: () => set({ detections: [] }),

  // Clusters
  clusters: [],
  setClusters: (clusters) => set({ clusters }),

  // UI
  selectedDetection: null,
  setSelectedDetection: (detection) => set({ selectedDetection: detection }),

  activeTab: 'detections',
  setActiveTab: (tab) => set({ activeTab: tab as any })
}));
```

---

## Example Code

### Main App Component

```tsx
// App.tsx
import { useEffect } from 'react';
import { useStore } from './store/useStore';
import { useWebSocket } from './hooks/useWebSocket';
import { api } from './lib/api';
import { StatusBar } from './components/StatusBar';
import { ControlPanel } from './components/ControlPanel';
import { SpectrumDisplay } from './components/SpectrumDisplay';
import { WaterfallDisplay } from './components/WaterfallDisplay';
import { DetectionsTable } from './components/DetectionsTable';

function parseSpectrumMessage(buffer: ArrayBuffer) {
  const view = new DataView(buffer);
  const timestamp = view.getFloat64(0, true);
  const centerFreq = view.getFloat32(8, true);
  const span = view.getFloat32(12, true);
  const numBins = view.getUint16(14, true);

  const payload = new Uint8Array(buffer, 16, numBins);
  const psdDb = new Float32Array(numBins);

  for (let i = 0; i < numBins; i++) {
    psdDb[i] = (payload[i] / 255) * 100 - 120;
  }

  const frequencies = Array.from({ length: numBins }, (_, i) =>
    centerFreq - span/2 + (i / numBins) * span
  );

  return { timestamp, centerFreq, span, numBins, psdDb, frequencies, payload };
}

export default function App() {
  const {
    status, setStatus,
    config, setConfig, updateConfig,
    spectrum, setSpectrum,
    detections, addDetection,
    setClusters
  } = useStore();

  // WebSocket for spectrum data
  useWebSocket({
    url: 'ws://localhost:8765/ws/spectrum',
    onBinaryMessage: (buffer) => {
      const data = parseSpectrumMessage(buffer);
      setSpectrum({
        frequencies: data.frequencies,
        psdDb: data.psdDb,
        timestamp: data.timestamp
      });

      // Update waterfall
      (window as any).__waterfallAddLine?.(data.payload);
    }
  });

  // WebSocket for detections
  useWebSocket({
    url: 'ws://localhost:8765/ws/detections',
    onJsonMessage: (msg) => {
      if (msg.type === 'detection') {
        addDetection(msg.data);
      }
    }
  });

  // WebSocket for clusters
  useWebSocket({
    url: 'ws://localhost:8765/ws/clusters',
    onJsonMessage: (msg) => {
      if (msg.type === 'clusters') {
        setClusters(msg.data);
      }
    }
  });

  // Fetch initial config and poll status
  useEffect(() => {
    api.getConfig().then(setConfig);

    const interval = setInterval(async () => {
      const status = await api.getStatus();
      setStatus(status);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const handleConfigChange = async (partial: Partial<Config>) => {
    updateConfig(partial);
    await api.updateConfig(partial);
  };

  const handleStart = async () => {
    await api.start();
  };

  const handleStop = async () => {
    await api.stop();
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-white">
      {/* Status Bar */}
      {status && <StatusBar status={status} />}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Control Panel */}
        {config && (
          <ControlPanel
            config={config}
            onConfigChange={handleConfigChange}
            onStart={handleStart}
            onStop={handleStop}
            isRunning={status?.state === 'running'}
          />
        )}

        {/* Main Display */}
        <div className="flex-1 flex flex-col p-4 gap-4 overflow-hidden">
          {/* Spectrum */}
          <div className="flex-shrink-0">
            {spectrum && (
              <SpectrumDisplay
                frequencies={spectrum.frequencies}
                psdDb={Array.from(spectrum.psdDb)}
              />
            )}
          </div>

          {/* Waterfall */}
          <div className="flex-1 min-h-0">
            <WaterfallDisplay
              width={800}
              height={300}
              numFreqBins={1024}
              numTimeLines={256}
            />
          </div>

          {/* Bottom Panel */}
          <div className="flex-shrink-0 h-64 bg-gray-900 rounded border border-gray-800">
            <DetectionsTable detections={detections} />
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## Performance Tips

1. **Waterfall Display**
   - Use WebGL for rendering (Canvas2D will not keep up at 30 fps)
   - Use `LUMINANCE` texture format (1 byte/pixel) not RGBA
   - Batch texture updates - don't call `texImage2D` more than 30x/sec
   - Consider using `texSubImage2D` to update only the new line

2. **Spectrum Chart**
   - Disable animations: `isAnimationActive={false}`
   - Use `throttle` or `requestAnimationFrame` to limit update rate
   - Consider using Canvas-based charting (e.g., `uPlot`) for better performance

3. **State Management**
   - Don't store raw spectrum data in React state if possible
   - Use refs for high-frequency data (spectrum, waterfall)
   - Only trigger re-renders for UI-relevant state changes

4. **WebSocket**
   - Parse binary messages efficiently (avoid creating intermediate objects)
   - Use `ArrayBuffer` and `DataView` for binary parsing
   - Consider using a Web Worker for parsing if needed

---

## Color Schemes

### Recommended Colormaps for Waterfall

| Name | Use Case |
|------|----------|
| **Viridis** | Default - perceptually uniform, colorblind-friendly |
| **Plasma** | High contrast for weak signals |
| **Turbo** | Maximum contrast (not colorblind-friendly) |
| **Grayscale** | Classic oscilloscope look |

### UI Color Palette (Tailwind)

```javascript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        // Signal strength colors
        'signal-strong': '#22c55e',  // green-500
        'signal-medium': '#eab308',  // yellow-500
        'signal-weak': '#6b7280',    // gray-500

        // Frequency markers
        'freq-marker': '#3b82f6',    // blue-500
        'freq-detection': '#ef4444', // red-500

        // Background
        'panel-bg': '#111827',       // gray-900
        'display-bg': '#030712',     // gray-950
      }
    }
  }
}
```

---

## Additional Component Specifications

### 5.6 ConstellationDiagram

The constellation diagram displays I/Q samples as a scatter plot, essential for modulation analysis.

#### Interface

```typescript
interface ConstellationDiagramProps {
  // I/Q sample data (interleaved: [I0, Q0, I1, Q1, ...])
  samples: Float32Array;

  // Display configuration
  width: number;
  height: number;

  // Visual options
  pointSize?: number;          // Default: 2
  pointColor?: string;         // Default: '#22c55e'
  backgroundColor?: string;    // Default: '#030712'
  gridColor?: string;          // Default: '#374151'

  // Axis range (symmetric: -range to +range)
  axisRange?: number;          // Default: 1.5

  // Optional: highlight decision regions
  showDecisionRegions?: boolean;
  modulationType?: 'BPSK' | 'QPSK' | '8PSK' | '16QAM' | '64QAM';

  // Performance
  maxPoints?: number;          // Default: 4096 (downsample if more)
  fadeTrail?: boolean;         // Show temporal fade effect
}
```

#### WebGL Implementation

```typescript
// src/components/ConstellationDiagram.tsx
import React, { useRef, useEffect, useCallback } from 'react';

interface ConstellationDiagramProps {
  samples: Float32Array;
  width: number;
  height: number;
  pointSize?: number;
  pointColor?: string;
  backgroundColor?: string;
  axisRange?: number;
  maxPoints?: number;
}

// Vertex shader for constellation points
const VERTEX_SHADER = `
  attribute vec2 a_position;
  attribute float a_age;

  uniform vec2 u_resolution;
  uniform float u_pointSize;
  uniform float u_axisRange;

  varying float v_age;

  void main() {
    // Normalize to clip space [-1, 1]
    vec2 normalized = a_position / u_axisRange;
    gl_Position = vec4(normalized, 0.0, 1.0);
    gl_PointSize = u_pointSize;
    v_age = a_age;
  }
`;

// Fragment shader with fade effect
const FRAGMENT_SHADER = `
  precision mediump float;

  uniform vec3 u_color;
  varying float v_age;

  void main() {
    // Circular point
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5) {
      discard;
    }

    // Fade based on age (1.0 = newest, 0.0 = oldest)
    float alpha = 0.3 + 0.7 * v_age;
    gl_FragColor = vec4(u_color, alpha);
  }
`;

export function ConstellationDiagram({
  samples,
  width,
  height,
  pointSize = 3,
  pointColor = '#22c55e',
  backgroundColor = '#030712',
  axisRange = 1.5,
  maxPoints = 4096,
}: ConstellationDiagramProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const bufferRef = useRef<WebGLBuffer | null>(null);

  // Parse hex color to RGB
  const parseColor = useCallback((hex: string): [number, number, number] => {
    const r = parseInt(hex.slice(1, 3), 16) / 255;
    const g = parseInt(hex.slice(3, 5), 16) / 255;
    const b = parseInt(hex.slice(5, 7), 16) / 255;
    return [r, g, b];
  }, []);

  // Initialize WebGL
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl', {
      alpha: false,
      antialias: true,
      preserveDrawingBuffer: false,
    });

    if (!gl) {
      console.error('WebGL not supported');
      return;
    }

    glRef.current = gl;

    // Compile shaders
    const vertShader = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vertShader, VERTEX_SHADER);
    gl.compileShader(vertShader);

    const fragShader = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fragShader, FRAGMENT_SHADER);
    gl.compileShader(fragShader);

    // Create program
    const program = gl.createProgram()!;
    gl.attachShader(program, vertShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    programRef.current = program;

    // Create buffer
    bufferRef.current = gl.createBuffer();

    // Enable blending for transparency
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    return () => {
      gl.deleteProgram(program);
      gl.deleteShader(vertShader);
      gl.deleteShader(fragShader);
      gl.deleteBuffer(bufferRef.current);
    };
  }, []);

  // Render constellation
  useEffect(() => {
    const gl = glRef.current;
    const program = programRef.current;
    const buffer = bufferRef.current;

    if (!gl || !program || !buffer || samples.length === 0) return;

    // Downsample if needed
    let pointCount = samples.length / 2;
    let stride = 1;
    if (pointCount > maxPoints) {
      stride = Math.ceil(pointCount / maxPoints);
      pointCount = Math.floor(pointCount / stride);
    }

    // Build vertex data: [x, y, age] for each point
    const vertexData = new Float32Array(pointCount * 3);
    for (let i = 0; i < pointCount; i++) {
      const srcIdx = i * stride * 2;
      vertexData[i * 3] = samples[srcIdx];       // I
      vertexData[i * 3 + 1] = samples[srcIdx + 1]; // Q
      vertexData[i * 3 + 2] = i / pointCount;    // Age (0 = oldest, 1 = newest)
    }

    // Set viewport
    gl.viewport(0, 0, width, height);

    // Clear with background color
    const [bgR, bgG, bgB] = parseColor(backgroundColor);
    gl.clearColor(bgR, bgG, bgB, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Use program
    gl.useProgram(program);

    // Upload vertex data
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertexData, gl.DYNAMIC_DRAW);

    // Set attributes
    const posLoc = gl.getAttribLocation(program, 'a_position');
    const ageLoc = gl.getAttribLocation(program, 'a_age');

    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 12, 0);

    gl.enableVertexAttribArray(ageLoc);
    gl.vertexAttribPointer(ageLoc, 1, gl.FLOAT, false, 12, 8);

    // Set uniforms
    const resLoc = gl.getUniformLocation(program, 'u_resolution');
    const sizeLoc = gl.getUniformLocation(program, 'u_pointSize');
    const rangeLoc = gl.getUniformLocation(program, 'u_axisRange');
    const colorLoc = gl.getUniformLocation(program, 'u_color');

    gl.uniform2f(resLoc, width, height);
    gl.uniform1f(sizeLoc, pointSize * window.devicePixelRatio);
    gl.uniform1f(rangeLoc, axisRange);
    gl.uniform3fv(colorLoc, parseColor(pointColor));

    // Draw points
    gl.drawArrays(gl.POINTS, 0, pointCount);

  }, [samples, width, height, pointSize, pointColor, backgroundColor, axisRange, maxPoints, parseColor]);

  // Draw grid overlay using Canvas 2D
  const drawGrid = useCallback((ctx: CanvasRenderingContext2D) => {
    const scale = width / (2 * axisRange);
    const centerX = width / 2;
    const centerY = height / 2;

    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;

    // Draw grid lines
    for (let i = -Math.floor(axisRange); i <= Math.floor(axisRange); i++) {
      const x = centerX + i * scale;
      const y = centerY - i * scale;

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;

    // X-axis
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, height);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px monospace';
    ctx.fillText('I', width - 15, centerY - 5);
    ctx.fillText('Q', centerX + 5, 15);
  }, [width, height, axisRange]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="rounded"
      />
      {/* Grid overlay rendered separately for clarity */}
      <canvas
        width={width}
        height={height}
        className="absolute top-0 left-0 pointer-events-none rounded"
        ref={(canvas) => {
          if (canvas) {
            const ctx = canvas.getContext('2d');
            if (ctx) drawGrid(ctx);
          }
        }}
      />
    </div>
  );
}
```

#### Usage Example

```tsx
function ModulationAnalysis() {
  const { selectedSignal } = useSignalStore();
  const [samples, setSamples] = useState<Float32Array>(new Float32Array());

  // Fetch I/Q samples for selected signal
  useEffect(() => {
    if (!selectedSignal) return;

    fetch(`/api/signals/${selectedSignal.id}/samples?count=2048`)
      .then(res => res.arrayBuffer())
      .then(buffer => setSamples(new Float32Array(buffer)));
  }, [selectedSignal]);

  return (
    <div className="bg-gray-900 p-4 rounded-lg">
      <h3 className="text-sm font-medium text-gray-400 mb-2">
        Constellation Diagram
      </h3>
      <ConstellationDiagram
        samples={samples}
        width={300}
        height={300}
        pointSize={2}
        axisRange={1.5}
      />
      {selectedSignal && (
        <div className="mt-2 text-xs text-gray-500">
          Detected: {selectedSignal.modulation || 'Unknown'}
        </div>
      )}
    </div>
  );
}
```

---

### 5.7 ClusterVisualization

Displays emitter clustering results from the GPU DBSCAN algorithm, showing relationships between detected signals.

#### Interface

```typescript
interface ClusterNode {
  id: string;
  label: string;
  frequency: number;
  bandwidth: number;
  signalStrength: number;
  clusterId: number;
  features: number[];  // Feature vector for positioning
}

interface ClusterVisualizationProps {
  nodes: ClusterNode[];
  width: number;
  height: number;

  // Interaction
  onNodeClick?: (node: ClusterNode) => void;
  onNodeHover?: (node: ClusterNode | null) => void;
  selectedNodeId?: string;

  // Visual options
  colorByCluster?: boolean;     // Default: true
  showLabels?: boolean;         // Default: true
  showConnections?: boolean;    // Default: true (within clusters)
  animateLayout?: boolean;      // Default: true

  // Layout
  layoutType?: '2d' | '3d';     // Default: '2d'
}
```

#### D3.js Force-Directed Implementation

```typescript
// src/components/ClusterVisualization.tsx
import React, { useRef, useEffect, useCallback } from 'react';
import * as d3 from 'd3';

interface ClusterNode {
  id: string;
  label: string;
  frequency: number;
  bandwidth: number;
  signalStrength: number;
  clusterId: number;
  features: number[];
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface ClusterLink {
  source: string;
  target: string;
  strength: number;
}

interface ClusterVisualizationProps {
  nodes: ClusterNode[];
  width: number;
  height: number;
  onNodeClick?: (node: ClusterNode) => void;
  selectedNodeId?: string;
  showLabels?: boolean;
  showConnections?: boolean;
}

// Cluster color palette
const CLUSTER_COLORS = [
  '#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16',
];

export function ClusterVisualization({
  nodes,
  width,
  height,
  onNodeClick,
  selectedNodeId,
  showLabels = true,
  showConnections = true,
}: ClusterVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<ClusterNode, ClusterLink> | null>(null);

  // Generate links between nodes in the same cluster
  const generateLinks = useCallback((nodes: ClusterNode[]): ClusterLink[] => {
    const links: ClusterLink[] = [];
    const clusterGroups = new Map<number, ClusterNode[]>();

    // Group by cluster
    nodes.forEach(node => {
      if (node.clusterId >= 0) {  // -1 = noise
        const group = clusterGroups.get(node.clusterId) || [];
        group.push(node);
        clusterGroups.set(node.clusterId, group);
      }
    });

    // Create links within clusters
    clusterGroups.forEach(group => {
      for (let i = 0; i < group.length; i++) {
        for (let j = i + 1; j < group.length; j++) {
          links.push({
            source: group[i].id,
            target: group[j].id,
            strength: 0.3,
          });
        }
      }
    });

    return links;
  }, []);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container groups
    const g = svg.append('g');
    const linksG = g.append('g').attr('class', 'links');
    const nodesG = g.append('g').attr('class', 'nodes');
    const labelsG = g.append('g').attr('class', 'labels');

    // Generate links
    const links = showConnections ? generateLinks(nodes) : [];

    // Create simulation
    const simulation = d3.forceSimulation<ClusterNode>(nodes)
      .force('link', d3.forceLink<ClusterNode, ClusterLink>(links)
        .id(d => d.id)
        .strength(d => d.strength))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    simulationRef.current = simulation;

    // Draw links
    const linkElements = linksG
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', '#374151')
      .attr('stroke-width', 1)
      .attr('stroke-opacity', 0.6);

    // Draw nodes
    const nodeElements = nodesG
      .selectAll('circle')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('r', d => 5 + Math.log(d.bandwidth + 1) * 3)
      .attr('fill', d => {
        if (d.clusterId < 0) return '#6b7280';  // Noise
        return CLUSTER_COLORS[d.clusterId % CLUSTER_COLORS.length];
      })
      .attr('stroke', d => d.id === selectedNodeId ? '#fff' : 'none')
      .attr('stroke-width', 2)
      .attr('cursor', 'pointer')
      .on('click', (event, d) => onNodeClick?.(d))
      .call(d3.drag<SVGCircleElement, ClusterNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }));

    // Draw labels
    const labelElements = showLabels ? labelsG
      .selectAll('text')
      .data(nodes)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', -15)
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text(d => d.label) : null;

    // Tooltip
    const tooltip = d3.select('body')
      .append('div')
      .attr('class', 'cluster-tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background', '#1f2937')
      .style('border', '1px solid #374151')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('color', '#e5e7eb');

    nodeElements
      .on('mouseover', (event, d) => {
        tooltip
          .style('visibility', 'visible')
          .html(`
            <strong>${d.label}</strong><br/>
            Freq: ${(d.frequency / 1e6).toFixed(3)} MHz<br/>
            BW: ${(d.bandwidth / 1e3).toFixed(1)} kHz<br/>
            Cluster: ${d.clusterId >= 0 ? d.clusterId : 'Noise'}
          `);
      })
      .on('mousemove', (event) => {
        tooltip
          .style('top', `${event.pageY - 10}px`)
          .style('left', `${event.pageX + 10}px`);
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden');
      });

    // Update positions on tick
    simulation.on('tick', () => {
      linkElements
        .attr('x1', d => (d.source as unknown as ClusterNode).x!)
        .attr('y1', d => (d.source as unknown as ClusterNode).y!)
        .attr('x2', d => (d.target as unknown as ClusterNode).x!)
        .attr('y2', d => (d.target as unknown as ClusterNode).y!);

      nodeElements
        .attr('cx', d => d.x!)
        .attr('cy', d => d.y!);

      if (labelElements) {
        labelElements
          .attr('x', d => d.x!)
          .attr('y', d => d.y!);
      }
    });

    // Zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    return () => {
      simulation.stop();
      tooltip.remove();
    };
  }, [nodes, width, height, onNodeClick, selectedNodeId, showLabels, showConnections, generateLinks]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="bg-gray-950 rounded"
    />
  );
}
```

#### WebSocket Integration for Real-Time Clusters

```typescript
// Hook for cluster WebSocket
function useClusterUpdates() {
  const [clusters, setClusters] = useState<ClusterNode[]>([]);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765/ws/clusters');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'cluster_update') {
        setClusters(data.nodes.map((n: any) => ({
          id: n.id,
          label: `Signal ${n.id.slice(0, 6)}`,
          frequency: n.center_freq,
          bandwidth: n.bandwidth,
          signalStrength: n.power_db,
          clusterId: n.cluster_id,
          features: n.features,
        })));
      }
    };

    return () => ws.close();
  }, []);

  return clusters;
}
```

---

### 5.8 DemodulationPanel

Control panel for protocol selection and display of demodulated data.

#### Interface

```typescript
interface DemodulationPanelProps {
  // Currently selected signal
  signalId?: string;
  signalInfo?: {
    frequency: number;
    bandwidth: number;
    modulation?: string;
  };

  // Demodulation state
  isDemodulating: boolean;
  demodulatedData?: DemodulatedOutput;

  // Callbacks
  onStartDemod: (protocol: string, options: DemodOptions) => void;
  onStopDemod: () => void;
}

interface DemodOptions {
  protocol: 'lora' | 'ble' | 'psk' | 'qam' | 'fsk' | 'auto';

  // LoRa specific
  loraSpreadingFactor?: number;  // 7-12
  loraBandwidth?: number;        // 125000, 250000, 500000
  loraCodingRate?: number;       // 1-4 (4/5 to 4/8)

  // PSK/QAM specific
  symbolRate?: number;
  constellation?: 'BPSK' | 'QPSK' | '8PSK' | '16QAM' | '64QAM';

  // BLE specific
  bleChannel?: number;           // 0-39
}

interface DemodulatedOutput {
  protocol: string;
  timestamp: number;

  // Raw data
  rawBits?: Uint8Array;
  rawBytes?: Uint8Array;

  // Decoded payload (protocol-specific)
  payload?: any;

  // Quality metrics
  snr?: number;
  ber?: number;
  evm?: number;

  // Display format
  hexDump?: string;
  asciiDump?: string;
}
```

#### Implementation

```tsx
// src/components/DemodulationPanel.tsx
import React, { useState, useCallback } from 'react';

interface DemodulationPanelProps {
  signalId?: string;
  signalInfo?: {
    frequency: number;
    bandwidth: number;
    modulation?: string;
  };
  isDemodulating: boolean;
  demodulatedData?: DemodulatedOutput;
  onStartDemod: (protocol: string, options: DemodOptions) => void;
  onStopDemod: () => void;
}

const PROTOCOLS = [
  { id: 'auto', name: 'Auto-Detect', icon: '🔍' },
  { id: 'lora', name: 'LoRa', icon: '📡' },
  { id: 'ble', name: 'Bluetooth LE', icon: '🔷' },
  { id: 'psk', name: 'PSK/QPSK', icon: '◐' },
  { id: 'qam', name: 'QAM', icon: '▦' },
  { id: 'fsk', name: 'FSK', icon: '〰' },
];

export function DemodulationPanel({
  signalId,
  signalInfo,
  isDemodulating,
  demodulatedData,
  onStartDemod,
  onStopDemod,
}: DemodulationPanelProps) {
  const [selectedProtocol, setSelectedProtocol] = useState('auto');
  const [loraSettings, setLoraSettings] = useState({
    spreadingFactor: 7,
    bandwidth: 125000,
    codingRate: 1,
  });
  const [pskSettings, setPskSettings] = useState({
    symbolRate: 9600,
    constellation: 'QPSK' as const,
  });

  const handleStart = useCallback(() => {
    const options: DemodOptions = {
      protocol: selectedProtocol as any,
      ...(selectedProtocol === 'lora' && {
        loraSpreadingFactor: loraSettings.spreadingFactor,
        loraBandwidth: loraSettings.bandwidth,
        loraCodingRate: loraSettings.codingRate,
      }),
      ...(selectedProtocol === 'psk' && {
        symbolRate: pskSettings.symbolRate,
        constellation: pskSettings.constellation,
      }),
    };
    onStartDemod(selectedProtocol, options);
  }, [selectedProtocol, loraSettings, pskSettings, onStartDemod]);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-200">Demodulation</h3>
        {signalInfo && (
          <span className="text-sm text-gray-400">
            {(signalInfo.frequency / 1e6).toFixed(3)} MHz
          </span>
        )}
      </div>

      {/* Protocol Selection */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        {PROTOCOLS.map(protocol => (
          <button
            key={protocol.id}
            onClick={() => setSelectedProtocol(protocol.id)}
            disabled={isDemodulating}
            className={`
              px-3 py-2 rounded text-sm font-medium transition-colors
              ${selectedProtocol === protocol.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }
              ${isDemodulating ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            <span className="mr-1">{protocol.icon}</span>
            {protocol.name}
          </button>
        ))}
      </div>

      {/* Protocol-specific settings */}
      {selectedProtocol === 'lora' && !isDemodulating && (
        <div className="mb-4 p-3 bg-gray-800 rounded">
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">SF</label>
              <select
                value={loraSettings.spreadingFactor}
                onChange={e => setLoraSettings(s => ({
                  ...s,
                  spreadingFactor: parseInt(e.target.value),
                }))}
                className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1 text-sm"
              >
                {[7, 8, 9, 10, 11, 12].map(sf => (
                  <option key={sf} value={sf}>SF{sf}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">BW</label>
              <select
                value={loraSettings.bandwidth}
                onChange={e => setLoraSettings(s => ({
                  ...s,
                  bandwidth: parseInt(e.target.value),
                }))}
                className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1 text-sm"
              >
                <option value={125000}>125 kHz</option>
                <option value={250000}>250 kHz</option>
                <option value={500000}>500 kHz</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">CR</label>
              <select
                value={loraSettings.codingRate}
                onChange={e => setLoraSettings(s => ({
                  ...s,
                  codingRate: parseInt(e.target.value),
                }))}
                className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1 text-sm"
              >
                <option value={1}>4/5</option>
                <option value={2}>4/6</option>
                <option value={3}>4/7</option>
                <option value={4}>4/8</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {selectedProtocol === 'psk' && !isDemodulating && (
        <div className="mb-4 p-3 bg-gray-800 rounded">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Symbol Rate</label>
              <input
                type="number"
                value={pskSettings.symbolRate}
                onChange={e => setPskSettings(s => ({
                  ...s,
                  symbolRate: parseInt(e.target.value),
                }))}
                className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1 text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Modulation</label>
              <select
                value={pskSettings.constellation}
                onChange={e => setPskSettings(s => ({
                  ...s,
                  constellation: e.target.value as any,
                }))}
                className="w-full bg-gray-700 text-gray-200 rounded px-2 py-1 text-sm"
              >
                <option value="BPSK">BPSK</option>
                <option value="QPSK">QPSK</option>
                <option value="8PSK">8PSK</option>
                <option value="16QAM">16-QAM</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Start/Stop Button */}
      <div className="mb-4">
        {!signalId ? (
          <div className="text-center text-gray-500 text-sm py-3">
            Select a signal to demodulate
          </div>
        ) : isDemodulating ? (
          <button
            onClick={onStopDemod}
            className="w-full py-2 bg-red-600 hover:bg-red-700 text-white rounded font-medium"
          >
            Stop Demodulation
          </button>
        ) : (
          <button
            onClick={handleStart}
            className="w-full py-2 bg-green-600 hover:bg-green-700 text-white rounded font-medium"
          >
            Start Demodulation
          </button>
        )}
      </div>

      {/* Demodulated Output */}
      {demodulatedData && (
        <div className="border-t border-gray-800 pt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-300">Output</span>
            <span className="text-xs text-gray-500">
              {new Date(demodulatedData.timestamp).toLocaleTimeString()}
            </span>
          </div>

          {/* Quality metrics */}
          {demodulatedData.snr !== undefined && (
            <div className="flex gap-4 mb-3 text-xs">
              <span className="text-gray-400">
                SNR: <span className="text-green-400">{demodulatedData.snr.toFixed(1)} dB</span>
              </span>
              {demodulatedData.ber !== undefined && (
                <span className="text-gray-400">
                  BER: <span className="text-yellow-400">{demodulatedData.ber.toExponential(2)}</span>
                </span>
              )}
            </div>
          )}

          {/* Hex dump */}
          {demodulatedData.hexDump && (
            <div className="bg-gray-950 p-3 rounded font-mono text-xs overflow-x-auto">
              <pre className="text-green-400 whitespace-pre-wrap break-all">
                {demodulatedData.hexDump}
              </pre>
            </div>
          )}

          {/* ASCII interpretation */}
          {demodulatedData.asciiDump && (
            <div className="mt-2 bg-gray-950 p-3 rounded font-mono text-xs">
              <pre className="text-blue-400 whitespace-pre-wrap">
                {demodulatedData.asciiDump}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

---

### 5.9 SignalInspector

Detailed analysis view for a selected signal, showing forensic analysis results.

#### Interface

```typescript
interface SignalInspectorProps {
  signal: {
    id: string;
    frequency: number;
    bandwidth: number;
    powerDb: number;
    firstSeen: number;
    lastSeen: number;
  };

  // Forensic analysis results (from backend)
  analysis?: {
    modulation: {
      detected: string;
      confidence: number;
      alternatives: Array<{ type: string; probability: number }>;
    };
    symbolRate: {
      estimated: number;
      confidence: number;
      method: 'squaring' | 'cyclostationary';
    };
    cumulants: {
      c20: { real: number; imag: number };
      c40: { real: number; imag: number };
      c42: { real: number; imag: number };
    };
    cyclicFeatures: {
      alphaValues: number[];
      peakAlpha: number;
      scdMagnitude: number[];
    };
  };

  // Sample data for visualization
  samples?: Float32Array;

  onClose: () => void;
}
```

#### Implementation

```tsx
// src/components/SignalInspector.tsx
import React, { useState, useEffect } from 'react';
import { ConstellationDiagram } from './ConstellationDiagram';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

interface SignalInspectorProps {
  signal: {
    id: string;
    frequency: number;
    bandwidth: number;
    powerDb: number;
    firstSeen: number;
    lastSeen: number;
  };
  analysis?: SignalAnalysis;
  samples?: Float32Array;
  onClose: () => void;
}

interface SignalAnalysis {
  modulation: {
    detected: string;
    confidence: number;
    alternatives: Array<{ type: string; probability: number }>;
  };
  symbolRate: {
    estimated: number;
    confidence: number;
    method: string;
  };
  cumulants: {
    c20: { real: number; imag: number };
    c40: { real: number; imag: number };
    c42: { real: number; imag: number };
  };
  cyclicFeatures: {
    alphaValues: number[];
    peakAlpha: number;
    scdMagnitude: number[];
  };
}

export function SignalInspector({
  signal,
  analysis,
  samples,
  onClose,
}: SignalInspectorProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'modulation' | 'cyclic'>('overview');

  const formatFreq = (hz: number) => {
    if (hz >= 1e9) return `${(hz / 1e9).toFixed(6)} GHz`;
    if (hz >= 1e6) return `${(hz / 1e6).toFixed(3)} MHz`;
    if (hz >= 1e3) return `${(hz / 1e3).toFixed(1)} kHz`;
    return `${hz.toFixed(0)} Hz`;
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  // Cyclic spectrum chart data
  const cyclicChartData = analysis?.cyclicFeatures?.alphaValues.map((alpha, i) => ({
    alpha,
    magnitude: analysis.cyclicFeatures.scdMagnitude[i] || 0,
  })) || [];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-900 rounded-lg border border-gray-700 w-[800px] max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <div>
            <h2 className="text-lg font-semibold text-gray-100">
              Signal Inspector
            </h2>
            <p className="text-sm text-gray-400">
              {formatFreq(signal.frequency)} • {formatFreq(signal.bandwidth)} BW
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-800 rounded transition-colors"
          >
            <span className="text-gray-400 text-xl">&times;</span>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-800">
          {(['overview', 'modulation', 'cyclic'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium capitalize transition-colors
                ${activeTab === tab
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-gray-200'
                }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'overview' && (
            <div className="grid grid-cols-2 gap-4">
              {/* Signal properties */}
              <div className="bg-gray-800 rounded p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Properties</h3>
                <dl className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Center Frequency</dt>
                    <dd className="text-gray-200">{formatFreq(signal.frequency)}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Bandwidth</dt>
                    <dd className="text-gray-200">{formatFreq(signal.bandwidth)}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Power</dt>
                    <dd className="text-gray-200">{signal.powerDb.toFixed(1)} dBm</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">Duration</dt>
                    <dd className="text-gray-200">
                      {formatDuration(signal.lastSeen - signal.firstSeen)}
                    </dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-500">First Seen</dt>
                    <dd className="text-gray-200">
                      {new Date(signal.firstSeen).toLocaleTimeString()}
                    </dd>
                  </div>
                </dl>
              </div>

              {/* Constellation */}
              <div className="bg-gray-800 rounded p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">Constellation</h3>
                {samples ? (
                  <ConstellationDiagram
                    samples={samples}
                    width={250}
                    height={250}
                    pointSize={2}
                  />
                ) : (
                  <div className="h-[250px] flex items-center justify-center text-gray-500">
                    No sample data available
                  </div>
                )}
              </div>

              {/* Quick stats */}
              {analysis && (
                <div className="col-span-2 bg-gray-800 rounded p-4">
                  <h3 className="text-sm font-medium text-gray-300 mb-3">Analysis Summary</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-400">
                        {analysis.modulation.detected}
                      </div>
                      <div className="text-xs text-gray-500">Modulation</div>
                      <div className="text-xs text-gray-400">
                        {(analysis.modulation.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400">
                        {(analysis.symbolRate.estimated / 1000).toFixed(1)}k
                      </div>
                      <div className="text-xs text-gray-500">Symbol Rate</div>
                      <div className="text-xs text-gray-400">symbols/sec</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-400">
                        {analysis.cyclicFeatures.peakAlpha.toFixed(3)}
                      </div>
                      <div className="text-xs text-gray-500">Peak Alpha</div>
                      <div className="text-xs text-gray-400">cyclic frequency</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'modulation' && analysis && (
            <div className="space-y-4">
              {/* Primary detection */}
              <div className="bg-gray-800 rounded p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Modulation Classification
                </h3>
                <div className="flex items-center gap-4 mb-4">
                  <div className="text-3xl font-bold text-blue-400">
                    {analysis.modulation.detected}
                  </div>
                  <div className="flex-1">
                    <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500"
                        style={{ width: `${analysis.modulation.confidence * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      {(analysis.modulation.confidence * 100).toFixed(1)}% confidence
                    </div>
                  </div>
                </div>

                {/* Alternatives */}
                <div className="space-y-2">
                  <div className="text-xs text-gray-500 uppercase tracking-wide">
                    Alternative Classifications
                  </div>
                  {analysis.modulation.alternatives.map((alt, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-sm text-gray-300 w-20">{alt.type}</span>
                      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gray-500"
                          style={{ width: `${alt.probability * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 w-12">
                        {(alt.probability * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Cumulants */}
              <div className="bg-gray-800 rounded p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Higher-Order Cumulants
                </h3>
                <div className="grid grid-cols-3 gap-4 font-mono text-sm">
                  <div>
                    <div className="text-gray-500 text-xs mb-1">C20</div>
                    <div className="text-gray-200">
                      {analysis.cumulants.c20.real.toFixed(4)} + j{analysis.cumulants.c20.imag.toFixed(4)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs mb-1">C40</div>
                    <div className="text-gray-200">
                      {analysis.cumulants.c40.real.toFixed(4)} + j{analysis.cumulants.c40.imag.toFixed(4)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs mb-1">C42</div>
                    <div className="text-gray-200">
                      {analysis.cumulants.c42.real.toFixed(4)} + j{analysis.cumulants.c42.imag.toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Symbol Rate */}
              <div className="bg-gray-800 rounded p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Symbol Rate Estimation
                </h3>
                <div className="flex items-center gap-4">
                  <div className="text-2xl font-bold text-green-400">
                    {analysis.symbolRate.estimated.toLocaleString()}
                  </div>
                  <div className="text-gray-400">symbols/second</div>
                </div>
                <div className="text-xs text-gray-500 mt-2">
                  Method: {analysis.symbolRate.method} •
                  Confidence: {(analysis.symbolRate.confidence * 100).toFixed(0)}%
                </div>
              </div>
            </div>
          )}

          {activeTab === 'cyclic' && analysis && (
            <div className="space-y-4">
              {/* Cyclic Spectrum Display */}
              <div className="bg-gray-800 rounded p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-3">
                  Spectral Correlation Density
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cyclicChartData}>
                      <XAxis
                        dataKey="alpha"
                        stroke="#6b7280"
                        fontSize={10}
                        tickFormatter={v => v.toFixed(2)}
                      />
                      <YAxis
                        stroke="#6b7280"
                        fontSize={10}
                        tickFormatter={v => v.toFixed(0)}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1f2937',
                          border: '1px solid #374151',
                          borderRadius: '4px',
                        }}
                        labelStyle={{ color: '#9ca3af' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="magnitude"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="text-xs text-gray-500 mt-2">
                  Peak cyclic frequency: {analysis.cyclicFeatures.peakAlpha.toFixed(4)} Hz
                </div>
              </div>

              {/* Explanation */}
              <div className="bg-gray-800 rounded p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-2">
                  About Cyclostationary Analysis
                </h3>
                <p className="text-sm text-gray-400">
                  The spectral correlation density (SCD) reveals hidden periodicities
                  in the signal caused by modulation. Peaks in the cyclic frequency
                  (alpha) domain correspond to symbol rate harmonics and carrier
                  frequencies, enabling blind parameter estimation.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-800 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
```

---

### 5.10 RecordingControls

I/Q recording management interface for capturing raw samples to disk.

#### Interface

```typescript
interface RecordingControlsProps {
  // Current recording state
  isRecording: boolean;
  recordingInfo?: {
    filename: string;
    startTime: number;
    sampleCount: number;
    fileSize: number;
    format: 'cf32' | 'ci16' | 'cu8';
  };

  // Available recordings
  recordings: RecordingFile[];

  // Callbacks
  onStartRecording: (options: RecordingOptions) => void;
  onStopRecording: () => void;
  onDeleteRecording: (filename: string) => void;
  onDownloadRecording: (filename: string) => void;
  onPlaybackRecording: (filename: string) => void;
}

interface RecordingOptions {
  filename?: string;           // Auto-generated if not provided
  format: 'cf32' | 'ci16' | 'cu8';
  maxDuration?: number;        // Seconds (0 = unlimited)
  maxSize?: number;            // Bytes (0 = unlimited)

  // Optional frequency/bandwidth override
  centerFreq?: number;
  bandwidth?: number;
}

interface RecordingFile {
  filename: string;
  createdAt: number;
  size: number;
  duration: number;
  sampleRate: number;
  centerFreq: number;
  format: string;
}
```

#### Implementation

```tsx
// src/components/RecordingControls.tsx
import React, { useState, useCallback } from 'react';

interface RecordingControlsProps {
  isRecording: boolean;
  recordingInfo?: {
    filename: string;
    startTime: number;
    sampleCount: number;
    fileSize: number;
    format: string;
  };
  recordings: RecordingFile[];
  onStartRecording: (options: RecordingOptions) => void;
  onStopRecording: () => void;
  onDeleteRecording: (filename: string) => void;
  onDownloadRecording: (filename: string) => void;
  onPlaybackRecording: (filename: string) => void;
}

interface RecordingFile {
  filename: string;
  createdAt: number;
  size: number;
  duration: number;
  sampleRate: number;
  centerFreq: number;
  format: string;
}

interface RecordingOptions {
  filename?: string;
  format: 'cf32' | 'ci16' | 'cu8';
  maxDuration?: number;
  maxSize?: number;
}

const FORMATS = [
  { id: 'cf32', name: 'Complex Float32', desc: '8 bytes/sample, highest precision' },
  { id: 'ci16', name: 'Complex Int16', desc: '4 bytes/sample, good precision' },
  { id: 'cu8', name: 'Complex Uint8', desc: '2 bytes/sample, RTL-SDR compatible' },
];

export function RecordingControls({
  isRecording,
  recordingInfo,
  recordings,
  onStartRecording,
  onStopRecording,
  onDeleteRecording,
  onDownloadRecording,
  onPlaybackRecording,
}: RecordingControlsProps) {
  const [showNewRecording, setShowNewRecording] = useState(false);
  const [format, setFormat] = useState<'cf32' | 'ci16' | 'cu8'>('cf32');
  const [maxDuration, setMaxDuration] = useState<number>(0);
  const [filename, setFilename] = useState('');

  const formatFileSize = (bytes: number): string => {
    if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
    if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
    if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStart = useCallback(() => {
    onStartRecording({
      format,
      maxDuration: maxDuration > 0 ? maxDuration : undefined,
      filename: filename || undefined,
    });
    setShowNewRecording(false);
    setFilename('');
  }, [format, maxDuration, filename, onStartRecording]);

  // Live recording elapsed time
  const [elapsedTime, setElapsedTime] = useState(0);
  React.useEffect(() => {
    if (!isRecording || !recordingInfo) return;

    const interval = setInterval(() => {
      setElapsedTime(Date.now() - recordingInfo.startTime);
    }, 100);

    return () => clearInterval(interval);
  }, [isRecording, recordingInfo]);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-lg font-medium text-gray-200">I/Q Recordings</h3>
        {!isRecording && (
          <button
            onClick={() => setShowNewRecording(true)}
            className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors flex items-center gap-1"
          >
            <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
            New Recording
          </button>
        )}
      </div>

      {/* Active Recording */}
      {isRecording && recordingInfo && (
        <div className="p-4 bg-red-900/20 border-b border-red-800/50">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
              <span className="text-red-400 font-medium">Recording</span>
            </div>
            <button
              onClick={onStopRecording}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded"
            >
              Stop
            </button>
          </div>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-gray-500">Filename</div>
              <div className="text-gray-200 truncate">{recordingInfo.filename}</div>
            </div>
            <div>
              <div className="text-gray-500">Duration</div>
              <div className="text-gray-200 font-mono">
                {formatDuration(elapsedTime / 1000)}
              </div>
            </div>
            <div>
              <div className="text-gray-500">Size</div>
              <div className="text-gray-200">{formatFileSize(recordingInfo.fileSize)}</div>
            </div>
          </div>
        </div>
      )}

      {/* New Recording Form */}
      {showNewRecording && !isRecording && (
        <div className="p-4 bg-gray-800/50 border-b border-gray-800">
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Filename (optional)
              </label>
              <input
                type="text"
                value={filename}
                onChange={e => setFilename(e.target.value)}
                placeholder="Auto-generated timestamp"
                className="w-full bg-gray-700 text-gray-200 rounded px-3 py-2 text-sm"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Format</label>
              <div className="grid grid-cols-3 gap-2">
                {FORMATS.map(f => (
                  <button
                    key={f.id}
                    onClick={() => setFormat(f.id as any)}
                    className={`p-2 rounded text-left transition-colors ${
                      format === f.id
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    <div className="text-sm font-medium">{f.id.toUpperCase()}</div>
                    <div className="text-xs opacity-75">{f.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Max Duration (seconds, 0 = unlimited)
              </label>
              <input
                type="number"
                value={maxDuration}
                onChange={e => setMaxDuration(parseInt(e.target.value) || 0)}
                min={0}
                className="w-32 bg-gray-700 text-gray-200 rounded px-3 py-2 text-sm"
              />
            </div>

            <div className="flex gap-2 pt-2">
              <button
                onClick={handleStart}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm font-medium"
              >
                Start Recording
              </button>
              <button
                onClick={() => setShowNewRecording(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Recording List */}
      <div className="max-h-64 overflow-y-auto">
        {recordings.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            No recordings yet
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="bg-gray-800 sticky top-0">
              <tr className="text-left text-gray-400 text-xs uppercase tracking-wide">
                <th className="px-4 py-2">Filename</th>
                <th className="px-4 py-2">Duration</th>
                <th className="px-4 py-2">Size</th>
                <th className="px-4 py-2">Freq</th>
                <th className="px-4 py-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {recordings.map(rec => (
                <tr
                  key={rec.filename}
                  className="border-t border-gray-800 hover:bg-gray-800/50"
                >
                  <td className="px-4 py-2">
                    <div className="text-gray-200 truncate max-w-[150px]">
                      {rec.filename}
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(rec.createdAt).toLocaleDateString()}
                    </div>
                  </td>
                  <td className="px-4 py-2 text-gray-300 font-mono">
                    {formatDuration(rec.duration)}
                  </td>
                  <td className="px-4 py-2 text-gray-300">
                    {formatFileSize(rec.size)}
                  </td>
                  <td className="px-4 py-2 text-gray-300">
                    {(rec.centerFreq / 1e6).toFixed(1)} MHz
                  </td>
                  <td className="px-4 py-2">
                    <div className="flex gap-1">
                      <button
                        onClick={() => onPlaybackRecording(rec.filename)}
                        className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-green-400"
                        title="Playback"
                      >
                        ▶
                      </button>
                      <button
                        onClick={() => onDownloadRecording(rec.filename)}
                        className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-blue-400"
                        title="Download"
                      >
                        ⬇
                      </button>
                      <button
                        onClick={() => {
                          if (confirm(`Delete ${rec.filename}?`)) {
                            onDeleteRecording(rec.filename);
                          }
                        }}
                        className="p-1 hover:bg-gray-700 rounded text-gray-400 hover:text-red-400"
                        title="Delete"
                      >
                        ✕
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
```

#### API Endpoints for Recording

```typescript
// Recording REST API endpoints (backend reference)
// GET  /api/recordings           - List all recordings
// POST /api/recordings/start     - Start new recording
// POST /api/recordings/stop      - Stop current recording
// GET  /api/recordings/:filename - Download recording file
// DELETE /api/recordings/:filename - Delete recording

// Example hook implementation
function useRecordings() {
  const [recordings, setRecordings] = useState<RecordingFile[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingInfo, setRecordingInfo] = useState<RecordingInfo | null>(null);

  // Fetch recordings list
  const fetchRecordings = useCallback(async () => {
    const res = await fetch('/api/recordings');
    const data = await res.json();
    setRecordings(data.recordings);
  }, []);

  // Start recording
  const startRecording = useCallback(async (options: RecordingOptions) => {
    const res = await fetch('/api/recordings/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options),
    });
    const data = await res.json();
    setIsRecording(true);
    setRecordingInfo(data);
  }, []);

  // Stop recording
  const stopRecording = useCallback(async () => {
    await fetch('/api/recordings/stop', { method: 'POST' });
    setIsRecording(false);
    setRecordingInfo(null);
    fetchRecordings();  // Refresh list
  }, [fetchRecordings]);

  return {
    recordings,
    isRecording,
    recordingInfo,
    startRecording,
    stopRecording,
    fetchRecordings,
  };
}
```

---

## Docker Deployment

For Docker containerization of the frontend, see the following files in `/rf_forensics/frontend/`:

- `Dockerfile` - Multi-stage build with nginx
- `docker-compose.yml` - Full stack deployment (frontend + GPU backend)
- `nginx.conf` - Production configuration with WebSocket proxy
- `.dockerignore` - Build context optimization

### Quick Start

```bash
# Build and start all services
cd rf_forensics/frontend
docker-compose up -d --build

# Access the application
# Frontend: http://localhost:3000
# API: http://localhost:8000
# WebSocket: ws://localhost:8765

# Verify GPU access
docker-compose exec api nvidia-smi

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Questions?

Contact the backend team for:
- WebSocket message format changes
- New API endpoints
- Custom binary protocols
- GPU memory optimization

Good luck!
