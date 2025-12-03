# LiveMonitoring Page - Control Topology

## Status: DOCUMENTED (2025-12-03)

---

## HEADER CONTROLS (`LiveMonitoring.tsx`)

| Control | Handler | API Call |
|---------|---------|----------|
| **Start Button** | `handleStart()` | `POST /api/start` + WebSocket streams |
| **Stop Button** | `handleStop()` | `POST /api/stop` + stop streams |
| **Save Button** | `handleSave()` | `POST /api/recordings/start` → 5s → `stop` |
| **Alerts Button** | `setShowThresholdAlerts(true)` | Opens dialog |
| **Capture Button** | `captureSpectrumSnapshot()` | Opens bookmark dialog |
| **Report Button** | `setShowReportGenerator(true)` | Opens dialog |
| **Settings Button** | `setShowSettings(toggle)` | Toggle sidebar |

---

## KEYBOARD SHORTCUTS

| Key | Action |
|-----|--------|
| `Space` | Start/Stop pipeline |
| `r` | Toggle recording |
| `s` | Toggle settings panel |
| `p/d/c` | Switch tabs (Perf/Detections/Clusters) |
| `?` | Show shortcuts |
| `Shift+P` | Toggle Peak Hold |
| `Shift+A` | Toggle Average |
| `m` | Add marker |
| `Shift+M` | Clear markers |

---

## UnifiedSDRPanel CONTROLS

### Device Connection
| Control | API Call |
|---------|----------|
| **Device Dropdown** | Local state |
| **Refresh** | `GET /api/sdr/devices` |
| **Connect** | `POST /api/sdr/connect` |
| **Disconnect** | `POST /api/sdr/disconnect` |

### TUNING Section
| Control | API Call |
|---------|----------|
| Preset/Freq/Rate/BW | Local state |
| **Apply Tuning** | `POST /api/sdr/config` |

### RF FRONTEND Section
| Control | API Call |
|---------|----------|
| LNA/PA/Atten controls | Local state |
| **Apply Frontend** | `PUT /api/config` (devboard) |

### RECEIVER Section (LMS7002M)
| Control | API Call |
|---------|----------|
| RX Path/LNA/TIA/PGA | Local state |
| **Apply Receiver** | `POST /api/sdr/gain` + `POST /api/sdr/rx_path` |

### PERIPHERALS Section
| Control | API Call |
|---------|----------|
| VCTCXO/OSC/GPS/etc | Local state |
| **Apply Peripherals** | `PUT /api/config` (devboard) |

---

## EnhancedSpectrumDisplay CONTROLS

| Control | Action |
|---------|--------|
| Spectrum/Persistence/IQ tabs | Switch view |
| Clear buttons | Clear display |
| Settings popover | Decay rate |
| IQ Tab active | `startIQStream()` |

---

## FileManager CONTROLS

| Control | API Call |
|---------|----------|
| **Refresh** | `GET /api/recordings` |
| **Download** | `GET /api/recordings/{id}/download` |
| **Delete** | `DELETE /api/recordings/{id}` |

---

## WEBSOCKET STREAMS

| Stream | Endpoint | Store |
|--------|----------|-------|
| Spectrum | `/ws/spectrum` | `useSpectrumStore` |
| Detections | `/ws/detections` | `useDetectionStore` |
| Clusters | `/ws/clusters` | `useClusterStore` |
| Demodulation | `/ws/demodulation` | Callbacks |
| IQ | `/ws/iq` | `useSpectrumStore` |

---

## INITIAL DATA FETCH (on mount)

```typescript
fetchInitialData() {
  GET /api/detections → useDetectionStore
  GET /api/clusters → useClusterStore
}
```

---

## POLLING LOOPS

| Loop | Interval | API |
|------|----------|-----|
| System Status | 2000ms | `GET /api/status` |
| WS Connections | 1000ms | `areStreamsConnected()` |
| SDR Hook | 1000ms | `GET /api/sdr/status` |

---

## KEY FILES

- `client/src/pages/LiveMonitoring.tsx` - Main page
- `client/src/components/controls/UnifiedSDRPanel.tsx` - SDR controls
- `client/src/components/controls/LiveControlBar.tsx` - Start/Stop/Save
- `client/src/components/spectrum/EnhancedSpectrumDisplay.tsx` - Spectrum viz
- `client/src/services/websocket.ts` - WS streams + initial fetch
- `client/src/stores/*.ts` - Zustand stores
- `client/src/lib/api.ts` - REST API client
