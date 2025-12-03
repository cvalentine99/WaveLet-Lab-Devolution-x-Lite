#!/bin/bash
# ============================================================================
# FRONTEND ENDPOINT VERIFICATION SCRIPT
# ============================================================================
# This script tests EVERY endpoint the frontend needs to call.
# Use this as a reference for wiring up UI buttons.
#
# REST Base URL:      http://localhost:8000
# WebSocket Base URL: ws://localhost:8765
# ============================================================================

BASE_URL="http://localhost:8000"
WS_URL="ws://localhost:8765"

# The device ID returned from discovery - use this in stream start/stop
DEVICE_ID="bus=pci,device=usdr0"

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║          FRONTEND → BACKEND ENDPOINT VERIFICATION                       ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║  REST API:    http://localhost:8000                                     ║"
echo "║  WebSockets:  ws://localhost:8765                                       ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# STEP 1: LISTEN PAGE LOADS - GET DEVICES FOR DROPDOWN
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 1: PAGE LOAD - Populate SDR Dropdown                                │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: Listen page mounts / useEffect on load                             │"
echo "│ BUTTON: None - automatic on page load                                    │"
echo "│ ENDPOINT: GET /api/sdr/devices                                           │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "FRONTEND CODE:"
echo "─────────────────────────────────────────────────────────────────────────────"
cat << 'FRONTEND_CODE'
// In your Listen page component (e.g., Listen.tsx or SDRSelector.tsx)
useEffect(() => {
  const fetchDevices = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/sdr/devices');
      const data = await response.json();
      // data.devices is an array of {id, model, serial, status}
      setDevices(data.devices);
      setDropdownOptions(data.devices.map(d => ({
        value: d.id,
        label: `${d.model} (${d.id})`
      })));
    } catch (error) {
      console.error('Failed to fetch SDR devices:', error);
    }
  };
  fetchDevices();
}, []);
FRONTEND_CODE
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "CURL TEST:"
curl -s "${BASE_URL}/api/sdr/devices" | python3 -m json.tool
echo ""
echo "EXPECTED RESPONSE SHAPE:"
cat << 'EXPECTED'
{
  "devices": [
    {
      "id": "bus=pci,device=usdr0",   // USE THIS ID FOR STREAM START/STOP
      "model": "uSDR DevBoard",
      "serial": "",
      "status": "available"           // "available" | "connected" | "in_use"
    }
  ]
}
EXPECTED
echo ""
echo ""

# ============================================================================
# STEP 2: USER SELECTS SDR FROM DROPDOWN
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 2: USER SELECTS SDR - Enable Start Button                           │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: User picks an SDR from dropdown                                    │"
echo "│ BUTTON: Dropdown onChange                                                │"
echo "│ ENDPOINT: NONE - just update React state                                 │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "FRONTEND CODE:"
echo "─────────────────────────────────────────────────────────────────────────────"
cat << 'FRONTEND_CODE'
// SDR Dropdown onChange handler
const handleSDRSelect = (selectedDeviceId: string) => {
  setSelectedDeviceId(selectedDeviceId);  // e.g., "bus=pci,device=usdr0"
  setStartButtonEnabled(true);             // Enable the "Start Stream" button
  setIsStreaming(false);                   // Not streaming yet
};

// JSX
<Select
  options={dropdownOptions}
  onChange={(e) => handleSDRSelect(e.target.value)}
  placeholder="Select SDR Device..."
/>
<Button
  disabled={!startButtonEnabled || isStreaming}
  onClick={handleStartStream}
>
  Start Stream
</Button>
FRONTEND_CODE
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "NO API CALL - Frontend state only. DO NOT auto-connect or auto-stream!"
echo ""
echo ""

# ============================================================================
# STEP 3: USER CLICKS "START STREAM" BUTTON
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 3: START STREAM BUTTON CLICKED                                      │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: User clicks 'Start Stream' button                                  │"
echo "│ BUTTON: Start Stream                                                     │"
echo "│ ENDPOINT: POST /api/sdr/devices/{deviceId}/stream/start                  │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "FRONTEND CODE:"
echo "─────────────────────────────────────────────────────────────────────────────"
cat << 'FRONTEND_CODE'
// Start Stream button onClick handler
const handleStartStream = async () => {
  if (!selectedDeviceId) return;

  try {
    setIsLoading(true);

    // IMPORTANT: URL encode the device ID if it contains special chars
    const encodedDeviceId = encodeURIComponent(selectedDeviceId);

    const response = await fetch(
      `http://localhost:8000/api/sdr/devices/${encodedDeviceId}/stream/start`,
      { method: 'POST' }
    );

    const data = await response.json();

    if (data.success) {
      setIsStreaming(true);
      setStartButtonEnabled(false);
      setStopButtonEnabled(true);

      // NOW connect WebSockets for real-time data
      connectWebSockets();
    } else {
      console.error('Failed to start stream:', data);
      alert('Failed to start stream');
    }
  } catch (error) {
    console.error('Start stream error:', error);
  } finally {
    setIsLoading(false);
  }
};
FRONTEND_CODE
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "CURL TEST:"
# URL encode the device ID for curl
ENCODED_DEVICE_ID=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${DEVICE_ID}', safe=''))")
curl -s -X POST "${BASE_URL}/api/sdr/devices/${ENCODED_DEVICE_ID}/stream/start" | python3 -m json.tool
echo ""
echo "EXPECTED RESPONSE:"
cat << 'EXPECTED'
{
  "success": true,
  "device_id": "bus=pci,device=usdr0",
  "streaming": true,
  "message": "Streaming started"
}
EXPECTED
echo ""
echo ""

# ============================================================================
# STEP 4: CONNECT WEBSOCKETS FOR REAL-TIME DATA
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 4: CONNECT WEBSOCKETS (after stream starts)                         │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: Immediately after stream/start returns success                     │"
echo "│ ENDPOINTS:                                                               │"
echo "│   ws://localhost:8765/ws/spectrum     - FFT spectrum data (binary)       │"
echo "│   ws://localhost:8765/ws/detections   - Signal detections (JSON)         │"
echo "│   ws://localhost:8765/ws/clusters     - Clustered signals (JSON)         │"
echo "│   ws://localhost:8765/ws/iq           - Raw IQ samples (binary)          │"
echo "│   ws://localhost:8765/ws/demodulation - Demodulated audio (binary)       │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "FRONTEND CODE:"
echo "─────────────────────────────────────────────────────────────────────────────"
cat << 'FRONTEND_CODE'
// WebSocket connection function - call this AFTER stream starts
const connectWebSockets = () => {
  // Spectrum WebSocket (for waterfall/FFT display)
  const spectrumWs = new WebSocket('ws://localhost:8765/ws/spectrum');
  spectrumWs.binaryType = 'arraybuffer';  // IMPORTANT: spectrum is binary

  spectrumWs.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      // Binary spectrum data
      // Header: 20 bytes (timestamp: f64, center_freq: f32, span: f32, bins: u16, flags: u16)
      // Data: Float32 array of power values in dBFS
      const view = new DataView(event.data);
      const timestamp = view.getFloat64(0, true);
      const centerFreq = view.getFloat32(8, true);
      const span = view.getFloat32(12, true);
      const numBins = view.getUint16(16, true);

      const powerData = new Float32Array(event.data.slice(20));
      updateWaterfall(powerData, centerFreq, span);
    }
  };

  // Detections WebSocket (for signal list)
  const detectionsWs = new WebSocket('ws://localhost:8765/ws/detections');

  detectionsWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data.type === 'detection'
    // data.data contains: detection_id, center_freq_hz, bandwidth_hz,
    //   peak_power_db, snr_db, modulation_type, confidence, timestamp, etc.
    addDetection(data.data);
  };

  // Clusters WebSocket (for grouped signals)
  const clustersWs = new WebSocket('ws://localhost:8765/ws/clusters');

  clustersWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // data.type === 'cluster'
    // data.data contains: cluster_id, center_freq_hz, signals, anomaly_score, etc.
    updateClusters(data.data);
  };

  // Store refs for cleanup
  wsRefs.current = { spectrumWs, detectionsWs, clustersWs };
};

// Cleanup on unmount or stop
const disconnectWebSockets = () => {
  Object.values(wsRefs.current).forEach(ws => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
  });
};
FRONTEND_CODE
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "WEBSOCKET TEST (Python):"
cat << 'PYTEST'
# Quick WebSocket test
import asyncio
import websockets

async def test_spectrum():
    async with websockets.connect('ws://localhost:8765/ws/spectrum') as ws:
        data = await ws.recv()
        print(f"Received {len(data)} bytes of spectrum data")

asyncio.run(test_spectrum())
PYTEST
echo ""
echo ""

# ============================================================================
# STEP 5: USER CLICKS "STOP STREAM" BUTTON
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 5: STOP STREAM BUTTON CLICKED                                       │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: User clicks 'Stop Stream' button                                   │"
echo "│ BUTTON: Stop Stream                                                      │"
echo "│ ENDPOINT: POST /api/sdr/devices/{deviceId}/stream/stop                   │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "FRONTEND CODE:"
echo "─────────────────────────────────────────────────────────────────────────────"
cat << 'FRONTEND_CODE'
// Stop Stream button onClick handler
const handleStopStream = async () => {
  if (!selectedDeviceId) return;

  try {
    setIsLoading(true);

    // Close WebSockets FIRST
    disconnectWebSockets();

    const encodedDeviceId = encodeURIComponent(selectedDeviceId);

    const response = await fetch(
      `http://localhost:8000/api/sdr/devices/${encodedDeviceId}/stream/stop`,
      { method: 'POST' }
    );

    const data = await response.json();

    if (data.success) {
      setIsStreaming(false);
      setStartButtonEnabled(true);   // Re-enable start button
      setStopButtonEnabled(false);
      // Dropdown stays enabled - user can switch SDRs
    }
  } catch (error) {
    console.error('Stop stream error:', error);
  } finally {
    setIsLoading(false);
  }
};
FRONTEND_CODE
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "CURL TEST:"
curl -s -X POST "${BASE_URL}/api/sdr/devices/${ENCODED_DEVICE_ID}/stream/stop" | python3 -m json.tool
echo ""
echo "EXPECTED RESPONSE:"
cat << 'EXPECTED'
{
  "success": true,
  "device_id": "bus=pci,device=usdr0",
  "streaming": false,
  "message": "Streaming stopped"
}
EXPECTED
echo ""
echo ""

# ============================================================================
# STEP 6: FREQUENCY CONTROL (Slider or Input)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 6: FREQUENCY CONTROL                                                │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: User changes frequency slider/input                                │"
echo "│ CONTROL: Frequency slider, input box, or tuning dial                     │"
echo "│ ENDPOINT: POST /api/sdr/frequency                                        │"
echo "│ BODY: { \"centerFreqHz\": 915000000 }  (camelCase for frontend)           │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "FRONTEND CODE:"
echo "─────────────────────────────────────────────────────────────────────────────"
cat << 'FRONTEND_CODE'
// Frequency change handler (debounce this!)
const handleFrequencyChange = async (freqHz: number) => {
  try {
    const response = await fetch('http://localhost:8000/api/sdr/frequency', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ centerFreqHz: freqHz })  // camelCase!
    });

    const data = await response.json();
    if (data.success) {
      setCurrentFrequency(data.freq_hz);
    }
  } catch (error) {
    console.error('Frequency change error:', error);
  }
};

// With debouncing (recommended for sliders)
const debouncedFreqChange = useMemo(
  () => debounce(handleFrequencyChange, 100),
  []
);
FRONTEND_CODE
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "CURL TEST:"
curl -s -X POST "${BASE_URL}/api/sdr/frequency" \
  -H "Content-Type: application/json" \
  -d '{"centerFreqHz": 915000000}' | python3 -m json.tool
echo ""
echo "EXPECTED RESPONSE:"
cat << 'EXPECTED'
{
  "success": true,
  "freq_hz": 915000000
}
EXPECTED
echo ""
echo ""

# ============================================================================
# STEP 7: GAIN CONTROL
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 7: GAIN CONTROL                                                     │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: User adjusts gain slider(s)                                        │"
echo "│ CONTROL: Gain slider or input                                            │"
echo "│ ENDPOINT: POST /api/sdr/gain                                             │"
echo "│ BODY: { \"lna_db\": 15, \"tia_db\": 9, \"pga_db\": 12 }                      │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "FRONTEND CODE:"
echo "─────────────────────────────────────────────────────────────────────────────"
cat << 'FRONTEND_CODE'
// Gain change handler
const handleGainChange = async (lna: number, tia: number, pga: number) => {
  try {
    const response = await fetch('http://localhost:8000/api/sdr/gain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        lna_db: lna,   // 0-30 dB
        tia_db: tia,   // 0, 3, 9, or 12 dB
        pga_db: pga    // 0-32 dB
      })
    });

    const data = await response.json();
    if (data.success) {
      setGainSettings(data.gain);
    }
  } catch (error) {
    console.error('Gain change error:', error);
  }
};

// Or if you have a single "total gain" slider:
const handleTotalGainChange = async (totalDb: number) => {
  // Split total gain across stages (example)
  const lna = Math.min(30, totalDb);
  const remaining = totalDb - lna;
  const tia = Math.min(12, remaining);
  const pga = Math.min(32, remaining - tia);

  await handleGainChange(lna, tia, pga);
};
FRONTEND_CODE
echo "─────────────────────────────────────────────────────────────────────────────"
echo ""
echo "CURL TEST:"
curl -s -X POST "${BASE_URL}/api/sdr/gain" \
  -H "Content-Type: application/json" \
  -d '{"lna_db": 15, "tia_db": 9, "pga_db": 12}' | python3 -m json.tool
echo ""
echo ""

# ============================================================================
# STEP 8: BAND SELECTOR (DevBoard Duplexer)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 8: BAND SELECTOR                                                    │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: Page load (populate dropdown) or user selects band                 │"
echo "│ GET BANDS: GET /api/sdr/bands                                            │"
echo "│ SET BAND:  POST /api/sdr/bands  { \"band\": \"gsm900\" }                    │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "CURL TEST - Get available bands:"
curl -s "${BASE_URL}/api/sdr/bands" | python3 -m json.tool
echo ""
echo "CURL TEST - Set band:"
curl -s -X POST "${BASE_URL}/api/sdr/bands" \
  -H "Content-Type: application/json" \
  -d '{"band": "gsm900"}' | python3 -m json.tool
echo ""
echo ""

# ============================================================================
# STEP 9: GET CURRENT CONFIG (for displaying in UI)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 9: GET CURRENT CONFIG                                               │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: Page load, after changes, or to sync UI                            │"
echo "│ ENDPOINT: GET /api/sdr/config                                            │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "CURL TEST:"
curl -s "${BASE_URL}/api/sdr/config" | python3 -m json.tool
echo ""
echo ""

# ============================================================================
# STEP 10: FULL CONFIG UPDATE (Advanced Settings Panel)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 10: FULL CONFIG UPDATE                                              │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: User submits full config form                                      │"
echo "│ ENDPOINT: POST /api/sdr/config                                           │"
echo "│ ACCEPTS BOTH FORMATS:                                                    │"
echo "│   - camelCase (frontend): centerFreqHz, sampleRateHz, gainDb             │"
echo "│   - snake_case (backend): center_freq_hz, sample_rate_hz, gain.lna_db    │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "FRONTEND FORMAT (camelCase):"
curl -s -X POST "${BASE_URL}/api/sdr/config" \
  -H "Content-Type: application/json" \
  -d '{
    "centerFreqHz": 915000000,
    "sampleRateHz": 10000000,
    "gainDb": 36
  }' | python3 -m json.tool
echo ""
echo ""

# ============================================================================
# STEP 11: GET SDR STATUS (for status indicators)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 11: GET SDR STATUS                                                  │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: Polling for status updates, or after actions                       │"
echo "│ ENDPOINT: GET /api/sdr/status                                            │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "CURL TEST:"
curl -s "${BASE_URL}/api/sdr/status" | python3 -m json.tool
echo ""
echo ""

# ============================================================================
# STEP 12: GET SDR HEALTH (for health indicators)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 12: GET SDR HEALTH                                                  │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: Dashboard health display, monitoring                               │"
echo "│ ENDPOINT: GET /api/sdr/health                                            │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "CURL TEST:"
curl -s "${BASE_URL}/api/sdr/health" | python3 -m json.tool
echo ""
echo ""

# ============================================================================
# STEP 13: GET SYSTEM STATUS (pipeline state)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐"
echo "│ STEP 13: GET SYSTEM STATUS                                               │"
echo "├──────────────────────────────────────────────────────────────────────────┤"
echo "│ WHEN: Dashboard, monitoring pipeline state                               │"
echo "│ ENDPOINT: GET /api/status                                                │"
echo "└──────────────────────────────────────────────────────────────────────────┘"
echo ""
echo "CURL TEST:"
curl -s "${BASE_URL}/api/status" | python3 -m json.tool
echo ""
echo ""

# ============================================================================
# SUMMARY: BUTTON → ENDPOINT MAPPING
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                    BUTTON → ENDPOINT QUICK REFERENCE                    ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║                                                                         ║"
echo "║  PAGE LOAD (Listen):                                                    ║"
echo "║    → GET /api/sdr/devices                                               ║"
echo "║    → Populate dropdown with response.devices                            ║"
echo "║                                                                         ║"
echo "║  DROPDOWN SELECT:                                                       ║"
echo "║    → NO API CALL - just setSelectedDeviceId(id)                         ║"
echo "║    → Enable 'Start Stream' button                                       ║"
echo "║                                                                         ║"
echo "║  START STREAM BUTTON:                                                   ║"
echo "║    → POST /api/sdr/devices/{id}/stream/start                            ║"
echo "║    → Then connect WebSockets                                            ║"
echo "║                                                                         ║"
echo "║  STOP STREAM BUTTON:                                                    ║"
echo "║    → Close WebSockets first                                             ║"
echo "║    → POST /api/sdr/devices/{id}/stream/stop                             ║"
echo "║                                                                         ║"
echo "║  FREQUENCY SLIDER:                                                      ║"
echo "║    → POST /api/sdr/frequency {centerFreqHz: value}                      ║"
echo "║                                                                         ║"
echo "║  GAIN SLIDER:                                                           ║"
echo "║    → POST /api/sdr/gain {lna_db, tia_db, pga_db}                        ║"
echo "║                                                                         ║"
echo "║  BAND SELECTOR:                                                         ║"
echo "║    → GET /api/sdr/bands (populate options)                              ║"
echo "║    → POST /api/sdr/bands {band: 'gsm900'}                               ║"
echo "║                                                                         ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║                         WEBSOCKET URLS                                  ║"
echo "╠════════════════════════════════════════════════════════════════════════╣"
echo "║  Spectrum (FFT):     ws://localhost:8765/ws/spectrum      [BINARY]      ║"
echo "║  Detections:         ws://localhost:8765/ws/detections    [JSON]        ║"
echo "║  Clusters:           ws://localhost:8765/ws/clusters      [JSON]        ║"
echo "║  Raw IQ:             ws://localhost:8765/ws/iq            [BINARY]      ║"
echo "║  Demodulation:       ws://localhost:8765/ws/demodulation  [BINARY]      ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Environment variables the frontend should use:"
echo "  VITE_BACKEND_URL=http://localhost:8000"
echo "  VITE_BACKEND_WS_URL=ws://localhost:8765"
echo ""
echo "Done. Wire up these endpoints to your buttons!"
