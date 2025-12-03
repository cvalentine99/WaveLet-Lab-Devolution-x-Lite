# Sprint 7: Port 8765 REST API Fix

## Problem Statement

The frontend was configured to communicate with the backend on **port 8765** for both:
- REST API calls (`/api/sdr/*`, `/api/status`, `/health`, etc.)
- WebSocket connections (`/ws/spectrum`, `/ws/detections`, etc.)

However, the backend had a **dual-port architecture**:
- Port 8000: REST API only
- Port 8765: WebSocket only

This caused the frontend to receive `404 Not Found` or empty `{"paths": {}}` when calling REST endpoints on port 8765.

## Root Cause

In `api/websocket_server.py`, the `create_websocket_app()` function only registered WebSocket endpoints:

```python
# BEFORE (broken)
def create_websocket_app(server: SpectrumWebSocketServer) -> "FastAPI":
    app = FastAPI(title="RF Forensics WebSocket API")

    # Only WebSocket endpoints were registered
    @app.websocket("/ws/spectrum")
    async def spectrum_ws(websocket: WebSocket):
        ...
```

No REST routers were included, so port 8765 had **0 REST paths**.

## Solution

### 1. Added REST Routers to WebSocket App

Modified `api/websocket_server.py` to include all REST routers:

```python
# AFTER (fixed)
def create_websocket_app(server: SpectrumWebSocketServer, api_manager=None) -> "FastAPI":
    app = FastAPI(title="RF Forensics WebSocket API")

    # Store API manager for dependency injection
    if api_manager is not None:
        app.state.api_manager = api_manager

    # Include REST API routers so frontend can use port 8765 for everything
    from rf_forensics.api.routers import (
        system_router, config_router, detections_router,
        recordings_router, sdr_router, analysis_router,
    )
    app.include_router(system_router)
    app.include_router(config_router)
    app.include_router(detections_router)
    app.include_router(recordings_router)
    app.include_router(sdr_router)
    app.include_router(analysis_router)

    # WebSocket endpoints follow...
```

### 2. Updated run_server.py to Share api_manager

Modified `scripts/run_server.py` to pass `api_manager` to the WebSocket app:

```python
# Create shared API manager
api_manager = RFForensicsAPI()

# Create FastAPI apps with shared state
rest_app = create_rest_api(api_manager)
ws_app = create_websocket_app(ws_server, api_manager)  # Share api_manager
```

## Files Modified

| File | Change |
|------|--------|
| `api/websocket_server.py` | Added `api_manager` parameter, included 6 REST routers |
| `scripts/run_server.py` | Pass `api_manager` to `create_websocket_app()` |

## Verification

After the fix, port 8765 serves **42 REST paths** plus WebSocket endpoints:

```bash
# REST endpoints now work on 8765
curl http://localhost:8765/health
# {"status":"ok","gpu_available":true,...}

curl http://localhost:8765/api/sdr/devices
# {"devices":[]}

curl http://localhost:8765/api/status
# {"state":"idle","uptime_seconds":...}

curl http://localhost:8765/api/sdr/config
# {"device_type":"usdr","center_freq_hz":915000000,...}

# WebSocket endpoints also work
# ws://localhost:8765/ws/spectrum - CONNECTED
# ws://localhost:8765/ws/detections - CONNECTED
# ws://localhost:8765/ws/clusters - CONNECTED
# ws://localhost:8765/ws/iq - CONNECTED
```

## Architecture After Fix

```
Port 8765 (WebSocket App)
├── REST Endpoints (via routers)
│   ├── /health
│   ├── /api/status
│   ├── /api/start, /api/stop, /api/pause
│   ├── /api/config
│   ├── /api/sdr/devices, /connect, /config, /frequency, /bands
│   ├── /api/detections, /api/clusters
│   └── /api/recordings/*
│
└── WebSocket Endpoints
    ├── /ws/spectrum (binary)
    ├── /ws/detections (JSON)
    ├── /ws/clusters (JSON)
    └── /ws/iq (binary)

Port 8000 (REST App) - Still available for direct REST access
```

## Frontend Configuration

Frontend should connect to **port 8765 only**:

```typescript
// Frontend config
const BACKEND_URL = 'http://localhost:8765';
const WS_URL = 'ws://localhost:8765';
```

## Date

Fixed: 2025-12-03
