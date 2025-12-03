# RF Forensics Architecture & Solutions Guide

This document explains how the RF Forensics GPU pipeline works, the threading/async model, and the CuPy compatibility issues that were resolved.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Main Process                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  REST API       │    │  WebSocket      │    │  Async Event    │         │
│  │  (FastAPI)      │    │  Server         │    │  Loop           │         │
│  │  Port 8000      │    │  Port 8765      │    │  (uvicorn)      │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           └──────────────────────┼──────────────────────┘                   │
│                                  │                                          │
│                    ┌─────────────▼─────────────┐                           │
│                    │  PipelineWebSocketBridge  │                           │
│                    │  (Sync → Async adapter)   │                           │
│                    └─────────────┬─────────────┘                           │
│                                  │                                          │
│  ════════════════════════════════╪══════════════════════════════════════   │
│                    THREAD BOUNDARY (callbacks cross here)                   │
│  ════════════════════════════════╪══════════════════════════════════════   │
│                                  │                                          │
│                    ┌─────────────▼─────────────┐                           │
│                    │  RFForensicsPipeline      │ ◄── Processing Thread     │
│                    │  (orchestrator.py)        │                           │
│                    └─────────────┬─────────────┘                           │
│                                  │                                          │
│           ┌──────────────────────┼──────────────────────┐                  │
│           │                      │                      │                   │
│  ┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐          │
│  │  SDR Manager    │   │  GPU Ring       │   │  CUDA Stream    │          │
│  │  (usdr driver)  │   │  Buffer         │   │  Manager        │          │
│  └────────┬────────┘   └────────┬────────┘   └─────────────────┘          │
│           │                     │                                          │
│           │              ┌──────▼──────┐                                   │
│           │              │  GPU DSP    │                                   │
│           │              │  Pipeline   │                                   │
│           │              └──────┬──────┘                                   │
│           │                     │                                          │
│  ┌────────▼────────┐   ┌───────▼───────┐   ┌─────────────────┐            │
│  │  SDR Callback   │   │  PSD → CFAR   │   │  Clustering     │            │
│  │  Thread         │   │  → Peaks      │   │  (cuML DBSCAN)  │            │
│  └─────────────────┘   └───────────────┘   └─────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Threading Model

### Three Execution Contexts

1. **Async Event Loop (Main Thread)**
   - Runs FastAPI REST endpoints
   - Runs WebSocket server
   - All `async def` functions execute here
   - Managed by uvicorn

2. **Pipeline Processing Thread**
   - Runs `_processing_loop()` in `orchestrator.py`
   - Pulls samples from GPU ring buffer
   - Executes PSD, CFAR, peak detection, clustering
   - Fires callbacks when results are ready

3. **SDR Callback Thread**
   - Created by libusdr driver
   - Receives raw IQ samples from hardware
   - Pushes samples into GPU ring buffer

### The Callback Problem

When the pipeline processing thread has results (spectrum, detections, etc.), it needs to notify the WebSocket server to broadcast to clients. But:

- The pipeline runs in a **synchronous thread**
- The WebSocket server runs in an **async event loop**
- You cannot call `async` functions from sync code directly
- You cannot use `asyncio.create_task()` from a non-async thread

### The Solution: Sync Wrappers with Thread-Safe Dispatch

```python
# In PipelineWebSocketBridge (pipeline_integration.py)

def _get_loop(self):
    """Get event loop, caching it for efficiency."""
    if self._loop is None or not self._loop.is_running():
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
    return self._loop

def on_detection_sync(self, detections: List):
    """Called from pipeline thread - dispatches to async context."""
    loop = self._get_loop()
    if loop:
        asyncio.run_coroutine_threadsafe(self.on_detection(detections), loop)

async def on_detection(self, detections: List):
    """Runs in async context - can await WebSocket sends."""
    await self._ws_server.broadcast_detection(detections)
```

**Key insight**: `asyncio.run_coroutine_threadsafe()` is the bridge between threads and async. It schedules a coroutine on an event loop from any thread.

---

## CuPy vs NumPy Compatibility

### The Confusion

CuPy is designed to be a drop-in replacement for NumPy with identical APIs. However:

1. **Implicit conversion is forbidden**: CuPy arrays cannot be silently converted to NumPy
2. **Some NumPy functions don't exist in CuPy**: e.g., `np.divide(..., where=mask)`
3. **cuML returns cupy arrays**: Need explicit `.get()` to convert to numpy
4. **Import order matters**: If cuML import fails but cupy succeeded, `cp = np` overwrites the real cupy

### Problems Encountered

#### 1. The `where=` Parameter

```python
# BROKEN - CuPy doesn't support 'where' parameter
cp.divide(psd, noise_estimate, out=self._snr_out, where=noise_estimate > 0)

# FIXED - Use safe division instead
safe_noise = cp.maximum(noise_estimate, 1e-12)
cp.divide(psd, safe_noise, out=self._snr_out)
```

#### 2. Implicit Array Conversion

```python
# BROKEN - iterating over cupy array triggers implicit conversion
for label in set(self._labels):  # self._labels is cupy array
    ...

# FIXED - explicitly convert first
if hasattr(self._labels, 'get'):
    self._labels = self._labels.get()
unique_labels = set(self._labels)  # now it's numpy
```

#### 3. Import Order Bug

```python
# BROKEN - if cuML fails, cp becomes np but cupy was already imported
try:
    import cupy as cp
    from cuml.cluster import DBSCAN  # This might fail!
    CUML_AVAILABLE = True
except ImportError:
    cp = np  # Overwrites working cupy import!

# FIXED - separate imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False

try:
    from cuml.cluster import DBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False
```

#### 4. cuML Return Types

cuML can return:
- CuPy arrays
- cuDF Series (pandas-like on GPU)
- NumPy arrays (sometimes)

```python
# FIXED - handle all cases
if hasattr(self._labels, 'get'):
    self._labels = self._labels.get()  # cupy array
elif hasattr(self._labels, 'values'):
    # cuDF Series
    self._labels = self._labels.values.get()
if not isinstance(self._labels, np.ndarray):
    self._labels = np.asarray(self._labels)
```

---

## Data Flow

### Sample Path (SDR → GPU → Results)

```
1. SDR Hardware
   └─► libusdr callback (SDR thread)
       └─► PinnedMemoryManager.get_buffer()
           └─► H2D async copy to GPU ring buffer
               └─► Processing thread picks up segment
                   ├─► WelchPSD.compute_psd() [cuFFT]
                   ├─► CFARDetector.detect() [Numba CUDA kernel]
                   ├─► PeakDetector.find_peaks()
                   ├─► EmitterClusterer.fit() [cuML DBSCAN]
                   └─► D2H async copy of results
                       └─► Callbacks fire with numpy arrays
                           └─► run_coroutine_threadsafe()
                               └─► WebSocket broadcast (async)
```

### Memory Management

```
┌──────────────────────────────────────────────────────────────┐
│                        CPU Memory                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Pinned Memory Pool (PinnedMemoryManager)              │  │
│  │  - Pre-allocated page-locked buffers                   │  │
│  │  - Enables async DMA transfers                         │  │
│  │  - Double-buffered for H2D overlap                     │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ PCIe (async memcpy)
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        GPU Memory                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  GPU Ring Buffer (GPURingBuffer)                       │  │
│  │  - Circular buffer of IQ samples                       │  │
│  │  - Segment-based processing                            │  │
│  │  - Backpressure via fill level monitoring              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Pre-allocated Work Buffers                            │  │
│  │  - FFT workspace (cuFFT plans cached)                  │  │
│  │  - PSD output buffer                                   │  │
│  │  - CFAR threshold/mask buffers                         │  │
│  │  - Detection feature arrays                            │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Backpressure Handling

When the GPU can't keep up with the SDR sample rate:

```
Buffer Fill Level
     │
100% ┤ ████████████████████  DROP SAMPLES
     │
 85% ┤ ─────────────────────  high_watermark (start throttling)
     │ ████████████████
     │ ████████████
     │ ████████
 50% ┤ ─────────────────────  low_watermark (stop throttling)
     │ ████
     │
  0% ┤
     └──────────────────────► Time
```

1. **Buffer fill > 85%**: Set `_throttle_sdr = True`, SDR callback drops samples
2. **Buffer fill < 50%**: Clear throttle flag, resume normal operation
3. **Hysteresis prevents oscillation** between states

---

## Docker Development Setup

The `docker-compose.yml` mounts the source directory for live code updates:

```yaml
volumes:
  - ./rf_forensics:/app  # Source mount for development
  - rf-recordings:/app/recordings
  - rf-logs:/app/logs
```

This means:
- Code changes on host are immediately visible in container
- No rebuild needed for Python changes
- Just restart the container: `docker compose restart backend`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `pipeline/orchestrator.py` | Main processing loop, callback dispatch |
| `api/pipeline_integration.py` | Sync→Async bridge for WebSocket |
| `api/websocket_server.py` | WebSocket broadcast implementation |
| `core/ring_buffer.py` | GPU ring buffer with backpressure |
| `core/async_transfer.py` | Non-blocking D2H transfers |
| `dsp/psd.py` | Welch PSD with cuFFT |
| `detection/cfar.py` | CFAR detector (Numba CUDA) |
| `detection/peaks.py` | Peak extraction and characterization |
| `ml/clustering.py` | cuML DBSCAN clustering |
| `sdr/usdr_driver.py` | libusdr ctypes wrapper |

---

## Debugging Tips

### Check Pipeline Status
```bash
curl http://localhost:8000/api/status
```

### View Container Logs
```bash
docker logs rf-forensics-backend --tail 50
```

### Restart with Code Changes
```bash
docker compose restart backend
```

### Profile with Nsight Systems
```bash
docker exec rf-forensics-backend \
  nsys profile --trace=cuda,nvtx \
  python -m rf_forensics.profiling.run_profiling --trace
```

### Common Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| "Implicit conversion to NumPy" | CuPy array used where numpy expected | Add `.get()` call |
| "no running event loop" | Async called from sync thread | Use `run_coroutine_threadsafe()` |
| "Buffer fill 87%" | Processing slower than ingest | Reduce sample rate or optimize |
| EBUSY on SDR connect | Device already open | Stop other processes using SDR |

---

## Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Ingest Rate | 50 MSPS | 10-20 MSPS |
| Frame Latency | <10 ms | 1-4 ms |
| GPU Utilization | >70% | 40-60% |
| Drop Rate | <0.1% | 0% at 10 MSPS |
| Buffer Fill | <50% | 0-20% |
