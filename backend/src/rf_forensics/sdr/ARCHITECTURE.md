# SDR Architecture - Unified Driver Ownership

## Overview

This document describes the SDR subsystem architecture after the hardening changes
implemented to fix critical issues identified during PCIe USDR DevBoard testing.

## Problem Statement

Hardware testing revealed several critical architectural issues:

| Issue | Severity | Description |
|-------|----------|-------------|
| Driver Dual Ownership | CRITICAL | Pipeline created NEW `USDRDriver()`, API used singleton |
| No Authentication | CRITICAL | Anyone could control SDR via `/api/sdr/*` |
| CORS Wildcard | HIGH | `allow_origins=["*"]` with credentials |
| Missing Observability | HIGH | Overflow/drops counted but not exposed |
| No Backpressure | HIGH | Samples dropped silently when buffer full |
| No Reconnection | HIGH | Manual intervention required on disconnect |
| Placeholder Device | MEDIUM | Fake device inserted when none found |

## Architecture Solution

### Before: Dual Ownership (BROKEN)

```
┌─────────────────────┐     ┌─────────────────────┐
│   API Router        │     │   Pipeline          │
│   (singleton)       │     │   (new instance)    │
│                     │     │                     │
│ get_usdr_driver() ──┼──→  │ USDRInterface()     │
│     ↓               │     │     ↓               │
│ _driver_instance    │     │ self._driver =      │
│                     │     │   USDRDriver() ←NEW!│
└─────────────────────┘     └─────────────────────┘
         │                           │
         └─────────┬─────────────────┘
                   ↓
            SAME HARDWARE
           (conflict/race)
```

### After: SDRManager Singleton (FIXED)

```
┌─────────────────────────────────────────────────┐
│              SDRManager (singleton)              │
│  ─────────────────────────────────────────────  │
│  - Thread-safe singleton with RLock             │
│  - Owns the single USDRDriver instance          │
│  - Provides proxy for pipeline                  │
│  - Exposes metrics & state to API               │
│  - Handles reconnection & backpressure          │
└─────────────────────────────────────────────────┘
         ↑                        ↑
         │                        │
    ┌────┴────┐              ┌────┴────┐
    │   API   │              │Pipeline │
    │ Router  │              │         │
    └─────────┘              └─────────┘
```

## Component Details

### SDRManager (`sdr/manager.py`)

Thread-safe singleton that owns the SDR driver:

```python
from rf_forensics.sdr.manager import get_sdr_manager

# Get the singleton (always same instance)
manager = get_sdr_manager()

# Connect to SDR
devices = manager.discover()
manager.connect(devices[0].id)

# Configure
manager.configure(USDRConfig(
    center_freq_hz=915_000_000,
    sample_rate_hz=10_000_000,
))

# Start streaming
manager.start_streaming(my_callback)

# Get metrics
metrics = manager.get_metrics()
print(f"Overflows: {metrics.total_overflows}")
print(f"Temperature: {metrics.temperature_c}°C")
```

**Features:**
- Thread-safe singleton with `threading.Lock`
- RLock for driver operations (reentrant)
- Configuration persistence for auto-reconnection
- Metrics integration with `MetricsTracker`
- Buffer fill tracking for backpressure signaling
- Async auto-reconnection loop

### MetricsTracker (`sdr/metrics.py`)

Real-time metrics with rolling window calculations:

```python
from rf_forensics.sdr.metrics import MetricsTracker

tracker = MetricsTracker(window_seconds=10.0)
tracker.record_overflow(count=5)
tracker.record_samples(received=1000, dropped=10)

metrics = tracker.metrics
print(f"Overflow rate: {metrics.overflow_rate_per_sec}/s")
print(f"Drop rate: {metrics.drop_rate_percent}%")
```

**Tracked Metrics:**
- `total_overflows` - Hardware buffer overflows
- `overflow_rate_per_sec` - Rolling window overflow rate
- `total_samples_received` - Total samples from SDR
- `total_samples_dropped` - Samples dropped due to backpressure
- `drop_rate_percent` - Percentage of samples dropped
- `temperature_c` - Hardware temperature
- `pll_locked` - PLL lock status
- `streaming_uptime_seconds` - Time since streaming started
- `reconnect_count` - Number of auto-reconnections
- `backpressure_events` - Times buffer exceeded 75% fill

### API Endpoints

New observability endpoints in `/api/sdr/`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Real-time SDR hardware metrics |
| `/capabilities` | GET | Hardware capability ranges |
| `/health` | GET | Health status with warnings |

**Example `/api/sdr/metrics` response:**
```json
{
  "overflow": {
    "total": 0,
    "rate_per_sec": 0.0,
    "last_timestamp": null
  },
  "samples": {
    "total_received": 10000000,
    "total_dropped": 0,
    "drop_rate_percent": 0.0
  },
  "hardware": {
    "temperature_c": 45.2,
    "pll_locked": true,
    "actual_sample_rate_hz": 10000000,
    "actual_freq_hz": 915000000
  },
  "streaming": {
    "uptime_seconds": 120.5,
    "reconnect_count": 0,
    "last_error": null
  },
  "backpressure": {
    "events": 0,
    "buffer_fill_percent": 25.0
  }
}
```

**Example `/api/sdr/health` response:**
```json
{
  "status": "healthy",
  "connected": true,
  "streaming": true,
  "warnings": [],
  "metrics_summary": {
    "overflow_rate": 0.0,
    "drop_rate_percent": 0.0,
    "buffer_fill_percent": 25.0,
    "temperature_c": 45.2,
    "uptime_seconds": 120.5
  }
}
```

### Ring Buffer Backpressure (`core/ring_buffer.py`)

Buffer fill tracking with backpressure events:

```python
buffer = GPURingBuffer(num_segments=8, samples_per_segment=500000)

# Backpressure tracked automatically in try_get_write_segment()
# Events logged when fill > 75%

status = buffer.get_status()
print(f"Fill level: {status['fill_level']*100:.1f}%")
print(f"Backpressure events: {status['backpressure_events']}")
```

## Security Hardening

### CORS Whitelist

Changed from wildcard to whitelist:

```python
# Before (INSECURE)
allow_origins=["*"]

# After (SECURE)
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173"
).split(",")
```

**Configure via environment:**
```bash
export CORS_ORIGINS="https://myapp.com,https://admin.myapp.com"
```

## File Changes Summary

| File | Change |
|------|--------|
| `sdr/manager.py` | NEW - SDRManager singleton |
| `sdr/metrics.py` | NEW - Metrics tracking |
| `sdr/__init__.py` | Updated exports |
| `sdr/usdr_driver.py` | Removed placeholder, added overflow tracking |
| `sdr/interface.py` | Uses SDRManager |
| `api/routers/sdr.py` | Uses SDRManager, new endpoints |
| `api/rest_api.py` | CORS whitelist |
| `pipeline/orchestrator.py` | Uses SDRManager |
| `core/ring_buffer.py` | Backpressure tracking |

## Migration Guide

### For API Users

No changes required - endpoints remain the same, but now have:
- Better error handling
- Metrics endpoints
- Health monitoring

### For Pipeline Users

Replace direct driver usage:

```python
# Before
from rf_forensics.sdr.usdr_driver import get_usdr_driver
driver = get_usdr_driver()

# After
from rf_forensics.sdr.manager import get_sdr_manager
manager = get_sdr_manager()
driver = manager.get_driver()  # If you need raw driver access
```

### For Custom Integrations

Use the manager for all SDR operations:

```python
from rf_forensics.sdr import get_sdr_manager, USDRConfig, USDRGain

manager = get_sdr_manager()

# Configure
config = USDRConfig(
    center_freq_hz=915_000_000,
    sample_rate_hz=10_000_000,
    gain=USDRGain(lna_db=15, tia_db=9, pga_db=12)
)
manager.configure(config)

# Stream with metrics
def my_callback(samples, timestamp):
    # Process samples
    pass

manager.start_streaming(my_callback)

# Monitor health
while True:
    metrics = manager.get_metrics()
    if metrics.overflow_rate_per_sec > 1.0:
        print("WARNING: High overflow rate!")
    time.sleep(1)
```

## Future Work

- [ ] GPUDirect zero-copy path (nvidia-peermem)
- [ ] API key authentication
- [ ] Rate limiting on config endpoints
- [ ] PLL lock status from hardware
- [ ] Automatic gain control
