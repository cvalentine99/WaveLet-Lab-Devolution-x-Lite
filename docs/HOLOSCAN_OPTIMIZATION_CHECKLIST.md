# Holoscan GPU Optimization Checklist

Focus areas for NVIDIA/Holoscan team on hot path and GPU utilization.

## 1. Validate True Async D2H

**Current state:** `core/async_transfer.py` uses `memcpyAsync` with pinned buffers and double-buffering.

**Validation tasks:**
- [ ] Profile with Nsight Systems under 50 MSPS load
- [ ] Verify D2H stream timeline overlaps with PSD/CFAR compute stream
- [ ] Check for hidden synchronization points (CuPy implicit syncs)
- [ ] Measure actual overlap ratio vs theoretical

```bash
nsys profile --trace=cuda,nvtx python -m rf_forensics.pipeline.orchestrator
```

**Key metrics:**
- D2H copy duration vs compute duration
- Stream concurrency percentage
- Any unexpected gaps in timeline

---

## 2. SDR Ingest Path (H2D)

**Current state:** Pinned buffers via `PinnedMemoryManager`, ring buffer with async H2D on dedicated stream.

**Optimization tasks:**
- [ ] Check uSDR PCIe driver for GPUDirect RDMA support
- [ ] If no GPUDirect: optimize H2D staging
  - Chunk size tuning (current: `samples_per_buffer` from config)
  - Stream priority for H2D vs compute
  - Double-buffer H2D to overlap with compute
- [ ] Profile PCIe bandwidth utilization

**GPUDirect check:**
```python
# If uSDR supports it:
# samples -> GPU memory directly (no CPU staging)
# Requires nvidia-peermem kernel module
```

---

## 3. Kernel Efficiency

### cuFFT Plan Reuse
**Location:** `dsp/psd.py` - `WelchPSD`

- [ ] Verify FFT plan is created once and reused
- [ ] Check plan cache for different input sizes
- [ ] Consider `cufftSetAutoAllocation(plan, false)` for workspace control

### CFAR Kernel
**Location:** `detection/cfar.py` - `CFARDetector`

- [ ] Profile occupancy (target >50%)
- [ ] Check memory throughput vs theoretical bandwidth
- [ ] Consider shared memory for reference cell windows
- [ ] Evaluate kernel fusion: PSD → log10 → CFAR in single kernel

### Peak Detection
**Location:** `detection/peaks.py` - `PeakDetector`

- [ ] Profile CuPy/Numba kernels for peak extraction
- [ ] Consider warp-level primitives for reduction
- [ ] Batch peak output to minimize kernel launches

**Profiling:**
```bash
ncu --set full python -m rf_forensics.pipeline.orchestrator
```

---

## 4. GPU Clustering/Features

**Current state:** Features extracted on CPU, clustering via `EmitterClusterer` (likely sklearn DBSCAN).

**GPU migration tasks:**
- [ ] Move feature extraction to GPU (keep features as CuPy arrays)
- [ ] Replace sklearn DBSCAN with cuML DBSCAN
- [ ] Tune DBSCAN parameters for RF detection data:
  - `eps`: frequency/bandwidth scale
  - `min_samples`: minimum cluster size
  - Consider HDBSCAN for varying density

```python
from cuml.cluster import DBSCAN as cuDBSCAN

# GPU-native clustering
clusterer = cuDBSCAN(eps=0.1, min_samples=5)
labels = clusterer.fit_predict(features_gpu)  # No D2H needed
```

- [ ] Only transfer cluster summaries to CPU (not per-detection labels)

---

## 5. Holoscan Operator Graph

**Target architecture:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ SDR Ingest  │───>│ PSD Compute │───>│ CFAR Detect │
│ (H2D async) │    │ (cuFFT)     │    │ (GPU)       │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ WebSocket   │<───│ Clustering  │<───│ Peak Extract│
│ Output      │    │ (cuML)      │    │ + Tracking  │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Holoscan tasks:**
- [ ] Map each stage to Holoscan Operator
- [ ] Define explicit CUDA streams per operator
- [ ] Configure QoS policies:
  - Input queue depths
  - Backpressure thresholds
  - Drop policies (oldest vs newest)
- [ ] Set latency budget constraints
- [ ] Verify scheduler maintains <10ms frame latency

**Stream assignment:**
```cpp
// Holoscan operator streams
Stream h2d_stream;      // SDR → GPU transfer
Stream compute_stream;  // PSD, CFAR, peaks
Stream d2h_stream;      // Results → CPU
Stream cluster_stream;  // Async clustering
```

---

## 6. Telemetry & Backpressure

**Current state:**
- Drop rate, buffer fill, throttle flag exposed via `get_status()`
- Stats broadcast at 1 Hz via WebSocket
- Hysteresis throttling at 85%/50% watermarks

**Holoscan integration tasks:**
- [ ] Expose metrics via Holoscan monitoring API
- [ ] Wire throttle signal to SDR operator
- [ ] Consider adaptive rate control:
  - Reduce sample rate when buffer >70%
  - Increase when buffer <30%
  - Hysteresis to prevent oscillation

**Tuning parameters (hardware-dependent):**
```yaml
backpressure:
  high_watermark: 0.85    # Start throttling
  low_watermark: 0.50     # Stop throttling
  rate_step: 0.1          # 10% rate adjustment per step
  min_rate_msps: 10       # Don't go below 10 MSPS
  max_rate_msps: 50       # uSDR max rate
```

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Ingest rate | 50 MSPS sustained | TBD |
| Frame latency | <10 ms | TBD |
| GPU utilization | >70% | TBD |
| Drop rate | <0.1% at steady state | TBD |
| D2H/compute overlap | >80% | TBD |

---

## Profiling Commands

### Built-in Profiling Suite

```bash
# Run full profiling suite (benchmarks + backpressure + memory + transfers)
python -m rf_forensics.profiling.run_profiling --full

# Run only GPU benchmarks
python -m rf_forensics.profiling.run_profiling --benchmarks

# Run backpressure stress tests
python -m rf_forensics.profiling.run_profiling --backpressure

# Profile memory usage
python -m rf_forensics.profiling.run_profiling --memory

# Profile PCIe bandwidth
python -m rf_forensics.profiling.run_profiling --transfers

# Run with NVTX tracing (for Nsight Systems)
python -m rf_forensics.profiling.run_profiling --trace --trace-duration 10

# Generate summary report from collected data
python -m rf_forensics.profiling.run_profiling --report
```

### Nsight Systems (Timeline Analysis)

```bash
# Full system trace with NVTX markers
nsys profile --trace=cuda,nvtx,osrt \
  --output=rf_forensics_trace \
  python -m rf_forensics.profiling.run_profiling --trace

# Trace with actual pipeline
nsys profile --trace=cuda,nvtx \
  --output=pipeline_trace \
  python -m rf_forensics.pipeline.orchestrator
```

### Nsight Compute (Kernel Analysis)

```bash
# Kernel-level analysis
ncu --set full --target-processes all \
  python -m rf_forensics.profiling.run_profiling --trace

# Memory bandwidth analysis
ncu --metrics dram__bytes_read,dram__bytes_write \
  python -m rf_forensics.profiling.run_profiling --trace
```

### Python API Usage

```python
from rf_forensics.profiling import (
    run_all_benchmarks,
    nvtx_range,
    GPUMetricsCollector,
    run_backpressure_suite,
)

# Run benchmarks programmatically
results = run_all_benchmarks(fft_size=2048, iterations=100)

# Add NVTX markers to custom code
with nvtx_range("my_operation"):
    do_gpu_work()

# Measure memory and transfers
collector = GPUMetricsCollector()
collector.snapshot_memory("before")
# ... do work ...
collector.snapshot_memory("after")
collector.print_report()

# Run backpressure tests
bp_results = run_backpressure_suite()
```

---

## Files to Review

- `core/async_transfer.py` - D2H async manager
- `core/ring_buffer.py` - GPU ring buffer
- `core/stream_manager.py` - CUDA stream allocation
- `dsp/psd.py` - Welch PSD (cuFFT)
- `detection/cfar.py` - CFAR detector kernel
- `detection/peaks.py` - Peak extraction
- `ml/clustering.py` - DBSCAN clusterer
- `pipeline/orchestrator.py` - Main processing loop
- `holoscan/operators/` - Holoscan operator implementations
