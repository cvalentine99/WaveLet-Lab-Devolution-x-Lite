# Holoscan Migration: Issues and Lessons Learned

## Summary

Attempted to migrate RF forensics pipeline to NVIDIA Holoscan for improved throughput. **Holoscan never actually ran.** Performance improvements achieved came from basic Python optimizations, not Holoscan.

---

## What Went Wrong

### 1. CUDA Version Mismatch

**Problem:** System has CUDA 13.0, Holoscan pip package requires CUDA 12.x.

```
$ python -c "import holoscan"
ImportError: libcudart.so.12: cannot open shared object file
```

**Root cause:** Installed `holoscan` via pip which pulled `holoscan-cu12`. The required `.so` files don't exist on a CUDA 13 system.

**Should have done:** Used the existing Docker container which has CUDA 12.2, or pulled the official Holoscan container immediately.

---

### 2. Forgot About Existing Docker Infrastructure

**Problem:** Spent time debugging pip installs and CUDA paths when a working `gpuforensics-backend` container already existed with CUDA 12.2 and all dependencies.

```yaml
# docker-compose.yml was RIGHT THERE
backend:
  build:
    context: ./rf_forensics
    dockerfile: Dockerfile.backend  # CUDA 12.2 + cupy-cuda12x
```

**Should have done:** `docker compose up -d backend && docker exec rf-forensics-backend python test.py`

---

### 3. C++ Code Never Compiled or Tested

**Problem:** Wrote 18 files of C++ Holoscan operators and CUDA kernels:
- `holoscan/operators/*.cpp` - PSD, CFAR, USDR source, Peak operators
- `holoscan/kernels/*.cu` - CUDA kernels
- `holoscan/CMakeLists.txt` - Build system

**None of this code was ever compiled or run.** It's theoretical.

**Should have done:** Built and tested incrementally in the container instead of writing everything speculatively.

---

### 4. Pip Package != Full SDK

**Problem:** The pip-installed `holoscan` package only provides Python bindings. The C++ operators require:
- Full Holoscan SDK headers and libraries
- GXF extensions
- `find_package(holoscan)` working in CMake

**Should have done:** Used NVIDIA's Holoscan container (`nvcr.io/nvidia/clara-holoscan/holoscan:v2.6.0-dgpu`) which has everything.

---

### 5. Mislabeled Code

**Problem:** Created `holoscan/python/native_pipeline.py` that showed 130 MSPS throughput. This file:
- Has zero Holoscan code
- Is just CuPy with pre-allocated buffers
- Is the same architecture that already existed

Called it "Holoscan" when it wasn't.

---

### 6. Path With Spaces Broke Docker Mounts

**Problem:** Project path `/home/cvalentine/GPU Forensics /rf_forensics` has a space. Docker volume mounts failed repeatedly:
```
docker: invalid reference format
```

Wasted time creating symlinks instead of just using the existing docker-compose setup.

---

## What Actually Worked

### Ring Buffer Architecture (No Holoscan Required)

The real performance improvement came from decoupling SDR callbacks from GPU processing:

```python
# Before: GPU work in callback (blocking)
def callback(samples):
    gpu_result = process_on_gpu(samples)  # Blocks SDR thread!

# After: Just queue data (fast)
def callback(samples):
    ring_buffer.write(samples)  # ~1μs

# Separate thread does GPU work
def gpu_worker():
    while running:
        samples = ring_buffer.read()
        process_on_gpu(samples)
```

**Results:**
| Metric | Before | After |
|--------|--------|-------|
| Throughput | 5.13 MSPS | 9.99 MSPS |
| Efficiency | 51% | 99.9% |
| Overflows | Many | 0 |

This is standard producer-consumer pattern, not Holoscan.

---

## Files Created (Status)

| File | Status |
|------|--------|
| `holoscan/CMakeLists.txt` | Untested |
| `holoscan/rf_forensics_app.cpp` | Untested |
| `holoscan/operators/*.cpp` | Untested |
| `holoscan/kernels/*.cu` | Untested |
| `holoscan/python/bindings.cpp` | Untested |
| `holoscan/python/native_pipeline.py` | Works (but isn't Holoscan) |
| `scripts/test_ring_buffer_usdr.py` | **Works - achieved 99.9% efficiency** |

---

## To Actually Use Holoscan

### Option 1: Use Existing Backend Container
```bash
# Rebuild with Holoscan added to Dockerfile.backend
docker compose build backend
docker compose up -d backend
docker exec -it rf-forensics-backend bash
cd /app/holoscan && mkdir build && cd build && cmake .. && make
```

### Option 2: Use Official Holoscan Container
```bash
docker run --gpus all -v /path/to/rf_forensics:/workspace \
  nvcr.io/nvidia/clara-holoscan/holoscan:v2.6.0-dgpu bash
```

### Option 3: Add Holoscan to Dockerfile.backend
```dockerfile
# Add to Dockerfile.backend
RUN pip3 install holoscan
# OR for C++ support:
RUN apt-get install -y holoscan-dev
```

---

## Lessons Learned

1. **Check existing infrastructure first** - Docker containers were already set up
2. **Test incrementally** - Don't write 18 files without compiling any
3. **Use containers for CUDA version issues** - Don't fight pip on the host
4. **Be honest about what's working** - "Native Holoscan Pipeline" wasn't Holoscan
5. **Simple solutions first** - Ring buffer achieved the target without Holoscan

---

## Current State

- **Throughput target (9.5 MSPS): ACHIEVED** via ring buffer architecture
- **Holoscan integration: NOT WORKING** - C++ code untested
- **Container: AVAILABLE** - `rf-forensics-backend` has CUDA 12.2

The ring buffer solution in `scripts/test_ring_buffer_usdr.py` achieves the performance target. Holoscan migration can proceed later if additional benefits are needed.
