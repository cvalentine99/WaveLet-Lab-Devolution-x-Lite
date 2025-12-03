# Valentine RF - Python Backend

GPU-accelerated IQ file analysis endpoint for the Valentine RF forensics platform.

## Features

- **4 IQ File Format Parsers**: complex64, int16_iq, uint8, SigMF
- **GPU Acceleration**: Automatic CuPy/CUDA support with NumPy fallback
- **CFAR Detection**: Cell-Averaging CFAR algorithm for signal detection
- **FastAPI**: Modern async Python web framework
- **CORS Enabled**: Ready for frontend integration

## Installation

### 1. Create Python Virtual Environment

```bash
cd python_backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

**CPU-only (NumPy fallback):**
```bash
pip install -r requirements.txt
```

**GPU-accelerated (NVIDIA GPU required):**
```bash
# For CUDA 12.x
pip install -r requirements.txt
pip install cupy-cuda12x==13.3.0

# For CUDA 11.x
pip install -r requirements.txt
pip install cupy-cuda11x==13.3.0
```

### 3. Configure Upload Directory

Set the `UPLOAD_DIR` environment variable to match Node.js configuration:

```bash
export UPLOAD_DIR="rf_forensics/api/uploads"
```

Or create a `.env` file:
```
UPLOAD_DIR=rf_forensics/api/uploads
PORT=8000
```

## Running the Backend

### Development Mode

```bash
cd python_backend
source venv/bin/activate
python main.py
```

Server will start on `http://localhost:8000`

### Production Mode (with Uvicorn)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### With Auto-Reload (Development)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### GET /

Health check endpoint.

**Response:**
```json
{
  "service": "Valentine RF Backend",
  "version": "1.0.0",
  "gpu_available": true
}
```

### GET /health

Detailed health check with GPU status.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 3090",
    "memory_total_gb": 24.0,
    "memory_free_gb": 20.5,
    "memory_used_pct": 14.6
  },
  "upload_dir": "rf_forensics/api/uploads"
}
```

### POST /api/analysis/start

Process uploaded IQ file and return detections + spectrum.

**Request:**
```json
{
  "analysis_id": "uuid-string",
  "format": "complex64",
  "sample_rate_hz": 20000000,
  "center_freq_hz": 915000000,
  "fft_size": 2048
}
```

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "status": "complete",
  "sample_rate_hz": 20000000,
  "center_freq_hz": 915000000,
  "fft_size": 2048,
  "detections": [
    {
      "detection_id": 1,
      "center_freq_hz": 915125000,
      "bandwidth_hz": 125000,
      "bandwidth_3db_hz": 120000,
      "bandwidth_6db_hz": 130000,
      "peak_power_db": -45.2,
      "snr_db": 15.3,
      "start_bin": 100,
      "end_bin": 150,
      "timestamp": 1234567890
    }
  ],
  "spectrum": {
    "magnitude_db": [/* array of float values */],
    "fft_size": 2048,
    "sample_rate_hz": 20000000,
    "center_freq_hz": 915000000
  }
}
```

## File Formats

### complex64 (.cf32, .cfile)
- Float32 I/Q pairs (little-endian)
- Interleaved: [I0, Q0, I1, Q1, ...]
- **Requires:** `sample_rate_hz` and `center_freq_hz` in request

### int16_iq (.cs16, .iq)
- Int16 I/Q pairs (little-endian)
- Normalized to -1.0 to 1.0 range
- **Requires:** `sample_rate_hz` and `center_freq_hz` in request

### uint8 (.cu8)
- Uint8 I/Q pairs (offset 128 for signed)
- Normalized to -1.0 to 1.0 range
- **Requires:** `sample_rate_hz` and `center_freq_hz` in request

### SigMF (.sigmf-data + .sigmf-meta)
- Standard SigMF format with JSON metadata
- Auto-detects sample rate and center frequency from `.sigmf-meta`
- Supports: cf32_le, ci16_le, cu8 datatypes

## CFAR Detection Parameters

The CFAR (Constant False Alarm Rate) detector uses these default parameters:

- **ref_cells**: 16 (reference cells on each side)
- **guard_cells**: 4 (guard cells on each side)
- **pfa**: 1e-6 (probability of false alarm)
- **threshold_offset_db**: 10.0 (additional threshold in dB)

These can be adjusted in `main.py` for different detection sensitivities.

## Testing

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Start analysis (after file is uploaded by Node.js)
curl -X POST http://localhost:8000/api/analysis/start \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "test-123",
    "format": "complex64",
    "sample_rate_hz": 20000000,
    "center_freq_hz": 915000000,
    "fft_size": 2048
  }'
```

### Test with Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Start analysis
response = requests.post(
    "http://localhost:8000/api/analysis/start",
    json={
        "analysis_id": "test-123",
        "format": "complex64",
        "sample_rate_hz": 20000000,
        "center_freq_hz": 915000000,
        "fft_size": 2048,
    }
)
print(response.json())
```

## Integration with Frontend

1. **Start Python backend** on port 8000
2. **Start Node.js dev server** on port 3000
3. **Upload file** via frontend → Node.js stores in `rf_forensics/api/uploads/`
4. **Node.js proxies** to Python `/api/analysis/start`
5. **Python processes** file and returns results
6. **Frontend displays** detections and spectrum

## Environment Variables

- `UPLOAD_DIR`: Upload directory path (default: `rf_forensics/api/uploads`)
- `PORT`: Server port (default: `8000`)

## GPU Requirements

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** 11.x or 12.x
- **CuPy** installed (see installation instructions above)

If no GPU is available, the backend automatically falls back to NumPy/SciPy (CPU mode).

## Troubleshooting

### "File not found for analysis_id"

- Check that `UPLOAD_DIR` matches between Node.js and Python
- Verify file was uploaded successfully by Node.js
- Check file permissions

### "sample_rate_hz required for complex64 format"

- For non-SigMF formats, you must provide `sample_rate_hz` in the request
- SigMF format reads this from `.sigmf-meta` automatically

### GPU not detected

- Check CUDA installation: `nvidia-smi`
- Verify CuPy installation: `python -c "import cupy; print(cupy.__version__)"`
- Backend will fall back to CPU mode if GPU unavailable

## Performance

- **GPU Mode**: ~10-50ms for 1M samples (depending on GPU)
- **CPU Mode**: ~100-500ms for 1M samples (depending on CPU)
- **CFAR Detection**: O(n) complexity, scales linearly with FFT size

## License

Proprietary - Valentine RF
