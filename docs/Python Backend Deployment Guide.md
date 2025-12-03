# Python Backend Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
cd python_backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: Install CuPy for GPU acceleration
# For CUDA 12.x:
pip install cupy-cuda12x==13.3.0
# For CUDA 11.x:
pip install cupy-cuda11x==13.3.0
```

### 2. Configure Environment

Create `.env` file in `python_backend/` directory:

```bash
# Upload directory (absolute path recommended)
UPLOAD_DIR=/home/ubuntu/rf-forensics-web/rf_forensics/api/uploads

# Server port
PORT=8001
```

Or use environment variables:

```bash
export UPLOAD_DIR=/home/ubuntu/rf-forensics-web/rf_forensics/api/uploads
export PORT=8001
```

### 3. Start Backend

**Development:**
```bash
cd python_backend
source venv/bin/activate
python main.py
```

**Production (systemd service):**
```bash
sudo cp python-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable python-backend
sudo systemctl start python-backend
sudo systemctl status python-backend
```

## Systemd Service Configuration

Create `/etc/systemd/system/python-backend.service`:

```ini
[Unit]
Description=Valentine RF Python Backend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/rf-forensics-web/python_backend
Environment="PATH=/home/ubuntu/rf-forensics-web/python_backend/venv/bin"
Environment="UPLOAD_DIR=/home/ubuntu/rf-forensics-web/rf_forensics/api/uploads"
Environment="PORT=8001"
ExecStart=/home/ubuntu/rf-forensics-web/python_backend/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Nginx Reverse Proxy

Add to your Nginx configuration:

```nginx
# Python backend proxy
location /api/analysis/ {
    proxy_pass http://localhost:8001;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_cache_bypass $http_upgrade;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # Increase timeout for large file processing
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;
}
```

## Docker Deployment

Create `Dockerfile` in `python_backend/`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Set environment variables
ENV UPLOAD_DIR=/app/uploads
ENV PORT=8001

# Create upload directory
RUN mkdir -p /app/uploads

# Run application
CMD ["python", "main.py"]
```

Build and run:

```bash
docker build -t valentine-rf-backend .
docker run -d \
  --name valentine-backend \
  -p 8001:8001 \
  -v /path/to/uploads:/app/uploads \
  valentine-rf-backend
```

## GPU Support in Docker

For GPU acceleration with Docker:

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir cupy-cuda12x==13.3.0

COPY . .

EXPOSE 8001
ENV UPLOAD_DIR=/app/uploads
ENV PORT=8001

CMD ["python3", "main.py"]
```

Run with GPU:

```bash
docker run -d \
  --name valentine-backend \
  --gpus all \
  -p 8001:8001 \
  -v /path/to/uploads:/app/uploads \
  valentine-rf-backend
```

## Health Checks

Monitor backend health:

```bash
# Basic health check
curl http://localhost:8001/health

# Expected response:
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_info": {
    "name": "NVIDIA GeForce RTX 3090",
    "memory_total_gb": 24.0,
    "memory_free_gb": 20.5,
    "memory_used_pct": 14.6
  },
  "upload_dir": "/home/ubuntu/rf-forensics-web/rf_forensics/api/uploads"
}
```

## Monitoring & Logging

### Systemd Logs

```bash
# View logs
sudo journalctl -u python-backend -f

# Last 100 lines
sudo journalctl -u python-backend -n 100

# Since specific time
sudo journalctl -u python-backend --since "1 hour ago"
```

### Application Logs

Backend logs to stdout/stderr. Redirect to file:

```bash
python main.py > backend.log 2>&1 &
```

Or use systemd's journal.

## Performance Tuning

### Uvicorn Workers

For production, use multiple workers:

```python
# In main.py, change:
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    workers = int(os.getenv("WORKERS", 4))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info"
    )
```

Or run with uvicorn CLI:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

### CFAR Detection Tuning

Adjust detection sensitivity in `main.py`:

```python
# More sensitive (more detections, more false alarms)
cfar_detections = cfar_detection(
    psd_db,
    ref_cells=8,           # Fewer reference cells
    guard_cells=2,         # Fewer guard cells
    pfa=1e-4,              # Higher Pfa
    threshold_offset_db=5.0,  # Lower threshold
)

# Less sensitive (fewer detections, fewer false alarms)
cfar_detections = cfar_detection(
    psd_db,
    ref_cells=32,          # More reference cells
    guard_cells=8,         # More guard cells
    pfa=1e-8,              # Lower Pfa
    threshold_offset_db=15.0,  # Higher threshold
)
```

## Troubleshooting

### Backend Not Starting

1. Check port availability:
   ```bash
   lsof -i :8001
   ```

2. Check Python version:
   ```bash
   python3 --version  # Should be 3.11+
   ```

3. Check dependencies:
   ```bash
   pip list | grep -E "fastapi|uvicorn|numpy"
   ```

### File Not Found Errors

1. Verify upload directory exists:
   ```bash
   ls -la /home/ubuntu/rf-forensics-web/rf_forensics/api/uploads/
   ```

2. Check environment variable:
   ```bash
   echo $UPLOAD_DIR
   ```

3. Test glob pattern:
   ```bash
   python3 -c "import glob, os; print(glob.glob(os.path.join('$UPLOAD_DIR', 'test-123_*')))"
   ```

### GPU Not Detected

1. Check CUDA installation:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. Check CuPy installation:
   ```bash
   python3 -c "import cupy; print(cupy.__version__)"
   ```

3. Backend will automatically fall back to CPU mode if GPU unavailable

### Slow Processing

1. Check CPU/GPU usage:
   ```bash
   htop
   nvidia-smi
   ```

2. Reduce FFT size for faster processing (trade-off: lower frequency resolution)

3. Use GPU acceleration (CuPy) for 5-10x speedup

## Security Considerations

1. **CORS**: In production, restrict `allow_origins` to specific domains:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com"],
       allow_credentials=True,
       allow_methods=["POST", "GET"],
       allow_headers=["*"],
   )
   ```

2. **File Cleanup**: Optionally delete files after processing to save disk space:
   ```python
   # In start_analysis endpoint, after processing:
   os.remove(file_path)
   ```

3. **Rate Limiting**: Add rate limiting for production:
   ```bash
   pip install slowapi
   ```

4. **Authentication**: Add API key authentication if needed

## Integration Testing

Test complete upload → analyze flow:

```bash
# 1. Upload file via Node.js (port 3000)
curl -X POST http://localhost:3000/api/trpc/upload.uploadFile \
  -F "file=@test_files/test_single_tone.cf32" \
  -F "format=complex64" \
  -F "sampleRate=20000000" \
  -F "centerFreq=915000000"

# Response will include analysis_id

# 2. Analyze via Python backend (port 8001)
curl -X POST http://localhost:8001/api/analysis/start \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "<analysis_id_from_upload>",
    "format": "complex64",
    "sample_rate_hz": 20000000,
    "center_freq_hz": 915000000,
    "fft_size": 2048
  }'
```

## Maintenance

### Update Dependencies

```bash
cd python_backend
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Backup

Backup configuration and logs:

```bash
tar -czf backend-backup-$(date +%Y%m%d).tar.gz \
  python_backend/*.py \
  python_backend/requirements.txt \
  python_backend/.env \
  /var/log/python-backend/
```

### Monitoring Disk Space

Monitor upload directory:

```bash
du -sh /home/ubuntu/rf-forensics-web/rf_forensics/api/uploads/
```

Set up automatic cleanup:

```bash
# Cron job to delete files older than 7 days
0 2 * * * find /home/ubuntu/rf-forensics-web/rf_forensics/api/uploads/ -type f -mtime +7 -delete
```
