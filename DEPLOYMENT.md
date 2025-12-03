# RF Forensics Deployment Guide

GPU-accelerated RF signal analysis platform with uSDR DevBoard support.

## System Requirements

### Hardware
- **GPU**: NVIDIA RTX 3080 or better (tested on RTX 4090)
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 100GB SSD for recordings
- **SDR**: Wavelet Lab uSDR DevBoard (PCIe)

### Software
- Ubuntu 22.04 LTS or later
- Docker 24.0+ with NVIDIA Container Toolkit
- NVIDIA Driver 535+ with CUDA 12.2+
- uSDR kernel driver (`usdr-dkms`)

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/cvalentine99/WaveLet-Lab-Devolution-x-Lite.git
cd WaveLet-Lab-Devolution-x-Lite
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work for most setups)
```

### 3. Start Services
```bash
# Development mode (with hot reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production mode
docker compose up -d
```

### 4. Access Dashboard
- **Frontend**: http://localhost:3000 (prod) or http://localhost:3001 (dev)
- **Backend API**: http://localhost:8000
- **WebSocket**: ws://localhost:8765

---

## Installation Details

### Install NVIDIA Container Toolkit
```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Install uSDR Driver
```bash
# Install uSDR DKMS driver from Wavelet Lab
# See: https://github.com/wavelet-lab/usdr-lib
sudo apt install usdr-dkms libusdr0

# Verify device
ls -la /dev/usdr0
```

### Verify GPU Access
```bash
docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│                    http://localhost:3000                     │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST API / WebSocket
┌─────────────────────────▼───────────────────────────────────┐
│                   Backend (Python/FastAPI)                   │
│              REST: :8000  │  WebSocket: :8765               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              RF Forensics Pipeline                   │    │
│  │  SDR Manager → GPU FFT → CFAR → Clustering → ML    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │ PCIe DMA
┌─────────────────────────▼───────────────────────────────────┐
│                    uSDR DevBoard                             │
│           70 MHz - 6 GHz  │  Up to 61.44 MSPS               │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend API URL |
| `VITE_WS_URL` | `ws://localhost:8765` | WebSocket URL |
| `RF_RECORDINGS_DIR` | `/data/recordings` | Recording storage |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins |
| `DEBUG` | `0` | Enable debug logging |

### SDR Configuration

Default tuning (configurable via dashboard):
- **Center Frequency**: 915 MHz (ISM band)
- **Sample Rate**: 10 MSPS
- **Bandwidth**: 10 MHz
- **Gains**: LNA=15dB, TIA=9dB, PGA=12dB

### Duplexer Bands

| Band | Frequency Range | Use Case |
|------|-----------------|----------|
| `rxlpf1200` | 0-1200 MHz | RX only, LPF |
| `rxlpf2100` | 0-2100 MHz | RX only, LPF |
| `band2` | 1850-1990 MHz | PCS/GSM 1900 |
| `band3` | 1710-1880 MHz | DCS/GSM 1800 |
| `band5` | 824-894 MHz | GSM 850 |
| `band7` | 2500-2690 MHz | LTE Band 7 |
| `band8` | 880-960 MHz | GSM 900 |

---

## API Reference

### SDR Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sdr/devices` | List available SDR devices |
| POST | `/api/sdr/connect` | Connect to SDR device |
| POST | `/api/sdr/devices/{id}/stream/start` | Start streaming |
| POST | `/api/sdr/devices/{id}/stream/stop` | Stop streaming |
| POST | `/api/sdr/frequency` | Set center frequency |
| POST | `/api/sdr/gain` | Set LNA/TIA/PGA gains |
| GET | `/api/sdr/bands` | Get duplexer bands |
| POST | `/api/sdr/bands` | Set duplexer band |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/api/status` | Pipeline status |
| GET | `/api/detections` | Get signal detections |
| GET | `/api/clusters` | Get signal clusters |

---

## Troubleshooting

### "Failed to connect to SDR"
1. Check device exists: `ls -la /dev/usdr0`
2. Verify permissions: device should be `rw-rw-rw-`
3. Check container has device access: `docker exec rf-forensics-backend ls -la /dev/usdr0`

### "GPU not available"
1. Verify NVIDIA driver: `nvidia-smi`
2. Check Docker GPU access: `docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi`
3. Ensure `runtime: nvidia` in docker-compose.yml

### Frontend shows "Backend Offline"
1. Check backend is running: `docker ps`
2. Check backend logs: `docker logs rf-forensics-backend`
3. Verify CORS settings include frontend origin

### WebSocket disconnections
1. Check network stability
2. Increase buffer sizes if needed
3. Monitor `docker logs rf-forensics-backend` for errors

---

## Development

### Hot Reload (Dev Mode)
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```
- Frontend changes: Auto-reload via Vite HMR
- Backend changes: Requires container restart

### Run Tests
```bash
# Backend tests
docker exec rf-forensics-backend pytest

# Frontend tests
docker exec rf-forensics-frontend npm test
```

### Build Production Images
```bash
docker compose build
```

---

## License

MIT License - See LICENSE file for details.

---

## Support

- **GitHub Issues**: https://github.com/cvalentine99/WaveLet-Lab-Devolution-x-Lite/issues
- **uSDR Documentation**: https://docs.wsdr.io
- **Wavelet Lab**: https://wavelet-lab.com
