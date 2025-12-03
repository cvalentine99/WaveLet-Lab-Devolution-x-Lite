# WaveLet Lab Devolution-x-Lite

**GPU-Accelerated RF Signal Forensics Platform**
<img width="1427" height="933" alt="image" src="https://github.com/user-attachments/assets/fdd9f111-12e6-40b4-bbd2-dabc53efddba" />

Real-time spectrum monitoring and signal analysis with CUDA-powered processing, designed for the Wavelet Lab uSDR DevBoard.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.2+-76B900.svg)

---

## Features

- **Real-time Spectrum Analysis** - GPU-accelerated FFT processing at 50+ MSPS
- **CFAR Detection** - Constant False Alarm Rate signal detection with multiple variants (CA, GO, SO, OS)
- **Signal Clustering** - DBSCAN-based clustering to group related detections
- **Protocol Decoding** - LoRa and BLE demodulation support
- **Web Dashboard** - React-based UI with WebSocket streaming
- **Recording & Playback** - SigMF-compatible I/Q recording

---

## Hardware Support

| SDR | Status | Max Sample Rate |
|-----|--------|-----------------|
| **uSDR DevBoard** | Full Support | 61.44 MSPS |


---

## Quick Start

```bash
# Clone
git clone https://github.com/cvalentine99/WaveLet-Lab-Devolution-x-Lite.git
cd WaveLet-Lab-Devolution-x-Lite

# Configure
cp .env.example .env

# Run
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access dashboard
open http://localhost:3001
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed installation instructions.

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    Frontend      │────▶│     Backend      │────▶│   uSDR DevBoard  │
│   React + Vite   │     │  FastAPI + CUDA  │     │    PCIe SDR      │
│   :3000/:3001    │     │   :8000/:8765    │     │   70M-6GHz       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                        │
        │    WebSocket           │    GPU Pipeline
        │    Spectrum Data       │    ┌─────────────────────┐
        │    Detections          │    │ SDR → FFT → CFAR    │
        │    Clusters            │    │ → Cluster → Demod   │
        └────────────────────────│    └─────────────────────┘
                                 │
                           NVIDIA RTX GPU
```

---

## System Requirements

- **GPU**: NVIDIA RTX 3080+ (12GB+ VRAM)
- **RAM**: 32GB minimum
- **OS**: Ubuntu 22.04 LTS
- **Docker**: 24.0+ with NVIDIA Container Toolkit
- **SDR**: Wavelet Lab uSDR DevBoard

---

## Project Structure

```
.
├── backend/                 # Python FastAPI backend
│   ├── src/rf_forensics/   # Core RF processing modules
│   │   ├── api/            # REST API and WebSocket
│   │   ├── gpu/            # CUDA kernels (FFT, CFAR)
│   │   ├── sdr/            # SDR drivers (uSDR)
│   │   └── pipeline/       # Processing orchestrator
│   └── Dockerfile
├── frontend/               # React web dashboard
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── pages/          # Page views
│   │   └── lib/            # API client
│   └── Dockerfile
├── holoscan/              # High-performance C++ pipeline (optional)
├── config/                # Configuration files
├── docs/                  # Documentation
└── docker-compose.yml     # Container orchestration
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sdr/devices` | GET | List SDR devices |
| `/api/sdr/connect` | POST | Connect to device |
| `/api/sdr/devices/{id}/stream/start` | POST | Start streaming |
| `/api/status` | GET | Pipeline status |
| `/api/detections` | GET | Signal detections |
| `/health` | GET | Health check |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend URL |
| `VITE_WS_URL` | `ws://localhost:8765` | WebSocket URL |
| `RF_RECORDINGS_DIR` | `/data/recordings` | Recording path |

### SDR Settings (via Dashboard)

- Center Frequency: 70 MHz - 6 GHz
- Sample Rate: 1-61.44 MSPS
- Bandwidth: 1-56 MHz
- Gains: LNA (0-30dB), TIA (0-12dB), PGA (0-32dB)

---

## Development

```bash
# Development mode with hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View logs
docker logs -f rf-forensics-backend

# Run backend tests
docker exec rf-forensics-backend pytest
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Wavelet Lab](https://wavelet-lab.com) - uSDR DevBoard hardware
- [NVIDIA](https://developer.nvidia.com/cuda-toolkit) - CUDA toolkit
- [CuPy](https://cupy.dev/) - GPU-accelerated NumPy

---

## Contributing

Contributions welcome! Please read the contribution guidelines before submitting PRs.

---

**Built with CUDA acceleration for real-time RF signal intelligence.**
