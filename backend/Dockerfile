# GPU RF Forensics Engine - Backend Dockerfile
# NVIDIA CUDA-enabled container for GPU-accelerated RF signal processing

# ============================================
# Base image: NVIDIA CUDA with cuDNN
# ============================================
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 AS base

# Prevent interactive prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# ============================================
# System Dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    build-essential \
    git \
    curl \
    wget \
    libfftw3-dev \
    libusb-1.0-0-dev \
    libboost-all-dev \
    pkg-config \
    cmake \
    ninja-build \
    # SDR library dependencies
    libuhd-dev \
    libhackrf-dev \
    librtlsdr-dev \
    liblimesuite-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ============================================
# Build libusdr from source (uSDR DevBoard)
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsoapysdr-dev \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
    && git clone --depth 1 https://github.com/wavelet-lab/usdr-lib.git \
    && cd usdr-lib \
    && mkdir build && cd build \
    && cmake ../src -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && make -j$(nproc) usdr usdr-dsp \
    && cp lib/libusdr.so* /usr/local/lib/ \
    && cp lib/xdsp/libusdr-dsp.so* /usr/local/lib/ \
    && ldconfig \
    && rm -rf /tmp/usdr-lib

# ============================================
# Create app user (non-root)
# ============================================
RUN useradd -m -s /bin/bash rfengine \
    && mkdir -p /app /data/recordings /var/log/rf_forensics /app/config \
    && chown -R rfengine:rfengine /app /data /var/log/rf_forensics

WORKDIR /app

# ============================================
# Python Dependencies
# ============================================
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install CuPy for GPU acceleration (matches CUDA 12.x)
RUN pip3 install --no-cache-dir cupy-cuda12x

# Copy requirements and install
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# ============================================
# Application Code
# ============================================
# Copy source code
COPY --chown=rfengine:rfengine src/ /app/src/
COPY --chown=rfengine:rfengine scripts/ /app/scripts/
COPY --chown=rfengine:rfengine tests/ /app/tests/
COPY --chown=rfengine:rfengine pyproject.toml /app/

# Install the package
RUN pip3 install --no-cache-dir -e .

# ============================================
# Environment Configuration
# ============================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    # CUDA optimization
    CUDA_CACHE_MAXSIZE=2147483648 \
    CUDA_CACHE_PATH=/tmp/cuda_cache \
    # RF Forensics config
    RF_CONFIG_PATH=/app/config/defaults.yaml \
    RF_RECORDINGS_DIR=/data/recordings \
    RF_LOG_DIR=/var/log/rf_forensics \
    # SDR library path
    SDR_LIB_PATH=/usr/local/lib/libusdr.so

# ============================================
# Health Check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ============================================
# Expose Ports
# ============================================
EXPOSE 8000 8765

# ============================================
# Runtime Configuration
# ============================================
USER rfengine

# Default command: run the RF forensics server
CMD ["python", "-m", "rf_forensics"]

# ============================================
# Development target (for docker-compose.dev.yml)
# ============================================
FROM base AS development
ENV DEBUG=1 \
    LOG_LEVEL=DEBUG
CMD ["python", "-m", "rf_forensics", "--simulate", "--debug"]
