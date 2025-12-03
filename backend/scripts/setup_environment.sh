#!/bin/bash
# GPU RF Forensics Engine - Environment Setup Script
# Requires: Conda/Mamba, NVIDIA Driver R525+, RTX 4090

set -e

ENV_NAME="rf-forensics-4090"
PYTHON_VERSION="3.10"
CUDA_VERSION="13.0"

echo "=============================================="
echo "GPU RF Forensics Engine - Environment Setup"
echo "=============================================="
echo ""

# Check for conda/mamba
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
    echo "[OK] Using mamba for faster package resolution"
elif command -v conda &> /dev/null; then
    PKG_MGR="conda"
    echo "[OK] Using conda"
else
    echo "[ERROR] Neither conda nor mamba found. Please install Miniforge or Anaconda."
    exit 1
fi

# Check for NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. Please install NVIDIA driver R525 or higher."
    exit 1
fi

echo ""
echo "Detected GPU(s):"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo ""

# Check for RTX 4090 (Compute Capability 8.9)
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q "8.9"; then
    echo "[OK] RTX 4090 (Compute Capability 8.9) detected"
else
    echo "[WARNING] RTX 4090 not detected. Proceeding anyway..."
fi

# Remove existing environment if it exists
if $PKG_MGR env list | grep -q "^${ENV_NAME} "; then
    echo ""
    read -p "Environment '${ENV_NAME}' exists. Remove and recreate? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $PKG_MGR env remove -n ${ENV_NAME} -y
    else
        echo "Aborting."
        exit 0
    fi
fi

echo ""
echo "Creating conda environment '${ENV_NAME}'..."
echo ""

# Create environment with RAPIDS 25.10 and CUDA 13.0
# Channel priority: rapidsai > nvidia > conda-forge
$PKG_MGR create -n ${ENV_NAME} \
    -c rapidsai \
    -c nvidia \
    -c conda-forge \
    python=${PYTHON_VERSION} \
    rapids=25.10 \
    cuda-version=13.0 \
    cusignal \
    cuml \
    cupy \
    numba \
    numpy \
    scipy \
    jupyterlab \
    -y

echo ""
echo "Activating environment and installing additional packages..."
echo ""

# Activate and install pip packages
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install additional packages via pip
pip install --no-cache-dir \
    websockets>=12.0 \
    fastapi>=0.109 \
    uvicorn[standard] \
    pydantic>=2.0 \
    pyyaml \
    python-multipart \
    httpx \
    pytest \
    pytest-asyncio \
    pytest-benchmark \
    aiofiles

# Optional: UHD for USRP support
echo ""
read -p "Install UHD (USRP SDR driver)? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    $PKG_MGR install -c conda-forge uhd -y
fi

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify the installation:"
echo "  python scripts/verify_environment.py"
echo ""
