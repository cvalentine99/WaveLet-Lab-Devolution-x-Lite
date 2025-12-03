#!/bin/bash
# Build script for RF Forensics Holoscan Pipeline
# Requirements: NVIDIA Holoscan SDK 2.0+, CUDA Toolkit 12.0+, CMake 3.20+

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${1:-Release}"

echo "=============================================="
echo "RF Forensics Holoscan Pipeline Build"
echo "=============================================="
echo "Build type: ${BUILD_TYPE}"
echo "Build dir:  ${BUILD_DIR}"
echo ""

# Check for required tools
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 not found. Please install it first."
        exit 1
    fi
}

check_tool cmake
check_tool nvcc

# Check for Holoscan SDK
if [ -z "${HOLOSCAN_ROOT}" ]; then
    # Try common locations
    for dir in /opt/nvidia/holoscan /usr/local/holoscan ~/holoscan; do
        if [ -d "$dir" ]; then
            export HOLOSCAN_ROOT="$dir"
            break
        fi
    done
fi

if [ -z "${HOLOSCAN_ROOT}" ] || [ ! -d "${HOLOSCAN_ROOT}" ]; then
    echo "WARNING: HOLOSCAN_ROOT not set or not found."
    echo "Set it with: export HOLOSCAN_ROOT=/path/to/holoscan"
    echo "Attempting build anyway (cmake may find it)..."
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
echo ""
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DBUILD_PYTHON_BINDINGS=ON

# Build
echo ""
echo "Building..."
cmake --build . --parallel $(nproc)

# Report results
echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Executables:"
ls -la "${BUILD_DIR}/rf_forensics_app" 2>/dev/null && echo "" || echo "  (not found)"

echo "Libraries:"
ls -la "${BUILD_DIR}"/*.so 2>/dev/null || echo "  (none found)"

echo ""
echo "To run the application:"
echo "  ${BUILD_DIR}/rf_forensics_app --simulate"
echo ""
echo "To run with real USDR hardware:"
echo "  ${BUILD_DIR}/rf_forensics_app --freq 915e6 --rate 10e6"
echo ""
echo "To use from Python:"
echo "  export PYTHONPATH=${BUILD_DIR}:\$PYTHONPATH"
echo "  python -c 'import rf_forensics_holoscan; print(rf_forensics_holoscan.__version__)'"
