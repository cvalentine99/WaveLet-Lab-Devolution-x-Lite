# Holoscan RF Forensics Development Container
# Base: NVIDIA Holoscan SDK v2.6.0 with dGPU support
# Adds: libusdr for PCIe SDR, build tools

FROM nvcr.io/nvidia/clara-holoscan/holoscan:v2.6.0-dgpu

LABEL maintainer="RF Forensics Team"
LABEL description="Holoscan SDK + uSDR for 50 MSPS RF processing"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libusb-1.0-0-dev \
    python3-dev \
    python3-pip \
    python3-yaml \
    pybind11-dev \
    libfftw3-dev \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Build libusdr from source
WORKDIR /opt
COPY usdr-lib-main /opt/usdr-lib

WORKDIR /opt/usdr-lib/src/build
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DENABLE_SOAPY=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_GUI=OFF \
    -DENABLE_DMONITOR=OFF \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# Install usdr kernel module headers (for development)
RUN mkdir -p /usr/local/include/usdr && \
    cp /opt/usdr-lib/src/lib/lowlevel/pcie_uram/driver/pcie_uram_driver_if.h \
       /usr/local/include/usdr/

# Python dependencies for hybrid mode
RUN pip3 install --no-cache-dir \
    numpy \
    cupy-cuda12x \
    numba \
    pybind11

# Set up application directory
WORKDIR /app/holoscan
COPY . /app/holoscan/

# Build RF Forensics Holoscan operators
RUN mkdir -p /app/holoscan/build && \
    cd /app/holoscan/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_BINDINGS=ON \
    && make -j$(nproc)

# Environment
ENV LD_LIBRARY_PATH=/usr/local/lib:/app/holoscan/build:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/app/holoscan/build:/app:${PYTHONPATH}

# Default command
CMD ["/app/holoscan/build/rf_forensics_app", "--config", "/app/holoscan/rf_forensics.yaml"]
