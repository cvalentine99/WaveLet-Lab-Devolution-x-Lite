"""
Centralized Configuration Defaults

Contains all default values and magic numbers used throughout the codebase.
Import these constants instead of hardcoding values.

Usage:
    from rf_forensics.config.defaults import (
        RECORDINGS_PATH,
        MAX_DETECTIONS_LIMIT,
        DEFAULT_FFT_SIZE,
    )
"""

import os

# =========================================================================
# File System Paths
# =========================================================================
RECORDINGS_PATH = os.getenv("RECORDINGS_DIR", "/data/recordings")
DATA_PATH = os.getenv("DATA_DIR", "/data")
LOGS_PATH = os.getenv("LOGS_DIR", "/var/log/rf_forensics")

# =========================================================================
# API Limits
# =========================================================================
MAX_DETECTIONS_LIMIT = 1000
MAX_CLUSTERS_LIMIT = 100
MAX_RECORDINGS_LIMIT = 100
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000

# =========================================================================
# WebSocket Settings
# =========================================================================
WS_SPECTRUM_RATE_HZ = 30.0
WS_MAX_CLIENTS = 10
WS_SLEEP_INTERVAL = 0.01  # seconds
WS_PING_INTERVAL = 30.0  # seconds
WS_PING_TIMEOUT = 10.0  # seconds

# =========================================================================
# Timeouts
# =========================================================================
DEFAULT_TIMEOUT_SECONDS = 5.0
SDR_CONNECT_TIMEOUT = 10.0
GPU_INIT_TIMEOUT = 30.0
PIPELINE_START_TIMEOUT = 10.0
PIPELINE_STOP_TIMEOUT = 5.0

# =========================================================================
# DSP / FFT Settings
# =========================================================================
DEFAULT_FFT_SIZE = 2048
DEFAULT_OVERLAP_PERCENT = 50
DEFAULT_WINDOW_TYPE = "hann"
SPECTROGRAM_TIME_BINS = 256

# Min/Max FFT sizes (power of 2)
MIN_FFT_SIZE = 256
MAX_FFT_SIZE = 65536

# =========================================================================
# SDR Hardware Bounds (LMS7002M / uSDR)
# =========================================================================
MIN_FREQ_HZ = 70_000_000  # 70 MHz
MAX_FREQ_HZ = 6_000_000_000  # 6 GHz

MIN_SAMPLE_RATE_HZ = 100_000  # 100 kHz
MAX_SAMPLE_RATE_HZ = 61_440_000  # 61.44 MHz

MIN_BANDWIDTH_HZ = 100_000  # 100 kHz
MAX_BANDWIDTH_HZ = 56_000_000  # 56 MHz

MIN_GAIN_DB = 0
MAX_GAIN_DB = 73  # LNA (30) + TIA (12) + PGA (31)

# =========================================================================
# Buffer Settings
# =========================================================================
DEFAULT_SAMPLES_PER_BUFFER = 65536
DEFAULT_NUM_RING_SEGMENTS = 8
MIN_RING_SEGMENTS = 2
MAX_RING_SEGMENTS = 64

# =========================================================================
# CFAR Detection
# =========================================================================
DEFAULT_CFAR_REFERENCE_CELLS = 32
DEFAULT_CFAR_GUARD_CELLS = 8
DEFAULT_CFAR_PFA = 1e-4  # Probability of false alarm
CFAR_VARIANT_DEFAULT = "ca"  # Cell-averaging

# =========================================================================
# Clustering
# =========================================================================
DEFAULT_CLUSTERING_EPS = 0.5
DEFAULT_CLUSTERING_MIN_SAMPLES = 3
MAX_CLUSTERS = 100

# =========================================================================
# Recording ID Validation
# =========================================================================
RECORDING_ID_MAX_LENGTH = 64
RECORDING_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"

# =========================================================================
# Logging
# =========================================================================
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT_JSON = "json"
LOG_FORMAT_CONSOLE = "console"

# =========================================================================
# Version
# =========================================================================
BACKEND_VERSION = "1.0.0"
API_VERSION = "v1"
