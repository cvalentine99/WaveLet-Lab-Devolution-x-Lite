"""
GPU RF Forensics Engine - SDR Module

SDR device interfaces and drivers.

Architecture:
- SDRManager: Unified singleton for driver ownership
- USDRDriver: Low-level USDR hardware driver
- SDRMetrics: Real-time metrics tracking
"""

from rf_forensics.sdr.manager import SDRManager, get_sdr_manager
from rf_forensics.sdr.metrics import MetricsTracker, SDRCapabilities, SDRMetrics
from rf_forensics.sdr.usdr_driver import (
    DUPLEXER_BANDS,
    USDRConfig,
    USDRDevice,
    USDRDriver,
    USDRGain,
    get_band_by_name,
)

__all__ = [
    # Manager (primary entry point)
    "get_sdr_manager",
    "SDRManager",
    # Metrics
    "SDRMetrics",
    "MetricsTracker",
    "SDRCapabilities",
    # Driver types
    "USDRDriver",
    "USDRDevice",
    "USDRConfig",
    "USDRGain",
    # Duplexer bands
    "DUPLEXER_BANDS",
    "get_band_by_name",
]
