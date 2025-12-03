"""
GPU RF Forensics Engine - REST API

FastAPI REST endpoints for system control and data access.
Modular architecture using FastAPI routers.

Security Features:
- CORS whitelist (configurable via CORS_ORIGINS env var)
- Rate limiting on sensitive endpoints
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# CORS whitelist - configurable via environment
# Default: localhost development origins
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3030,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:3030,http://127.0.0.1:5173",
).split(",")

from rf_forensics.forensics.recording_manager import RecordingManager


class RFForensicsAPI:
    """
    REST API manager for RF Forensics Engine.

    Provides endpoints for:
    - System control (start/stop/pause)
    - Configuration management
    - Detection and cluster data access
    - Recording management
    """

    def __init__(self, pipeline=None, recordings_dir: str = None):
        """
        Initialize API manager.

        Args:
            pipeline: Reference to RFForensicsPipeline instance.
            recordings_dir: Directory for SigMF recordings.
        """
        self._pipeline = pipeline
        self._detections: list[dict] = []
        self._clusters: list[dict] = []
        # Use local recordings directory if not specified
        if recordings_dir is None:
            recordings_dir = str(Path(__file__).parent.parent / "recordings")
        self._recording_manager = RecordingManager(output_dir=recordings_dir)
        # Session tracking for frontend compatibility
        self._session_counter = 0
        self._active_session: str | None = None
        self._monitoring_config: dict = {}

    def set_pipeline(self, pipeline):
        """Set pipeline reference."""
        self._pipeline = pipeline

    def add_detection(self, detection: dict):
        """Add a detection to the list."""
        self._detections.append(detection)
        # Keep only last 1000 detections
        if len(self._detections) > 1000:
            self._detections = self._detections[-1000:]

    def update_clusters(self, clusters: list[dict]):
        """Update cluster list."""
        self._clusters = clusters

    def get_status(self) -> dict:
        """Get current system status."""
        if self._pipeline is None:
            return {
                "state": "idle",
                "uptime_seconds": 0,
                "samples_processed": 0,
                "detections_count": len(self._detections),
                "current_throughput_msps": 0,
                "gpu_memory_used_gb": 0,
                "buffer_fill_level": 0.0,
                "processing_latency_ms": 0.0,
            }

        # Get status from pipeline
        status = self._pipeline.get_status() if hasattr(self._pipeline, "get_status") else {}
        # Ensure all required fields are present
        return {
            "state": status.get("state", "idle"),
            "uptime_seconds": status.get("uptime_seconds", 0),
            "samples_processed": status.get("samples_processed", 0),
            "detections_count": status.get("detections_count", len(self._detections)),
            "current_throughput_msps": status.get("current_throughput_msps", 0),
            "gpu_memory_used_gb": status.get("gpu_memory_used_gb", 0),
            "buffer_fill_level": status.get("buffer_fill_level", 0.0),
            "processing_latency_ms": status.get("processing_latency_ms", 0.0),
        }

    def get_frontend_config(self) -> dict:
        """Get configuration in frontend-expected format."""
        if self._pipeline is None or not hasattr(self._pipeline, "config"):
            return {
                "sdr": {"center_freq_hz": 915e6, "sample_rate_hz": 10e6, "gain_db": 40.0},
                "pipeline": {"fft_size": 1024, "window_type": "hann", "overlap": 0.5},
                "cfar": {"pfa": 1e-6, "ref_cells": 32, "guard_cells": 4},
                "clustering": {"enabled": True, "eps": 0.5, "min_samples": 5},
            }

        cfg = self._pipeline.config
        return {
            "sdr": {
                "center_freq_hz": cfg.sdr.center_freq_hz,
                "sample_rate_hz": cfg.sdr.sample_rate_hz,
                "gain_db": cfg.sdr.gain_db,
            },
            "pipeline": {
                "fft_size": cfg.fft.fft_size,
                "window_type": cfg.fft.window_type,
                "overlap": cfg.fft.overlap_percent / 100.0,  # Convert percent to 0-1
            },
            "cfar": {
                "pfa": cfg.cfar.pfa,
                "ref_cells": cfg.cfar.num_reference_cells,
                "guard_cells": cfg.cfar.num_guard_cells,
            },
            "clustering": {
                "enabled": True,
                "eps": cfg.clustering.eps,
                "min_samples": cfg.clustering.min_samples,
            },
        }

    def apply_frontend_config(self, frontend_cfg: dict) -> None:
        """Apply configuration from frontend format."""
        if self._pipeline is None:
            return

        # Translate frontend format to backend format
        backend_update = {}

        if "sdr" in frontend_cfg:
            backend_update["sdr"] = frontend_cfg["sdr"]

        if "pipeline" in frontend_cfg:
            p = frontend_cfg["pipeline"]
            backend_update["fft"] = {
                "fft_size": p.get("fft_size"),
                "window_type": p.get("window_type"),
                "overlap_percent": p.get("overlap", 0.5) * 100,  # Convert 0-1 to percent
            }

        if "cfar" in frontend_cfg:
            c = frontend_cfg["cfar"]
            backend_update["cfar"] = {
                "pfa": c.get("pfa"),
                "num_reference_cells": c.get("ref_cells"),
                "num_guard_cells": c.get("guard_cells"),
            }

        if "clustering" in frontend_cfg:
            backend_update["clustering"] = {
                "eps": frontend_cfg["clustering"].get("eps"),
                "min_samples": frontend_cfg["clustering"].get("min_samples"),
            }

        # Apply to pipeline
        if hasattr(self._pipeline, "update_config"):
            self._pipeline.update_config(backend_update)

    def get_sdr_config(self) -> dict[str, Any]:
        """Get current SDR configuration for recording metadata."""
        if self._pipeline is None or not hasattr(self._pipeline, "config"):
            return {"center_freq_hz": 915e6, "sample_rate_hz": 10e6, "gain_db": 40.0}
        cfg = self._pipeline.config
        return {
            "center_freq_hz": cfg.sdr.center_freq_hz,
            "sample_rate_hz": cfg.sdr.sample_rate_hz,
            "gain_db": cfg.sdr.gain_db,
        }


def create_rest_api(api_manager: RFForensicsAPI) -> FastAPI:
    """
    Create FastAPI application with modular REST endpoints.

    Uses FastAPI routers for clean organization:
    - system_router: /health, /api/start, /api/stop, /api/status
    - config_router: /api/config, /api/config/presets, /api/monitoring/*
    - detections_router: /api/detections/*, /api/clusters/*, /api/demod/*
    - recordings_router: /api/recordings/*, /api/recording/*
    - sdr_router: /api/sdr/* (devices, config, frequency, gain, bands)
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available")

    from rf_forensics.api.routers import (
        analysis_router,
        config_router,
        detections_router,
        recordings_router,
        sdr_router,
        system_router,
    )

    # Optional: simple analyze router under rf_forensics.backend (upload + decode + stub analysis)
    try:
        from rf_forensics.backend.routers.analyze import router as simple_analyze_router
    except Exception:
        simple_analyze_router = None

    app = FastAPI(
        title="RF Forensics Engine API",
        description="GPU-accelerated RF signal analysis and forensics",
        version="1.0.0",
    )

    # Store API manager in app state for dependency injection
    app.state.api_manager = api_manager

    # CORS middleware with whitelist (not wildcard)
    # Set CORS_ORIGINS env var to customize allowed origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )

    # Mount routers
    app.include_router(system_router)
    app.include_router(config_router)
    app.include_router(detections_router)
    app.include_router(recordings_router)
    app.include_router(sdr_router)
    app.include_router(analysis_router)
    if simple_analyze_router:
        app.include_router(simple_analyze_router)

    return app
