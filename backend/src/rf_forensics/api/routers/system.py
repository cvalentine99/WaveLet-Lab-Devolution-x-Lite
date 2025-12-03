"""
System Control Router

Endpoints for pipeline lifecycle and system status.
"""

import logging
import subprocess

from fastapi import APIRouter, Request

from rf_forensics.api.dependencies import get_api_manager
from rf_forensics.api.frontend_adapter import RESTResponseAdapter
from rf_forensics.api.schemas import StatusModel, SuccessResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])


def _check_gpu_status() -> dict:
    """Check GPU availability and memory."""
    result = {"available": False, "name": "No GPU", "memory_used_gb": 0.0, "memory_total_gb": 0.0}
    try:
        proc_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc_result.returncode == 0:
            parts = proc_result.stdout.strip().split(",")
            result["available"] = True
            result["name"] = parts[0].strip() if len(parts) > 0 else "Unknown GPU"
            result["memory_used_gb"] = float(parts[1].strip()) / 1024 if len(parts) > 1 else 0.0
            result["memory_total_gb"] = float(parts[2].strip()) / 1024 if len(parts) > 2 else 0.0
    except FileNotFoundError:
        logger.debug("nvidia-smi not found - no NVIDIA GPU available")
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out - GPU may be busy or unresponsive")
    except Exception as e:
        logger.warning(f"Unexpected error checking GPU status: {e}")
    return result


def _check_memory_status() -> dict:
    """Check system memory availability."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "available_gb": mem.available / (1024**3),
            "total_gb": mem.total / (1024**3),
            "percent_used": mem.percent,
        }
    except ImportError:
        return {"available_gb": 0.0, "total_gb": 0.0, "percent_used": 0.0}
    except Exception as e:
        logger.warning(f"Error checking memory status: {e}")
        return {"available_gb": 0.0, "total_gb": 0.0, "percent_used": 0.0}


@router.get("/health")
async def health_check():
    """
    Health check endpoint (frontend-compatible format).

    Returns GPU status and backend version.
    """
    gpu_status = _check_gpu_status()

    return RESTResponseAdapter.health_response(
        gpu_available=gpu_status["available"], gpu_name=gpu_status["name"], backend_version="1.0.0"
    )


@router.get("/health/detailed")
async def detailed_health_check(request: Request):
    """
    Detailed health check with comprehensive system status.

    Returns status of all major subsystems.
    """
    api_manager = get_api_manager(request)
    gpu_status = _check_gpu_status()
    memory_status = _check_memory_status()

    # Check SDR connection
    sdr_connected = False
    sdr_device = None
    try:
        if hasattr(api_manager, "_sdr_driver") and api_manager._sdr_driver:
            sdr_connected = api_manager._sdr_driver.is_connected
            if sdr_connected:
                sdr_device = getattr(api_manager._sdr_driver, "device_id", "connected")
    except Exception as e:
        logger.warning(f"Error checking SDR status: {e}")

    # Check pipeline status
    pipeline_initialized = False
    pipeline_state = "not_initialized"
    try:
        if api_manager._pipeline:
            pipeline_initialized = True
            pipeline_state = api_manager._pipeline.state.value
    except Exception as e:
        logger.warning(f"Error checking pipeline status: {e}")

    return {
        "status": "healthy" if gpu_status["available"] and pipeline_initialized else "degraded",
        "timestamp": __import__("time").time(),
        "version": "1.0.0",
        "checks": {
            "gpu": {
                "status": "ok" if gpu_status["available"] else "unavailable",
                "name": gpu_status["name"],
                "memory_used_gb": round(gpu_status["memory_used_gb"], 2),
                "memory_total_gb": round(gpu_status["memory_total_gb"], 2),
            },
            "sdr": {
                "status": "connected" if sdr_connected else "disconnected",
                "device": sdr_device,
            },
            "memory": {
                "status": "ok" if memory_status["percent_used"] < 90 else "warning",
                "available_gb": round(memory_status["available_gb"], 2),
                "percent_used": round(memory_status["percent_used"], 1),
            },
            "pipeline": {
                "status": "ok" if pipeline_initialized else "not_initialized",
                "state": pipeline_state,
            },
        },
    }


@router.post("/api/start", response_model=SuccessResponse)
async def start_acquisition(request: Request):
    """Start signal acquisition and processing."""
    api_manager = get_api_manager(request)
    try:
        if api_manager._pipeline:
            await api_manager._pipeline.start()
        return {"success": True}
    except RuntimeError as e:
        logger.error(f"Failed to start acquisition: {e}")
        return {"success": False}
    except Exception as e:
        logger.exception(f"Unexpected error starting acquisition: {e}")
        return {"success": False}


@router.post("/api/stop", response_model=SuccessResponse)
async def stop_acquisition(request: Request):
    """Stop signal acquisition."""
    api_manager = get_api_manager(request)
    try:
        if api_manager._pipeline:
            await api_manager._pipeline.stop()
        return {"success": True}
    except RuntimeError as e:
        logger.error(f"Failed to stop acquisition: {e}")
        return {"success": False}
    except Exception as e:
        logger.exception(f"Unexpected error stopping acquisition: {e}")
        return {"success": False}


@router.post("/api/pause", response_model=SuccessResponse)
async def pause_acquisition(request: Request):
    """Pause processing (keep SDR running)."""
    api_manager = get_api_manager(request)
    try:
        if api_manager._pipeline:
            await api_manager._pipeline.pause()
        return {"success": True}
    except RuntimeError as e:
        logger.error(f"Failed to pause acquisition: {e}")
        return {"success": False}
    except Exception as e:
        logger.exception(f"Unexpected error pausing acquisition: {e}")
        return {"success": False}


@router.post("/api/resume", response_model=SuccessResponse)
async def resume_acquisition(request: Request):
    """Resume processing."""
    api_manager = get_api_manager(request)
    try:
        if api_manager._pipeline:
            await api_manager._pipeline.resume()
        return {"success": True}
    except RuntimeError as e:
        logger.error(f"Failed to resume acquisition: {e}")
        return {"success": False}
    except Exception as e:
        logger.exception(f"Unexpected error resuming acquisition: {e}")
        return {"success": False}


@router.get("/api/status", response_model=StatusModel)
async def get_status(request: Request):
    """Get current system status."""
    api_manager = get_api_manager(request)
    return api_manager.get_status()


# =========================================================================
# GPS Status Endpoint
# =========================================================================


@router.get("/api/gps/status")
async def get_gps_status(request: Request):
    """
    Get GPS module status from uSDR DevBoard.

    The DevBoard has an onboard GPS module that provides:
    - Position (latitude/longitude)
    - Time synchronization
    - Satellite info

    Returns simulated data if GPS not available or not locked.
    """
    import time

    from rf_forensics.sdr.manager import get_sdr_manager

    manager = get_sdr_manager()

    # Default GPS status (no lock)
    gps_status = {
        "locked": False,
        "latitude": None,
        "longitude": None,
        "altitude_m": None,
        "satellites_visible": 0,
        "satellites_used": 0,
        "hdop": None,
        "utc_time": None,
        "fix_type": "none",  # none, 2d, 3d, dgps
        "speed_knots": None,
        "heading_deg": None,
        "last_update": time.time(),
    }

    # Try to get real GPS data from SDR if connected
    if manager.is_connected:
        try:
            driver = manager.get_driver()
            if driver and hasattr(driver, "get_gps_status"):
                real_gps = driver.get_gps_status()
                if real_gps:
                    gps_status.update(real_gps)
        except Exception as e:
            logger.debug(f"Could not get GPS status: {e}")

    # Check if GPS is enabled on DevBoard
    # The gps_ VFS parameter controls GPS module power
    gps_enabled = False
    if manager.is_connected:
        try:
            driver = manager.get_driver()
            if driver and driver._dev:
                # Try to read GPS enable status
                gps_enabled = True  # Assume enabled if connected
        except Exception:
            pass

    gps_status["gps_enabled"] = gps_enabled

    return gps_status


# =========================================================================
# System Health Endpoint (for SystemStatusPanel)
# =========================================================================


@router.get("/api/system/health")
async def get_system_health(request: Request):
    """
    Get comprehensive system health for the dashboard SystemStatusPanel.

    Returns:
    - GPU: name, memory usage, utilization, CUDA version
    - Hardware: PLL lock, uptime, temperature, overflow events, reconnects
    - Pipeline: state, throughput, latency
    """
    import time

    from rf_forensics.sdr.manager import get_sdr_manager

    api_manager = get_api_manager(request)
    manager = get_sdr_manager()
    gpu_status = _check_gpu_status()

    # Get GPU utilization
    gpu_utilization = 0.0
    cuda_version = "N/A"
    try:
        proc_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc_result.returncode == 0:
            parts = proc_result.stdout.strip().split(",")
            gpu_utilization = float(parts[0].strip()) if len(parts) > 0 else 0.0
            cuda_version = parts[1].strip() if len(parts) > 1 else "N/A"
    except Exception as e:
        logger.debug(f"Could not get GPU utilization: {e}")

    # Try to get CUDA version from toolkit
    try:
        import cupy as cp

        cuda_version = f"CUDA {cp.cuda.runtime.runtimeGetVersion()}"
    except Exception:
        pass

    # Get SDR metrics
    sdr_metrics = manager.get_metrics() if manager.is_connected else None

    # Get pipeline status
    pipeline_state = "not_initialized"
    pipeline_uptime = 0.0
    throughput_msps = 0.0
    latency_ms = 0.0
    try:
        if api_manager._pipeline:
            pipeline_state = api_manager._pipeline.state.value
            status = api_manager.get_status()
            pipeline_uptime = status.get("uptime_seconds", 0.0)
            throughput_msps = status.get("current_throughput_msps", 0.0)
            latency_ms = status.get("processing_latency_ms", 0.0)
    except Exception as e:
        logger.debug(f"Could not get pipeline status: {e}")

    return {
        "timestamp": time.time(),
        "gpu": {
            "name": gpu_status["name"],
            "available": gpu_status["available"],
            "memory_used_gb": round(gpu_status["memory_used_gb"], 2),
            "memory_total_gb": round(gpu_status["memory_total_gb"], 2),
            "memory_percent": round(
                (gpu_status["memory_used_gb"] / gpu_status["memory_total_gb"] * 100)
                if gpu_status["memory_total_gb"] > 0
                else 0,
                1,
            ),
            "utilization_percent": gpu_utilization,
            "cuda_version": cuda_version,
        },
        "hardware": {
            "pll_locked": sdr_metrics.pll_locked if sdr_metrics else False,
            "temperature_c": sdr_metrics.temperature_c if sdr_metrics else 0.0,
            "uptime_seconds": sdr_metrics.streaming_uptime_seconds if sdr_metrics else 0.0,
            "overflow_events": sdr_metrics.total_overflows if sdr_metrics else 0,
            "reconnect_count": sdr_metrics.reconnect_count if sdr_metrics else 0,
            "sdr_connected": manager.is_connected,
            "sdr_streaming": manager.is_streaming,
        },
        "pipeline": {
            "state": pipeline_state,
            "uptime_seconds": pipeline_uptime,
            "throughput_msps": round(throughput_msps, 2),
            "latency_ms": round(latency_ms, 2),
        },
    }
