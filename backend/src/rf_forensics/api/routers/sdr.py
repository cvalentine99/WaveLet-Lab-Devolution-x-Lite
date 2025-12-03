"""
SDR Router

Endpoints for SDR device management and configuration.
Includes duplexer band selection for uSDR DevBoard.

NOTE: This router fixes the duplicate /api/sdr/config endpoint issue
by using a single unified handler that accepts both formats.

Architecture: All endpoints use SDRManager singleton for unified driver ownership.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from rf_forensics.api.schemas import (
    DuplexerBandsResponse,
    SDRConnectRequest,
    SDRConnectResponse,
    SDRDeviceListResponse,
    SDRFullConfigModel,
    SDRStatusModel,
    SetBandRequest,
    SuccessResponse,
)

logger = logging.getLogger(__name__)


# Request models for frontend compatibility
class FrequencyRequest(BaseModel):
    """Frequency change request - accepts both camelCase and snake_case."""

    centerFreqHz: int | None = None
    center_freq_hz: int | None = None


class RxPathRequest(BaseModel):
    """RX path change request - accepts both camelCase and snake_case."""

    rxPath: str | None = None
    path: str | None = None


class GainRequest(BaseModel):
    """Gain change request."""

    lna_db: int | None = None
    tia_db: int | None = None
    pga_db: int | None = None


from rf_forensics.api.frontend_adapter import RESTRequestAdapter, RESTResponseAdapter
from rf_forensics.api.validators import validate_frequency, validate_sample_rate
from rf_forensics.sdr.manager import get_sdr_manager
from rf_forensics.sdr.usdr_driver import (
    USDRConfig,
    USDRGain,
    get_band_by_name,
)

router = APIRouter(prefix="/api/sdr", tags=["SDR"])


# =========================================================================
# Device Discovery and Connection
# =========================================================================


@router.get("/devices", response_model=SDRDeviceListResponse)
async def discover_sdr_devices():
    """
    Discover available SDR devices.

    If already connected, returns the connected device info.
    Otherwise performs device discovery via usdr-lib.
    """
    manager = get_sdr_manager()

    # If already connected, return connected device info
    if manager.is_connected:
        try:
            driver = manager.get_driver()
            status = driver.status_to_dict()
            return {
                "devices": [
                    {
                        "id": status.get("device_id", "usdr:0"),
                        "model": "uSDR DevBoard",
                        "serial": "",
                        "status": "connected",
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error getting connected device info: {e}")

    # Otherwise discover available devices
    try:
        devices = manager.discover()
        return {
            "devices": [
                {"id": dev.id, "model": dev.model, "serial": dev.serial, "status": dev.status}
                for dev in devices
            ]
        }
    except Exception as e:
        logger.error(f"Device discovery failed: {e}")
        return {"devices": []}


@router.post("/connect", response_model=SDRConnectResponse)
async def connect_sdr(request: SDRConnectRequest):
    """Connect to an SDR device."""
    manager = get_sdr_manager()
    success = manager.connect(request.device_id)
    return {"success": success, "device_id": request.device_id if success else ""}


@router.post("/disconnect", response_model=SuccessResponse)
async def disconnect_sdr():
    """Disconnect from current SDR device."""
    manager = get_sdr_manager()
    success = manager.disconnect()
    return {"success": success}


@router.get("/status", response_model=SDRStatusModel)
async def get_sdr_status():
    """Get real-time SDR hardware status."""
    manager = get_sdr_manager()
    driver = manager.get_driver()
    return driver.status_to_dict()


# =========================================================================
# Configuration (UNIFIED - accepts both formats)
# =========================================================================


@router.get("/config", response_model=SDRFullConfigModel)
async def get_sdr_config():
    """Get current SDR configuration (full USDR format)."""
    manager = get_sdr_manager()
    driver = manager.get_driver()
    return driver.to_dict()


@router.post("/config")
async def update_sdr_config(request: Request):
    """
    Update SDR configuration.

    Accepts BOTH formats:
    - Backend format (snake_case): SDRFullConfigModel with nested gain object
    - Frontend format (camelCase): FrontendSDRConfigRequest with flat fields

    The format is auto-detected based on the presence of camelCase keys.
    """
    from rf_forensics.api.dependencies import get_api_manager

    manager = get_sdr_manager()
    driver = manager.get_driver()
    body = await request.json()

    # Detect format by checking for camelCase keys
    is_frontend_format = any(key in body for key in ["centerFreqHz", "sampleRateHz", "gainDb"])

    if is_frontend_format:
        # Frontend format - parse and adapt
        backend_config = RESTRequestAdapter.parse_sdr_config(body)

        # Validate frequency and sample rate bounds
        validate_frequency(int(backend_config["center_freq_hz"]))
        validate_sample_rate(int(backend_config["sample_rate_hz"]))

        config = USDRConfig(
            center_freq_hz=backend_config["center_freq_hz"],
            sample_rate_hz=backend_config["sample_rate_hz"],
            bandwidth_hz=backend_config.get("bandwidth_hz") or backend_config["sample_rate_hz"],
            gain=USDRGain(
                lna_db=15,
                tia_db=9,
                pga_db=int(backend_config.get("gain_db", 36) - 24)
                if backend_config.get("gain_db")
                else 12,
            ),
        )

        success = manager.configure(config)

        # Sync config to pipeline so DSP components stay in sync
        api_manager = get_api_manager(request)
        if api_manager._pipeline and hasattr(api_manager._pipeline, "update_config"):
            api_manager._pipeline.update_config(
                {
                    "sdr": {
                        "center_freq_hz": config.center_freq_hz,
                        "sample_rate_hz": config.sample_rate_hz,
                        "bandwidth_hz": config.bandwidth_hz,
                        "gain_db": config.gain.total_db,
                    }
                }
            )

        return RESTResponseAdapter.sdr_config_response(
            config=backend_config,
            success=success,
            message="SDR configured successfully" if success else "Failed to configure SDR",
        )

    else:
        # Backend format - full USDR config
        parsed = SDRFullConfigModel(**body)

        # Validate frequency and sample rate bounds
        validate_frequency(parsed.center_freq_hz)
        validate_sample_rate(parsed.sample_rate_hz)

        gain = USDRGain(
            lna_db=parsed.gain.lna_db, tia_db=parsed.gain.tia_db, pga_db=parsed.gain.pga_db
        )

        usdr_config = USDRConfig(
            center_freq_hz=parsed.center_freq_hz,
            sample_rate_hz=parsed.sample_rate_hz,
            bandwidth_hz=parsed.bandwidth_hz,
            gain=gain,
            rx_path=parsed.rx_path,
        )

        # Apply devboard settings if provided (see docs.wsdr.io/hardware/devboard.html)
        if parsed.devboard:
            usdr_config.lna_enable = parsed.devboard.lna_enable
            usdr_config.pa_enable = parsed.devboard.pa_enable
            usdr_config.attenuator_db = parsed.devboard.attenuator_db
            usdr_config.vctcxo_dac = parsed.devboard.vctcxo_dac
            usdr_config.gps_enable = parsed.devboard.gps_enable
            usdr_config.osc_enable = parsed.devboard.osc_enable
            usdr_config.loopback_enable = parsed.devboard.loopback_enable
            usdr_config.uart_enable = parsed.devboard.uart_enable

        manager.configure(usdr_config)

        # Sync config to pipeline so DSP components stay in sync
        api_manager = get_api_manager(request)
        if api_manager._pipeline and hasattr(api_manager._pipeline, "update_config"):
            api_manager._pipeline.update_config(
                {
                    "sdr": {
                        "center_freq_hz": usdr_config.center_freq_hz,
                        "sample_rate_hz": usdr_config.sample_rate_hz,
                        "bandwidth_hz": usdr_config.bandwidth_hz,
                        "gain_db": usdr_config.gain.total_db,
                    }
                }
            )

        return driver.to_dict()


# =========================================================================
# Quick Configuration Adjustments
# =========================================================================


@router.post("/frequency")
async def set_sdr_frequency(request: FrequencyRequest):
    """
    Quick frequency change.

    Accepts JSON body with either:
    - centerFreqHz (camelCase from frontend)
    - center_freq_hz (snake_case)
    """
    # Accept either camelCase or snake_case
    freq_hz = request.centerFreqHz or request.center_freq_hz
    if not freq_hz:
        raise HTTPException(status_code=400, detail="centerFreqHz or center_freq_hz required")

    # Validate frequency bounds
    validate_frequency(freq_hz)

    manager = get_sdr_manager()
    if not manager.is_connected:
        raise HTTPException(status_code=400, detail="SDR not connected")
    driver = manager.get_driver()
    success = driver.set_frequency(freq_hz)
    return {"success": success, "freq_hz": freq_hz}


@router.post("/gain")
async def set_sdr_gain(lna_db: int = None, tia_db: int = None, pga_db: int = None):
    """Quick gain adjustment."""
    manager = get_sdr_manager()
    if not manager.is_connected:
        raise HTTPException(status_code=400, detail="SDR not connected")
    driver = manager.get_driver()
    success = driver.set_gain(lna_db=lna_db, tia_db=tia_db, pga_db=pga_db)
    cfg = driver.get_config()
    return {
        "success": success,
        "gain": {
            "lna_db": cfg.gain.lna_db,
            "tia_db": cfg.gain.tia_db,
            "pga_db": cfg.gain.pga_db,
            "total_db": cfg.gain.total_db,
        },
    }


@router.post("/rx_path")
async def set_sdr_rx_path(request: RxPathRequest):
    """
    Set RX antenna path (LNAH, LNAL, LNAW).

    Accepts JSON body with either:
    - rxPath (camelCase from frontend)
    - path (snake_case)
    """
    # Accept either camelCase or snake_case
    rx_path = request.rxPath or request.path
    if not rx_path:
        raise HTTPException(status_code=400, detail="rxPath or path required")

    manager = get_sdr_manager()
    if not manager.is_connected:
        raise HTTPException(status_code=400, detail="SDR not connected")
    if rx_path not in ("LNAH", "LNAL", "LNAW"):
        raise HTTPException(status_code=400, detail="Invalid RX path. Use LNAH, LNAL, or LNAW")
    driver = manager.get_driver()
    success = driver.set_rx_path(rx_path)
    return {"success": success, "rx_path": rx_path}


# =========================================================================
# Duplexer Band Selection (DevBoard Special Bands)
# =========================================================================


@router.get("/bands", response_model=DuplexerBandsResponse)
async def list_duplexer_bands(category: str = None):
    """
    List available duplexer bands for the DevBoard.

    Categories:
    - cellular: FDD cellular bands (Band 2/3/5/7/8, GSM, PCS, DCS)
    - tx_only: TX-only paths with different LPF cutoffs
    - rx_only: RX-only paths with LPF/BPF options
    - tdd: TDD/half-duplex modes for different frequency ranges
    """
    manager = get_sdr_manager()
    driver = manager.get_driver()
    bands = driver.get_duplexer_bands(category)
    return {"bands": bands, "categories": ["cellular", "tx_only", "rx_only", "tdd"]}


@router.get("/bands/{band_name}")
async def get_band_info(band_name: str):
    """Get details for a specific duplexer band."""
    band = get_band_by_name(band_name)
    if band is None:
        raise HTTPException(status_code=404, detail=f"Band '{band_name}' not found")
    return {
        "name": band.name,
        "aliases": band.aliases,
        "category": band.category,
        "freq_range_mhz": list(band.freq_range_mhz),
        "description": band.description,
        "pa_enable": band.pa_enable,
        "lna_enable": band.lna_enable,
        "trx_mode": band.trx_mode,
        "rx_filter": band.rx_filter,
        "tx_filter": band.tx_filter,
    }


@router.post("/bands")
async def set_duplexer_band(request: SetBandRequest):
    """
    Set duplexer band configuration.

    Accepts band name or alias (e.g., "band2", "gsm900", "pcs", "trx0_400").
    This configures the DevBoard's duplexer, filters, PA, and LNA automatically.
    """
    manager = get_sdr_manager()
    if not manager.is_connected:
        raise HTTPException(status_code=400, detail="SDR not connected")

    band = get_band_by_name(request.band)
    if band is None:
        raise HTTPException(status_code=400, detail=f"Unknown band: {request.band}")

    driver = manager.get_driver()
    success = driver.set_duplexer_band(request.band)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to set duplexer band")

    return {
        "success": True,
        "band": band.name,
        "category": band.category,
        "description": band.description,
        "freq_range_mhz": list(band.freq_range_mhz),
        "pa_enable": band.pa_enable,
        "lna_enable": band.lna_enable,
    }


# =========================================================================
# Observability: Metrics and Capabilities
# =========================================================================


@router.get("/metrics")
async def get_sdr_metrics():
    """
    Get detailed SDR hardware metrics.

    Returns real-time metrics including:
    - Overflow tracking (total overflows, rate per second)
    - Sample tracking (received, dropped, drop rate)
    - Hardware health (temperature, PLL lock)
    - Streaming metrics (uptime, reconnect count)
    - Backpressure events
    """
    manager = get_sdr_manager()
    metrics = manager.get_metrics()
    return metrics.to_dict()


@router.get("/capabilities")
async def get_sdr_capabilities():
    """
    Get SDR hardware capabilities.

    Returns actual hardware-supported ranges:
    - Frequency range (Hz)
    - Sample rate range (Hz)
    - Bandwidth range (Hz)
    - Gain range (dB)
    - Available RX paths
    - Supported formats
    - Feature flags (GPUDirect support, etc.)
    """
    manager = get_sdr_manager()
    caps = manager.get_capabilities()
    return caps.to_dict()


@router.get("/health")
async def get_sdr_health():
    """
    Get comprehensive SDR health status.

    Combines connection status, streaming status, and metrics
    into a single health check endpoint.
    """
    manager = get_sdr_manager()
    metrics = manager.get_metrics()

    # Determine health status
    health_status = "healthy"
    warnings = []

    if not manager.is_available:
        health_status = "unavailable"
        warnings.append("SDR library not available")
    elif not manager.is_connected:
        health_status = "disconnected"
        warnings.append("SDR not connected")
    else:
        # Check for issues
        if metrics.overflow_rate_per_sec > 1.0:
            health_status = "degraded"
            warnings.append(f"High overflow rate: {metrics.overflow_rate_per_sec:.1f}/s")

        if metrics.drop_rate_percent > 1.0:
            health_status = "degraded"
            warnings.append(f"High drop rate: {metrics.drop_rate_percent:.1f}%")

        if metrics.current_buffer_fill_percent > 80:
            health_status = "degraded"
            warnings.append(f"Buffer near full: {metrics.current_buffer_fill_percent:.1f}%")

        if metrics.temperature_c > 70:
            warnings.append(f"High temperature: {metrics.temperature_c:.1f}C")

    return {
        "status": health_status,
        "connected": manager.is_connected,
        "streaming": manager.is_streaming,
        "warnings": warnings,
        "metrics_summary": {
            "overflow_rate": metrics.overflow_rate_per_sec,
            "drop_rate_percent": metrics.drop_rate_percent,
            "buffer_fill_percent": metrics.current_buffer_fill_percent,
            "temperature_c": metrics.temperature_c,
            "uptime_seconds": metrics.streaming_uptime_seconds,
        },
    }


# =========================================================================
# Stream Control (per-device streaming start/stop)
# =========================================================================


@router.post("/devices/{device_id}/stream/start")
async def start_device_stream(device_id: str, request: Request):
    """
    Start streaming from a specific SDR device.

    This endpoint:
    1. Connects to the device if not already connected
    2. Allocates buffers and initializes DMA
    3. Starts GPU processing pipeline
    4. Begins WebSocket spectrum/detection updates

    The frontend should call this after user presses "Start Stream".
    """
    from rf_forensics.api.dependencies import get_api_manager

    manager = get_sdr_manager()

    # Connect to device if not already connected
    if not manager.is_connected:
        success = manager.connect(device_id)
        if not success:
            raise HTTPException(status_code=503, detail=f"Failed to connect to device: {device_id}")

    # Get API manager to start the pipeline
    api_manager = get_api_manager(request)

    # Start the pipeline (which starts streaming and WebSocket updates)
    if api_manager._pipeline:
        try:
            import asyncio

            if asyncio.iscoroutinefunction(api_manager._pipeline.start):
                await api_manager._pipeline.start()
            else:
                api_manager._pipeline.start()
            logger.info(f"Started streaming from device: {device_id}")
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start streaming: {e}")
    else:
        # No pipeline, just start SDR streaming directly
        logger.warning("No pipeline available, starting SDR streaming only")

    return {
        "success": True,
        "device_id": device_id,
        "streaming": manager.is_streaming,
        "message": "Streaming started",
    }


@router.post("/devices/{device_id}/stream/stop")
async def stop_device_stream(device_id: str, request: Request):
    """
    Stop streaming from a specific SDR device.

    This endpoint:
    1. Stops WebSocket spectrum/detection updates
    2. Halts GPU processing pipeline
    3. Stops DMA transfers
    4. Does NOT disconnect the device (user can restart without re-selecting)

    The frontend should call this after user presses "Stop Stream".
    """
    from rf_forensics.api.dependencies import get_api_manager

    manager = get_sdr_manager()
    api_manager = get_api_manager(request)

    # Stop the pipeline (which stops streaming and WebSocket updates)
    if api_manager._pipeline:
        try:
            import asyncio

            if asyncio.iscoroutinefunction(api_manager._pipeline.stop):
                await api_manager._pipeline.stop()
            else:
                api_manager._pipeline.stop()
            logger.info(f"Stopped streaming from device: {device_id}")
        except Exception as e:
            logger.error(f"Failed to stop pipeline: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to stop streaming: {e}")
    else:
        # No pipeline, just stop SDR streaming directly
        manager.stop_streaming()

    return {
        "success": True,
        "device_id": device_id,
        "streaming": False,
        "message": "Streaming stopped",
    }
