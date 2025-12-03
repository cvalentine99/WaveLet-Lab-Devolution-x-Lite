"""
Configuration Router

Endpoints for system configuration and presets.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from rf_forensics.api.dependencies import get_api_manager
from rf_forensics.api.frontend_adapter import RESTRequestAdapter, RESTResponseAdapter
from rf_forensics.api.schemas import FrontendConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Configuration"])


@router.get("/config", response_model=FrontendConfig)
async def get_config(request: Request):
    """Get current configuration in frontend format."""
    api_manager = get_api_manager(request)
    return api_manager.get_frontend_config()


@router.post("/config", response_model=FrontendConfig)
async def update_config(config: FrontendConfig, request: Request):
    """Update configuration (accepts partial config)."""
    api_manager = get_api_manager(request)
    api_manager.apply_frontend_config(config.model_dump(exclude_none=True))
    return api_manager.get_frontend_config()


@router.get("/config/presets")
async def list_presets():
    """List available configuration presets."""
    return {
        "presets": [
            {"name": "wideband_survey", "description": "20MHz span, fast sweep"},
            {"name": "narrowband_analysis", "description": "1MHz span, high resolution"},
            {"name": "burst_detection", "description": "Optimized for transient signals"},
            {"name": "ism_band_915", "description": "902-928MHz ISM band"},
        ]
    }


@router.post("/config/presets/{name}")
async def apply_preset(name: str, request: Request):
    """Apply a configuration preset."""
    from rf_forensics.config.schema import PRESETS

    api_manager = get_api_manager(request)

    if name not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    if api_manager._pipeline:
        api_manager._pipeline.apply_preset(name)
        return {"status": "applied", "preset": name}
    raise HTTPException(status_code=503, detail="Pipeline not initialized")


@router.post("/monitoring/start")
async def start_monitoring_frontend(request: Request):
    """
    Start monitoring (frontend-compatible format).

    Creates a session and starts spectrum/detection processing.
    """
    from rf_forensics.api.schemas import FrontendMonitoringStartRequest

    body = await request.json()
    monitoring_request = FrontendMonitoringStartRequest(**body)
    api_manager = get_api_manager(request)

    # Generate session ID
    api_manager._session_counter += 1
    session_id = f"session_{api_manager._session_counter:06d}"
    api_manager._active_session = session_id

    # Store monitoring config
    api_manager._monitoring_config = RESTRequestAdapter.parse_monitoring_start(
        monitoring_request.model_dump()
    )

    # Start pipeline if available
    success = True
    if api_manager._pipeline:
        try:
            await api_manager._pipeline.start()
        except RuntimeError as e:
            logger.error(f"Failed to start monitoring session {session_id}: {e}")
            success = False
        except Exception as e:
            logger.exception(f"Unexpected error starting monitoring session {session_id}: {e}")
            success = False

    return RESTResponseAdapter.monitoring_start_response(session_id=session_id, success=success)


@router.post("/monitoring/stop")
async def stop_monitoring_frontend(request: Request):
    """Stop monitoring (frontend-compatible format)."""
    api_manager = get_api_manager(request)
    session_id = api_manager._active_session

    success = True
    if api_manager._pipeline:
        try:
            await api_manager._pipeline.stop()
        except RuntimeError as e:
            logger.error(f"Failed to stop monitoring session {session_id}: {e}")
            success = False
        except Exception as e:
            logger.exception(f"Unexpected error stopping monitoring session {session_id}: {e}")
            success = False

    api_manager._active_session = None
    return RESTResponseAdapter.monitoring_stop_response(success=success)
