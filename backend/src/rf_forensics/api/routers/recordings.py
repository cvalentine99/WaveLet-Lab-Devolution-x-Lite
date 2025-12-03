"""
Recordings Router

Endpoints for I/Q recording management (SigMF format).
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from rf_forensics.api.dependencies import get_api_manager
from rf_forensics.api.frontend_adapter import RESTRequestAdapter, RESTResponseAdapter
from rf_forensics.api.schemas import (
    FrontendRecordingStartRequest,
    FrontendRecordingStopRequest,
    StartRecordingRequest,
    StartRecordingResponse,
    StopRecordingRequest,
    StopRecordingResponse,
)
from rf_forensics.api.validators import validate_recording_id

router = APIRouter(prefix="/api", tags=["Recordings"])


# Backend recording endpoints (snake_case)
@router.post("/recordings/start", response_model=StartRecordingResponse)
async def start_recording(recording_request: StartRecordingRequest, request: Request):
    """Start I/Q recording to SigMF format."""
    api_manager = get_api_manager(request)
    sdr_config = api_manager.get_sdr_config()
    recording_id = api_manager._recording_manager.start_recording(
        name=recording_request.name,
        description=recording_request.description,
        sdr_config=sdr_config,
        duration_seconds=recording_request.duration_seconds,
    )
    return {"recording_id": recording_id, "status": "recording"}


@router.post("/recordings/stop", response_model=StopRecordingResponse)
async def stop_recording(stop_request: StopRecordingRequest, request: Request):
    """Stop I/Q recording and generate SigMF metadata."""
    api_manager = get_api_manager(request)
    metadata = api_manager._recording_manager.stop_recording(stop_request.recording_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return {
        "recording_id": stop_request.recording_id,
        "status": "stopped",
        "file_size_bytes": metadata.file_size_bytes,
        "num_samples": metadata.num_samples,
    }


@router.get("/recordings")
async def list_recordings(request: Request):
    """List all SigMF recordings."""
    api_manager = get_api_manager(request)
    return {"recordings": api_manager._recording_manager.list_recordings()}


@router.get("/recordings/{recording_id}")
async def get_recording(
    recording_id: str = Depends(validate_recording_id), request: Request = None
):
    """Get details for a specific recording."""
    api_manager = get_api_manager(request)
    recording = api_manager._recording_manager.get_recording(recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return recording


@router.get("/recordings/{recording_id}/download")
async def download_recording(
    recording_id: str = Depends(validate_recording_id), request: Request = None
):
    """Download recording as SigMF ZIP archive."""
    api_manager = get_api_manager(request)
    zip_path = api_manager._recording_manager.create_download_zip(recording_id)
    if zip_path is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return FileResponse(
        path=str(zip_path), filename=f"{recording_id}.sigmf.zip", media_type="application/zip"
    )


@router.delete("/recordings/{recording_id}")
async def delete_recording(
    recording_id: str = Depends(validate_recording_id), request: Request = None
):
    """Delete a recording and its files."""
    api_manager = get_api_manager(request)
    success = api_manager._recording_manager.delete_recording(recording_id)
    if not success:
        raise HTTPException(status_code=404, detail="Recording not found or still active")
    return {"success": True}


# Frontend-compatible recording endpoints (camelCase)
@router.post("/recording/start")
async def start_recording_frontend(
    frontend_request: FrontendRecordingStartRequest, request: Request
):
    """
    Start recording (frontend-compatible format).

    Accepts filename/format/duration and returns recordingId/filepath.
    """
    api_manager = get_api_manager(request)

    # Parse frontend request
    backend_request = RESTRequestAdapter.parse_recording_start(frontend_request.model_dump())

    # Get SDR config for recording metadata
    sdr_config = api_manager.get_sdr_config()

    # Start recording
    recording_id = api_manager._recording_manager.start_recording(
        name=backend_request["name"],
        description=backend_request["description"],
        sdr_config=sdr_config,
        duration_seconds=backend_request["duration_seconds"],
    )

    # Get filepath
    filepath = f"/data/recordings/{recording_id}.sigmf-data"

    return RESTResponseAdapter.recording_start_response(
        recording_id=recording_id, filepath=filepath, success=True
    )


@router.post("/recording/stop")
async def stop_recording_frontend(frontend_request: FrontendRecordingStopRequest, request: Request):
    """Stop recording (frontend-compatible format)."""
    api_manager = get_api_manager(request)

    # Parse request
    backend_request = RESTRequestAdapter.parse_recording_stop(frontend_request.model_dump())
    recording_id = backend_request["recording_id"]

    # Stop recording
    metadata = api_manager._recording_manager.stop_recording(recording_id)

    if metadata is None:
        return RESTResponseAdapter.error_response(
            error_code="RECORDING_NOT_FOUND", message=f"Recording {recording_id} not found"
        )

    return RESTResponseAdapter.recording_stop_response(
        filepath=str(metadata.sigmf_data_path),
        file_size=metadata.file_size_bytes,
        duration=metadata.duration_seconds,
        success=True,
    )
