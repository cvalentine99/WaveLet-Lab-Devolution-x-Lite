from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from rf_forensics.backend.models.capture_descriptor import CaptureDescriptor
from rf_forensics.backend.utils.auto_detect import auto_detect_format
from rf_forensics.backend.utils.dsp_core import run_analysis_pipeline
from rf_forensics.backend.utils.file_storage import UPLOAD_DIR, save_uploaded_file
from rf_forensics.backend.utils.format_handlers import FORMAT_HANDLERS

router = APIRouter()


@router.post("/analyze")
async def analyze(
    descriptor: CaptureDescriptor = Depends(),
    file: UploadFile = File(...),
):
    """
    Upload + analyze an IQ file.

    - Saves the file to UPLOAD_DIR
    - Decodes using registered format handler (or auto-detect)
    - Runs analysis pipeline and returns results
    """
    saved_path = await save_uploaded_file(file)

    fmt = descriptor.format
    if fmt not in FORMAT_HANDLERS:
        try:
            fmt = auto_detect_format(str(saved_path))
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Unsupported format and auto-detect failed: {e}"
            )

    decoder = FORMAT_HANDLERS.get(fmt)
    if decoder is None:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    try:
        iq = decoder(saved_path, descriptor)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode IQ: {e}")

    try:
        results = run_analysis_pipeline(iq, descriptor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return {
        "status": "ok",
        "format_used": fmt,
        "file": file.filename,
        "upload_path": str(saved_path),
        "upload_dir": str(UPLOAD_DIR),
        "analysis": results,
    }
