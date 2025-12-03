"""
Analysis Router

Endpoints to ingest uploaded IQ files and run offline analysis
through the RF Forensics processing stack (PSD + CFAR + peaks).

This is a first-cut implementation that:
  - Loads files previously uploaded by the frontend (Node/tRPC) into an uploads directory
  - Decodes IQ based on requested format or SigMF metadata
  - Runs PSD/CFAR/peak detection using existing components
  - Returns detections and an optional downsampled PSD slice

Note: This does not start the live SDR pipeline; it processes files offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from rf_forensics.detection.cfar import CFARDetector
from rf_forensics.detection.peaks import PeakDetector
from rf_forensics.dsp.psd import WelchPSD

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

# Base directory where the Node layer stores uploaded files
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", Path(__file__).parent.parent / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _load_sigmf(base_path: Path):
    """
    Load SigMF data/metadata.

    Expects .sigmf-data and .sigmf-meta with the same basename.
    Returns: (iq np.ndarray complex64, sample_rate_hz, center_freq_hz)
    """
    meta_path = base_path.with_suffix(".sigmf-meta")
    data_path = base_path.with_suffix(".sigmf-data")

    if not meta_path.exists() or not data_path.exists():
        raise FileNotFoundError("SigMF data/meta not found")

    with open(meta_path) as f:
        meta = json.load(f)

    global_meta = meta.get("global", {})
    captures = meta.get("captures", [{}])

    dtype = global_meta.get("core:datatype", "cf32_le")
    sample_rate = global_meta.get("core:sample_rate")
    center_freq = captures[0].get("core:frequency", 0) if captures else 0

    if "cf32" in dtype:
        iq = np.fromfile(data_path, dtype=np.complex64)
    elif "ci16" in dtype:
        raw = np.fromfile(data_path, dtype=np.int16).reshape(-1, 2)
        iq = raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)
    else:
        raise ValueError(f"Unsupported SigMF datatype: {dtype}")

    return iq, sample_rate, center_freq


def _load_iq(path: Path, fmt: str, sample_rate_hz: float | None, center_freq_hz: float | None):
    """Load IQ data from disk based on format."""
    if fmt == "sigmf":
        return _load_sigmf(path)

    if fmt in ("complex64", "bin"):
        iq = np.fromfile(path, dtype=np.complex64)
    elif fmt == "int16_iq":
        raw = np.fromfile(path, dtype=np.int16).reshape(-1, 2)
        iq = raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)
    elif fmt == "uint8":
        raw = np.fromfile(path, dtype=np.uint8).reshape(-1, 2)
        iq = (raw[:, 0].astype(np.float32) - 128) + 1j * (raw[:, 1].astype(np.float32) - 128)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return iq, sample_rate_hz, center_freq_hz


def _downsample_psd(psd_db: np.ndarray, max_bins: int = 1024) -> list[float]:
    """Downsample/decimate PSD to a manageable size for response."""
    n = len(psd_db)
    if n <= max_bins:
        return psd_db.astype(np.float32).tolist()

    factor = n // max_bins
    reshaped = psd_db[: factor * max_bins].reshape(max_bins, factor)
    decimated = reshaped.mean(axis=1)
    return decimated.astype(np.float32).tolist()


@router.post("/start")
def start_analysis(payload: dict):
    """
    Process a previously uploaded IQ file offline.

    Expected input:
    {
      "analysis_id": str,
      "format": "complex64" | "int16_iq" | "uint8" | "sigmf",
      "sample_rate_hz"?: number,
      "center_freq_hz"?: number,
      "fft_size"?: number
    }
    """
    analysis_id = payload.get("analysis_id")
    fmt = payload.get("format")
    sample_rate_hz = payload.get("sample_rate_hz")
    center_freq_hz = payload.get("center_freq_hz")
    fft_size = int(payload.get("fft_size") or 2048)

    if not analysis_id or not fmt:
        raise HTTPException(status_code=400, detail="analysis_id and format are required")

    # Find the file written by the Node uploader (pattern: {analysis_id}_*)
    matches = list(UPLOAD_DIR.glob(f"{analysis_id}_*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Analysis file not found")

    file_path = matches[0]

    try:
        iq, sr, cf = _load_iq(file_path, fmt, sample_rate_hz, center_freq_hz)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load IQ: {e}")

    if sample_rate_hz is None:
        sample_rate_hz = sr
    if center_freq_hz is None:
        center_freq_hz = cf

    if sample_rate_hz is None:
        raise HTTPException(status_code=400, detail="sample_rate_hz is required")

    # PSD and detection
    psd = WelchPSD(fft_size=fft_size, sample_rate=sample_rate_hz)
    peak_detector = PeakDetector()
    cfar = CFARDetector()

    freqs, psd_lin = psd.compute_psd(iq)
    psd_db = 10 * np.log10(psd_lin + psd.log_floor)
    mask, snr = cfar.detect(psd_lin)
    detections = peak_detector.find_peaks(psd_db, mask, psd.freq_axis_cpu)

    # Convert detections to serializable dicts
    det_list = []
    for det in detections:
        det_list.append(
            {
                "detection_id": det.detection_id,
                "center_freq_hz": det.center_freq_hz,
                "bandwidth_hz": det.bandwidth_hz,
                "bandwidth_3db_hz": getattr(det, "bandwidth_3db_hz", None),
                "bandwidth_6db_hz": getattr(det, "bandwidth_6db_hz", None),
                "peak_power_db": det.peak_power_db,
                "snr_db": det.snr_db,
                "start_bin": det.start_bin,
                "end_bin": det.end_bin,
                "timestamp": det.timestamp,
            }
        )

    response = {
        "analysis_id": analysis_id,
        "status": "complete",
        "sample_rate_hz": sample_rate_hz,
        "center_freq_hz": center_freq_hz,
        "fft_size": fft_size,
        "detections": det_list,
        "spectrum": {
            "magnitude_db": _downsample_psd(psd_db),
            "fft_size": len(psd_db),
            "sample_rate_hz": sample_rate_hz,
            "center_freq_hz": center_freq_hz,
        },
    }

    return response


@router.post("/upload")
async def upload_analysis_file(
    file: UploadFile = File(...),
    fmt: str = Form(..., description="complex64 | int16_iq | uint8 | sigmf"),
    analysis_id: str | None = Form(None),
):
    """
    Upload an IQ/SigMF file for offline analysis.

    Saves the file to UPLOAD_DIR with pattern {analysis_id}_{filename}.
    Returns the generated analysis_id and file metadata.
    """
    fmt = fmt.lower()
    if fmt not in {"complex64", "int16_iq", "uint8", "sigmf", "bin"}:
        raise HTTPException(status_code=400, detail="Unsupported format")

    aid = analysis_id or uuid4().hex
    safe_name = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in file.filename or "upload.bin"
    )
    dest_path = UPLOAD_DIR / f"{aid}_{safe_name}"

    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    size_bytes = dest_path.stat().st_size if dest_path.exists() else 0

    return {
        "analysis_id": aid,
        "filename": safe_name,
        "file_size_bytes": size_bytes,
        "format": fmt,
        "upload_dir": str(UPLOAD_DIR),
        "file_path": str(dest_path),
    }
