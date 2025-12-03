"""
Detections Router

Endpoints for detection and cluster data access.
"""

import json

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from rf_forensics.api.dependencies import get_api_manager

router = APIRouter(prefix="/api", tags=["Detections"])


# Detection endpoints
@router.get("/detections")
async def list_detections(
    request: Request,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    min_snr_db: float | None = None,
):
    """List recent detections."""
    api_manager = get_api_manager(request)
    detections = api_manager._detections

    if min_snr_db is not None:
        detections = [d for d in detections if d.get("snr_db", 0) >= min_snr_db]

    return {
        "total": len(detections),
        "offset": offset,
        "limit": limit,
        "detections": detections[offset : offset + limit],
    }


@router.get("/detections/{detection_id}")
async def get_detection(detection_id: int, request: Request):
    """Get details for a specific detection."""
    api_manager = get_api_manager(request)
    for det in api_manager._detections:
        if det.get("detection_id") == detection_id:
            return det
    raise HTTPException(status_code=404, detail="Detection not found")


@router.post("/detections/export")
async def export_detections(request: Request, format: str = "json"):
    """Export detections to file."""
    api_manager = get_api_manager(request)
    if format == "json":
        content = json.dumps(api_manager._detections, indent=2)
        return StreamingResponse(
            iter([content]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=detections.json"},
        )
    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


# Cluster endpoints
@router.get("/clusters")
async def list_clusters(request: Request):
    """List detected signal clusters."""
    api_manager = get_api_manager(request)
    return {"clusters": api_manager._clusters}


@router.get("/clusters/{cluster_id}")
async def get_cluster(cluster_id: int, request: Request):
    """Get details for a specific cluster."""
    api_manager = get_api_manager(request)
    for cluster in api_manager._clusters:
        if cluster.get("cluster_id") == cluster_id:
            return cluster
    raise HTTPException(status_code=404, detail="Cluster not found")


@router.post("/clusters/{cluster_id}/label")
async def label_cluster(cluster_id: int, label: str, request: Request):
    """Assign a label to a cluster."""
    api_manager = get_api_manager(request)
    for cluster in api_manager._clusters:
        if cluster.get("cluster_id") == cluster_id:
            cluster["label"] = label
            return {"status": "labeled", "cluster_id": cluster_id, "label": label}
    raise HTTPException(status_code=404, detail="Cluster not found")


# Demodulation endpoints
@router.get("/demod/protocols")
async def list_protocols():
    """List supported demodulation protocols."""
    return {
        "protocols": [
            {"name": "lora", "description": "LoRa Chirp Spread Spectrum"},
            {"name": "ble", "description": "Bluetooth Low Energy GFSK"},
            {"name": "bpsk", "description": "Binary Phase Shift Keying"},
            {"name": "qpsk", "description": "Quadrature PSK"},
            {"name": "16qam", "description": "16-QAM"},
        ]
    }


@router.post("/demod/analyze")
async def analyze_signal(start_sample: int, num_samples: int, protocol: str):
    """Analyze a signal segment with specified protocol."""
    return {"status": "analyzed", "protocol": protocol, "results": {}}
