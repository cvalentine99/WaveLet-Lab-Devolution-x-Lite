"""
API Routers

FastAPI routers for modular endpoint organization.
"""

from rf_forensics.api.routers.analysis import router as analysis_router
from rf_forensics.api.routers.config import router as config_router
from rf_forensics.api.routers.detections import router as detections_router
from rf_forensics.api.routers.recordings import router as recordings_router
from rf_forensics.api.routers.sdr import router as sdr_router
from rf_forensics.api.routers.system import router as system_router

__all__ = [
    "system_router",
    "config_router",
    "detections_router",
    "recordings_router",
    "sdr_router",
    "analysis_router",
]
