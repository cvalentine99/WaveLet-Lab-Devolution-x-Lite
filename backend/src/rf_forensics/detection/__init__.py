"""
GPU RF Forensics Engine - Detection Module

CFAR detection and peak finding algorithms.
"""

from rf_forensics.detection.cfar import CFARDetector, CFARResult
from rf_forensics.detection.peaks import Detection, PeakDetector

__all__ = [
    "CFARDetector",
    "CFARResult",
    "PeakDetector",
    "Detection",
]
