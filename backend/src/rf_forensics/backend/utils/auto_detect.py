import os

import numpy as np


def looks_like_u8(path: str) -> bool:
    """Check if file looks like unsigned 8-bit IQ data."""
    data = np.fromfile(path, dtype=np.uint8, count=4096)
    return bool(np.all(data >= 0) and np.all(data <= 255))


def looks_like_ci16(path: str) -> bool:
    """Check if file looks like complex int16 IQ data."""
    try:
        _ = np.fromfile(path, dtype=np.int16, count=4096)
        return True
    except Exception:
        return False


def auto_detect_format(path: str) -> str:
    """Auto-detect IQ file format based on file size and content heuristics."""
    size = os.path.getsize(path)

    if looks_like_u8(path):
        return "u8"
    if size % 4 == 0 and looks_like_ci16(path):
        return "ci16"
    if size % 8 == 0:
        return "cf32"

    raise ValueError("Unknown IQ format; unable to auto-detect")
