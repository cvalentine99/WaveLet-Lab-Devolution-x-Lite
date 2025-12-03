"""
GPU Compatibility Module

Provides unified access to GPU array operations (CuPy) with automatic
fallback to NumPy when CUDA is not available.

Usage:
    from rf_forensics.core.gpu_compat import cp, np, CUPY_AVAILABLE, get_array_module

    # Use cp for GPU arrays (falls back to np if no GPU)
    arr = cp.zeros(1024, dtype=cp.complex64)

    # Check if GPU is available
    if CUPY_AVAILABLE:
        # GPU-specific code path
        ...

    # Get the appropriate module for an existing array
    xp = get_array_module(arr)
    result = xp.fft.fft(arr)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np  # type: ignore[misc]


def get_array_module(arr=None):
    """
    Get the appropriate array module (cupy or numpy).

    Args:
        arr: Optional array to check. If provided and it's a CuPy array,
             returns cupy. Otherwise returns numpy.

    Returns:
        cupy module if available and arr is a GPU array, else numpy.
    """
    if arr is not None and CUPY_AVAILABLE:
        # Check if it's a CuPy array
        if hasattr(arr, "device") or type(arr).__module__.startswith("cupy"):
            return cp
    return np


def to_numpy(arr):
    """
    Convert array to numpy, handling both numpy and cupy arrays.

    Args:
        arr: Input array (numpy or cupy).

    Returns:
        numpy array.
    """
    if CUPY_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def to_gpu(arr):
    """
    Convert array to GPU (cupy) if available, otherwise return as-is.

    Args:
        arr: Input array.

    Returns:
        CuPy array if GPU available, else numpy array.
    """
    if CUPY_AVAILABLE:
        return cp.asarray(arr)
    return np.asarray(arr)


def free_gpu_memory():
    """Free all GPU memory pools."""
    if CUPY_AVAILABLE:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception as e:
            logger.debug(f"Error freeing GPU memory pools: {e}")


def get_gpu_memory_info():
    """
    Get GPU memory usage information.

    Returns:
        Dict with 'free_gb', 'total_gb', 'used_gb' or None if no GPU.
    """
    if not CUPY_AVAILABLE:
        return None

    try:
        free, total = cp.cuda.runtime.memGetInfo()
        return {
            "free_gb": free / (1024**3),
            "total_gb": total / (1024**3),
            "used_gb": (total - free) / (1024**3),
        }
    except Exception:
        return None
