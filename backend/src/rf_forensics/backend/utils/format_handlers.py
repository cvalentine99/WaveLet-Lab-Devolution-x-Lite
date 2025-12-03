from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Registry of format decoders
FORMAT_HANDLERS: dict[str, Callable[[str, Any], NDArray[np.complexfloating]]] = {}


def register_format(fmt: str) -> Callable[[Callable], Callable]:
    """Decorator to register a decoder for a given format key."""

    def decorator(func: Callable) -> Callable:
        FORMAT_HANDLERS[fmt] = func
        return func

    return decorator


@register_format("ci16")
def decode_ci16(path: str, desc: Any) -> NDArray[np.complexfloating]:
    """Decode complex int16 IQ data."""
    dtype = "<i2" if desc.endianness == "little" else ">i2"
    raw = np.fromfile(path, dtype=dtype)
    raw = raw.astype(np.float32) / desc.scale_factor
    iq = raw.reshape(-1, 2)
    return iq[:, 0] + 1j * iq[:, 1]


@register_format("cf32")
def decode_cf32(path: str, desc: Any) -> NDArray[np.complexfloating]:
    """Decode complex float32 IQ data."""
    dtype = "<f4" if desc.endianness == "little" else ">f4"
    raw = np.fromfile(path, dtype=dtype)
    iq = raw.view(np.complex64)
    return iq


@register_format("u8")
def decode_u8(path: str, desc: Any) -> NDArray[np.complexfloating]:
    """Decode unsigned 8-bit IQ data (RTL-SDR format)."""
    raw = np.fromfile(path, dtype=np.uint8)
    raw = raw.astype(np.float32) - desc.iq_offset
    raw /= desc.scale_factor
    iq = raw.reshape(-1, 2)
    return iq[:, 0] + 1j * iq[:, 1]


@register_format("wav_iq")
def decode_wav_iq(path: str, desc: Any) -> NDArray[np.complexfloating]:
    """Decode WAV stereo IQ data."""
    import soundfile as sf

    raw, sr = sf.read(path)
    if sr != desc.sample_rate_hz:
        raise ValueError("Sample rate mismatch between WAV header and descriptor")
    # Assume stereo IQ
    return raw[:, 0] + 1j * raw[:, 1]
