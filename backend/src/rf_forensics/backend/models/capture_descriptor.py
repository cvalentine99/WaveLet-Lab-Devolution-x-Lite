from typing import Literal

from pydantic import BaseModel, Field


class CaptureDescriptor(BaseModel):
    """
    Metadata describing an IQ capture and how to decode it.
    """

    format: Literal[
        "ci8",
        "ci16",
        "ci32",
        "cf32",
        "cf64",
        "u8",
        "u16",
        "wav_iq",
        "complex64",
        "gr_complex64",
        "hackrf_raw",
        "rtlsdr_u8",
    ] = Field(..., description="IQ sample encoding format")

    sample_rate_hz: int = Field(..., gt=0)
    center_frequency_hz: int | None = Field(None, gt=0)

    endianness: Literal["little", "big"] | None = "little"
    interleaving: Literal["IQIQ", "IIQQ"] | None = "IQIQ"
    scale_factor: float | None = 1.0
    num_channels: int | None = 1
    iq_offset: float | None = 0.0

    metadata_version: int | None = 1
