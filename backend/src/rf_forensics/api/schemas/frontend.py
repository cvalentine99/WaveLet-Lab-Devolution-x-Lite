"""
Frontend Schema Models

Pydantic models for frontend REST API (camelCase format).
"""

from pydantic import BaseModel, Field


class FrontendSDRConfig(BaseModel):
    center_freq_hz: float = 915e6
    sample_rate_hz: float = 10e6
    gain_db: float = 40.0


class FrontendPipelineConfig(BaseModel):
    fft_size: int = 1024
    window_type: str = "hann"
    overlap: float = 0.5  # 0.0 to 1.0


class FrontendCFARConfig(BaseModel):
    pfa: float = 1e-6
    ref_cells: int = 32
    guard_cells: int = 4


class FrontendClusteringConfig(BaseModel):
    enabled: bool = True
    eps: float = 0.5
    min_samples: int = 5


class FrontendConfig(BaseModel):
    sdr: FrontendSDRConfig = Field(default_factory=FrontendSDRConfig)
    pipeline: FrontendPipelineConfig = Field(default_factory=FrontendPipelineConfig)
    cfar: FrontendCFARConfig = Field(default_factory=FrontendCFARConfig)
    clustering: FrontendClusteringConfig = Field(default_factory=FrontendClusteringConfig)


class FrontendSDRConfigRequest(BaseModel):
    """Frontend SDR config request (camelCase)."""

    centerFreqHz: int = 915000000
    sampleRateHz: int = 2048000
    gainDb: float = 40
    bandwidth: int = 2000000


class FrontendMonitoringStartRequest(BaseModel):
    """Frontend monitoring start request."""

    mode: str = "both"  # "spectrum", "detection", or "both"
    fftSize: int = 2048
    detectionThreshold: float = -80


class FrontendMonitoringStopRequest(BaseModel):
    """Frontend monitoring stop request."""

    sessionId: str = ""


class FrontendRecordingStartRequest(BaseModel):
    """Frontend recording start request."""

    filename: str = "capture.iq"
    format: str = "complex_float32"  # or "complex_int16"
    duration: int = 0  # 0 = unlimited


class FrontendRecordingStopRequest(BaseModel):
    """Frontend recording stop request."""

    recordingId: str
