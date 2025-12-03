"""
GPU RF Forensics Engine - Configuration Schema

Pydantic models for all configuration options with validation rules.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SDRConfig(BaseModel):
    """SDR device configuration."""

    device_type: Literal["usrp", "hackrf", "limesdr", "rtlsdr", "usdr"] = "usdr"
    device_args: str = ""
    center_freq_hz: float = Field(default=915e6, ge=1e6, le=6e9)
    sample_rate_hz: float = Field(default=10e6, ge=1e3, le=61.44e6)
    bandwidth_hz: float = Field(default=10e6, ge=1e3, le=56e6)
    gain_db: float = Field(default=40.0, ge=0, le=76)
    antenna: str = "TX/RX"

    @field_validator("bandwidth_hz")
    @classmethod
    def bandwidth_lte_sample_rate(cls, v, info):
        if "sample_rate_hz" in info.data and v > info.data["sample_rate_hz"]:
            raise ValueError("bandwidth_hz must be <= sample_rate_hz")
        return v


class BufferConfig(BaseModel):
    """Memory buffer configuration.

    Buffer sizing for optimal throughput:
    - samples_per_buffer: 500K samples = 50ms at 10 MSPS
    - num_ring_segments: 8 segments = 400ms buffer depth
    - This allows processing (9.4ms) << acquisition (50ms) to prevent backpressure
    """

    samples_per_buffer: int = Field(default=500_000, ge=1024, le=100_000_000)
    num_ring_segments: int = Field(default=8, ge=2, le=32)
    use_pinned_memory: bool = True


class FFTConfig(BaseModel):
    """FFT and spectral analysis configuration."""

    fft_size: int = Field(default=1024, ge=64, le=65536)
    window_type: Literal["hann", "hamming", "blackman", "kaiser", "flattop"] = "hann"
    overlap_percent: float = Field(default=50.0, ge=0, le=90)
    averaging_count: int = Field(default=16, ge=1, le=1000)

    @field_validator("fft_size")
    @classmethod
    def fft_size_power_of_two(cls, v):
        if v & (v - 1) != 0:
            raise ValueError("fft_size must be a power of 2")
        return v


class CFARConfig(BaseModel):
    """CFAR detector configuration."""

    num_reference_cells: int = Field(default=32, ge=4, le=256)
    num_guard_cells: int = Field(default=4, ge=1, le=32)
    pfa: float = Field(default=1e-6, ge=1e-12, le=0.1)
    variant: Literal["CA", "GO", "SO", "OS"] = "CA"


class ClusteringConfig(BaseModel):
    """DBSCAN clustering configuration."""

    eps: float = Field(default=0.5, ge=0.01, le=10.0)
    min_samples: int = Field(default=5, ge=1, le=100)
    auto_tune: bool = True


class WebSocketConfig(BaseModel):
    """WebSocket server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1024, le=65535)
    spectrum_update_rate_hz: float = Field(default=30.0, ge=1, le=120)
    max_clients: int = Field(default=10, ge=1, le=100)


class APIConfig(BaseModel):
    """REST API configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    enable_cors: bool = True
    cors_origins: list[str] = ["*"]


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str | None = None
    log_to_console: bool = True
    log_performance_metrics: bool = True


class PipelineConfig(BaseModel):
    """Pipeline processing configuration."""

    # Callback limits (for non-blocking callback dispatch)
    max_pending_callbacks: int = Field(default=100, ge=10, le=1000)

    # Detection storage limits
    max_recent_detections: int = Field(default=1000, ge=100, le=100000)

    # Processing timeouts
    segment_timeout_seconds: float = Field(default=0.1, ge=0.01, le=5.0)


class RFForensicsConfig(BaseModel):
    """Root configuration for RF Forensics Engine."""

    sdr: SDRConfig = Field(default_factory=SDRConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    fft: FFTConfig = Field(default_factory=FFTConfig)
    cfar: CFARConfig = Field(default_factory=CFARConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "RFForensicsConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Preset configurations
PRESET_WIDEBAND_SURVEY = RFForensicsConfig(
    sdr=SDRConfig(sample_rate_hz=20e6, bandwidth_hz=20e6),
    fft=FFTConfig(fft_size=2048, averaging_count=8),
    cfar=CFARConfig(pfa=1e-5),
)

PRESET_NARROWBAND_ANALYSIS = RFForensicsConfig(
    sdr=SDRConfig(sample_rate_hz=1e6, bandwidth_hz=1e6),
    fft=FFTConfig(fft_size=8192, averaging_count=64),
    cfar=CFARConfig(pfa=1e-7, num_reference_cells=64),
)

PRESET_BURST_DETECTION = RFForensicsConfig(
    sdr=SDRConfig(sample_rate_hz=10e6, bandwidth_hz=10e6),
    fft=FFTConfig(fft_size=512, averaging_count=4),
    cfar=CFARConfig(pfa=1e-4, num_guard_cells=8),
)

PRESET_ISM_BAND_915 = RFForensicsConfig(
    sdr=SDRConfig(center_freq_hz=915e6, sample_rate_hz=26e6, bandwidth_hz=26e6),
    fft=FFTConfig(fft_size=4096, averaging_count=16),
)

PRESETS = {
    "wideband_survey": PRESET_WIDEBAND_SURVEY,
    "narrowband_analysis": PRESET_NARROWBAND_ANALYSIS,
    "burst_detection": PRESET_BURST_DETECTION,
    "ism_band_915": PRESET_ISM_BAND_915,
}
