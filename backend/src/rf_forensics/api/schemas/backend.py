"""
Backend Schema Models

Pydantic models for backend REST API (snake_case format).
"""

from pydantic import BaseModel, Field


class SDRConfigModel(BaseModel):
    device_type: str = "usdr"
    center_freq_hz: float = Field(default=915e6, ge=1e6, le=6e9)
    sample_rate_hz: float = Field(default=10e6, ge=1e3, le=61.44e6)
    bandwidth_hz: float = Field(default=10e6, ge=1e3, le=56e6)
    gain_db: float = Field(default=40.0, ge=0, le=76)


class PipelineConfigModel(BaseModel):
    fft_size: int = Field(default=1024, ge=64, le=65536)
    window_type: str = "hann"
    overlap_percent: float = Field(default=50.0, ge=0, le=90)
    averaging_count: int = Field(default=16, ge=1, le=1000)


class CFARConfigModel(BaseModel):
    num_reference_cells: int = Field(default=32, ge=4, le=256)
    num_guard_cells: int = Field(default=4, ge=1, le=32)
    pfa: float = Field(default=1e-6, ge=1e-12, le=0.1)
    variant: str = "CA"


class ConfigUpdateModel(BaseModel):
    sdr: SDRConfigModel | None = None
    pipeline: PipelineConfigModel | None = None
    cfar: CFARConfigModel | None = None


class DetectionModel(BaseModel):
    detection_id: int
    center_freq_hz: float
    bandwidth_hz: float
    peak_power_db: float
    snr_db: float
    timestamp: float


class ClusterModel(BaseModel):
    cluster_id: int
    size: int
    dominant_frequency_hz: float
    avg_snr_db: float
    label: str = ""


class StatusModel(BaseModel):
    state: str  # "idle" | "configuring" | "running" | "paused" | "error"
    uptime_seconds: float
    samples_processed: int
    detections_count: int
    current_throughput_msps: float
    gpu_memory_used_gb: float
    buffer_fill_level: float  # 0.0 to 1.0
    processing_latency_ms: float
    # Additional fields per BACKEND_CONTRACT.md
    consecutive_errors: int = 0
    dropped_samples: int = 0
    sdr_throttled: bool = False
    gpu_utilization_percent: float = 0.0


class SuccessResponse(BaseModel):
    success: bool


# Recording models (SigMF)
class StartRecordingRequest(BaseModel):
    name: str = "Recording"
    description: str = ""
    duration_seconds: float | None = None


class StartRecordingResponse(BaseModel):
    recording_id: str
    status: str


class StopRecordingRequest(BaseModel):
    recording_id: str


class StopRecordingResponse(BaseModel):
    recording_id: str
    status: str
    file_size_bytes: int
    num_samples: int


class RecordingInfo(BaseModel):
    id: str
    name: str
    description: str
    center_freq_hz: float
    sample_rate_hz: float
    num_samples: int
    duration_seconds: float
    file_size_bytes: int
    created_at: str
    status: str
    sigmf_meta_path: str
    sigmf_data_path: str


# SDR Device Management Models (uSDR DevBoard)
class SDRDeviceModel(BaseModel):
    id: str
    model: str = "uSDR DevBoard"
    serial: str = ""
    status: str = "available"  # available, connected, busy, error


class SDRDeviceListResponse(BaseModel):
    devices: list[SDRDeviceModel]


class SDRConnectRequest(BaseModel):
    device_id: str = "usb://0"


class SDRConnectResponse(BaseModel):
    success: bool
    device_id: str = ""


class SDRGainModel(BaseModel):
    lna_db: int = 15  # 0-30
    tia_db: int = 9  # 0, 3, 9, 12
    pga_db: int = 12  # 0-32
    total_db: int = 36


class SDRDevBoardModel(BaseModel):
    """DevBoard settings (see docs.wsdr.io/hardware/devboard.html)."""

    # Core amplifier controls
    lna_enable: bool = True  # lna_: RX LNA +19.5dB (QPL9547TR7)
    pa_enable: bool = False  # pa_: TX PA +19.5dB (QPL9547TR7)
    attenuator_db: int = 0  # attn_: 0-18 dB RX attenuation
    vctcxo_dac: int = 32768  # dac_: VCTCXO tuning 0-65535 (default=32768, ±275Hz)
    # Additional DevBoard peripherals
    gps_enable: bool = False  # gps_: GPS module on/off
    osc_enable: bool = True  # osc_: Reference clock oscillator on/off
    loopback_enable: bool = False  # lb_: RX->TX loopback on/off
    uart_enable: bool = False  # uart_: UART interface on/off


class SDRFullConfigModel(BaseModel):
    device_type: str = "usdr"
    center_freq_hz: int = 915000000
    sample_rate_hz: int = 10000000
    bandwidth_hz: int = 10000000
    gain: SDRGainModel = Field(default_factory=SDRGainModel)
    rx_path: str = "LNAL"  # LNAH, LNAL, LNAW
    devboard: SDRDevBoardModel | None = None


class SDRStatusModel(BaseModel):
    connected: bool = False
    device_id: str | None = None
    temperature_c: float = 0.0
    actual_freq_hz: int = 0
    actual_sample_rate_hz: int = 0
    actual_bandwidth_hz: int = 0
    rx_path: str = ""
    streaming: bool = False


# Duplexer Band Models
class DuplexerBandModel(BaseModel):
    name: str
    aliases: list[str] = []
    category: str  # cellular, tx_only, rx_only, tdd
    freq_range_mhz: list[int]  # [min, max]
    description: str
    pa_enable: bool
    lna_enable: bool
    trx_mode: str
    rx_filter: str
    tx_filter: str


class DuplexerBandsResponse(BaseModel):
    bands: list[DuplexerBandModel]
    categories: list[str] = ["cellular", "tx_only", "rx_only", "tdd"]


class SetBandRequest(BaseModel):
    band: str  # band name or alias
