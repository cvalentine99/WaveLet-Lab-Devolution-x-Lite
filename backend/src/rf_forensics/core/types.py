"""
Valentine RF HOS Platform - Core Type Definitions
Production-grade type system for RF signal processing
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any

import numpy as np

# ==============================================================================
# Sample Format Definitions
# ==============================================================================


class SampleFormat(Enum):
    """Supported I/Q sample formats"""

    CI16 = "ci16"  # Complex int16 (native SDR format)
    CI8 = "ci8"  # Complex int8 (RTL-SDR)
    CF32 = "cf32"  # Complex float32 (processing format)
    CF64 = "cf64"  # Complex float64 (high precision)
    CS12 = "cs12"  # Complex signed 12-bit (packed)
    CS16 = "cs16"  # Complex signed 16-bit


class ClockSource(Enum):
    """Clock reference sources"""

    INTERNAL = "internal"
    EXTERNAL_10MHZ = "external_10mhz"
    GPSDO = "gpsdo"
    PPS = "pps"
    IEEE1588 = "ieee1588"


class PPSEdge(Enum):
    """PPS edge trigger selection"""

    RISING = "rising"
    FALLING = "falling"


# ==============================================================================
# Binary Frame Structures (C-compatible packed structs)
# ==============================================================================


@dataclass
class IQFrameCI16:
    """
    Packed I/Q frame for ci16 format
    Total size: 12 bytes per sample

    C equivalent:
    typedef struct __attribute__((packed)) {
        uint64_t timestamp_ns;
        int16_t i;
        int16_t q;
    } iq_ci16_frame_t;
    """

    STRUCT_FORMAT = "<Qhh"  # Little-endian: uint64, int16, int16
    SIZE = 12

    timestamp_ns: int
    i: int
    q: int

    def pack(self) -> bytes:
        return struct.pack(self.STRUCT_FORMAT, self.timestamp_ns, self.i, self.q)

    @classmethod
    def unpack(cls, data: bytes) -> IQFrameCI16:
        timestamp_ns, i, q = struct.unpack(cls.STRUCT_FORMAT, data[: cls.SIZE])
        return cls(timestamp_ns=timestamp_ns, i=i, q=q)

    def to_complex(self, scale: float = 1.0 / 32768.0) -> complex:
        return complex(self.i * scale, self.q * scale)


@dataclass
class IQFrameCF32:
    """
    Packed I/Q frame for cf32 format
    Total size: 16 bytes per sample

    C equivalent:
    typedef struct __attribute__((packed)) {
        uint64_t timestamp_ns;
        float i;
        float q;
    } iq_cf32_frame_t;
    """

    STRUCT_FORMAT = "<Qff"  # Little-endian: uint64, float32, float32
    SIZE = 16

    timestamp_ns: int
    i: float
    q: float

    def pack(self) -> bytes:
        return struct.pack(self.STRUCT_FORMAT, self.timestamp_ns, self.i, self.q)

    @classmethod
    def unpack(cls, data: bytes) -> IQFrameCF32:
        timestamp_ns, i, q = struct.unpack(cls.STRUCT_FORMAT, data[: cls.SIZE])
        return cls(timestamp_ns=timestamp_ns, i=i, q=q)

    def to_complex(self) -> complex:
        return complex(self.i, self.q)


@dataclass
class IQBurstHeader:
    """
    Burst header for streaming I/Q data blocks

    C equivalent:
    typedef struct __attribute__((packed)) {
        uint32_t magic;           // 0x56414C52 ('VALR')
        uint32_t version;         // Protocol version
        uint64_t sequence_num;    // Monotonic sequence counter
        uint64_t timestamp_ns;    // First sample timestamp
        uint32_t sample_count;    // Number of samples in burst
        uint32_t sample_format;   // SampleFormat enum
        uint64_t center_freq_hz;  // Center frequency
        uint32_t sample_rate_hz;  // Sample rate
        uint32_t flags;           // Status flags
        uint8_t  reserved[16];    // Future use
    } iq_burst_header_t;
    """

    MAGIC = 0x56414C52  # 'VALR'
    STRUCT_FORMAT = "<IIQQIIQI16s"
    SIZE = 64

    version: int
    sequence_num: int
    timestamp_ns: int
    sample_count: int
    sample_format: SampleFormat
    center_freq_hz: int
    sample_rate_hz: int
    flags: int

    # Flag definitions
    FLAG_OVERFLOW = 0x0001
    FLAG_UNDERFLOW = 0x0002
    FLAG_PPS_LOCKED = 0x0004
    FLAG_GPSDO_LOCKED = 0x0008
    FLAG_TIME_VALID = 0x0010
    FLAG_ADC_OVERLOAD = 0x0020

    def pack(self) -> bytes:
        return struct.pack(
            self.STRUCT_FORMAT,
            self.MAGIC,
            self.version,
            self.sequence_num,
            self.timestamp_ns,
            self.sample_count,
            self.sample_format.value
            if isinstance(self.sample_format, SampleFormat)
            else self.sample_format,
            self.center_freq_hz,
            self.sample_rate_hz,
            self.flags,
            b"\x00" * 16,
        )

    @classmethod
    def unpack(cls, data: bytes) -> IQBurstHeader:
        (magic, version, seq, ts, count, fmt, freq, rate, flags, _) = struct.unpack(
            cls.STRUCT_FORMAT, data[: cls.SIZE]
        )

        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}, expected 0x{cls.MAGIC:08X}")

        return cls(
            version=version,
            sequence_num=seq,
            timestamp_ns=ts,
            sample_count=count,
            sample_format=fmt,
            center_freq_hz=freq,
            sample_rate_hz=rate,
            flags=flags,
        )

    @property
    def is_overflow(self) -> bool:
        return bool(self.flags & self.FLAG_OVERFLOW)

    @property
    def is_pps_locked(self) -> bool:
        return bool(self.flags & self.FLAG_PPS_LOCKED)

    @property
    def is_gpsdo_locked(self) -> bool:
        return bool(self.flags & self.FLAG_GPSDO_LOCKED)


# ==============================================================================
# HOS (Higher-Order Statistics) Types
# ==============================================================================


@dataclass
class CumulantResult:
    """
    Result container for cumulant calculations
    Contains C20, C21, C40, C41, C42, C60, C61, C62, C63
    """

    # Second-order cumulants
    c20: complex = 0j  # E[x²]
    c21: complex = 0j  # E[|x|²]

    # Fourth-order cumulants
    c40: complex = 0j  # cum(x,x,x,x) = E[x⁴] - 3E[x²]²
    c41: complex = 0j  # cum(x,x,x,x*) = E[x³x*] - 3E[x²]E[xx*]
    c42: complex = 0j  # cum(x,x,x*,x*) = E[|x|⁴] - |E[x²]|² - 2E[|x|²]²

    # Sixth-order cumulants
    c60: complex = 0j  # E[x⁶] - 15E[x⁴]E[x²] + 30E[x²]³
    c61: complex = 0j
    c62: complex = 0j
    c63: complex = 0j

    # Normalized versions (invariant to signal power)
    c40_norm: complex = 0j
    c42_norm: complex = 0j
    c60_norm: complex = 0j

    # Statistics
    kurtosis: float = 0.0
    skewness: complex = 0j

    # Metadata
    window_size: int = 0
    overlap: int = 0
    timestamp_ns: int = 0

    def to_feature_vector(self) -> np.ndarray:
        """Convert to ML feature vector (real representation)"""
        return np.array(
            [
                np.real(self.c40_norm),
                np.imag(self.c40_norm),
                np.real(self.c42_norm),
                np.imag(self.c42_norm),
                np.real(self.c60_norm),
                np.imag(self.c60_norm),
                self.kurtosis,
                np.real(self.skewness),
                np.imag(self.skewness),
            ],
            dtype=np.float32,
        )


@dataclass
class SCDResult:
    """
    Spectral Correlation Density (SCD) computation result
    Contains the cyclic spectral density function S_x^α(f)
    """

    # SCD tensor: shape (num_alpha, num_freq)
    scd_magnitude: np.ndarray = field(default_factory=lambda: np.array([]))
    scd_phase: np.ndarray = field(default_factory=lambda: np.array([]))

    # Cyclic frequencies (alpha axis)
    alpha_axis: np.ndarray = field(default_factory=lambda: np.array([]))

    # Spectral frequencies (f axis)
    freq_axis: np.ndarray = field(default_factory=lambda: np.array([]))

    # Peak detection results
    peak_alphas: list[float] = field(default_factory=list)
    peak_freqs: list[float] = field(default_factory=list)
    peak_magnitudes: list[float] = field(default_factory=list)

    # Cyclic features
    max_cyclic_freq: float = 0.0
    spectral_coherence: float = 0.0
    cyclic_power_ratio: float = 0.0

    # Metadata
    fft_size: int = 0
    num_segments: int = 0
    sample_rate: float = 0.0
    timestamp_ns: int = 0

    def get_slice_at_alpha(self, alpha: float) -> np.ndarray:
        """Extract SCD slice at specific cyclic frequency"""
        idx = np.argmin(np.abs(self.alpha_axis - alpha))
        return self.scd_magnitude[idx, :]

    def to_feature_vector(self) -> np.ndarray:
        """Extract summary features for ML"""
        return np.array(
            [
                self.max_cyclic_freq,
                self.spectral_coherence,
                self.cyclic_power_ratio,
                np.max(self.peak_magnitudes) if self.peak_magnitudes else 0.0,
                len(self.peak_alphas),
            ],
            dtype=np.float32,
        )


# ==============================================================================
# ML Inference Types
# ==============================================================================


class ModulationType(IntEnum):
    """RF modulation type classification"""

    UNKNOWN = 0
    BPSK = 1
    QPSK = 2
    PSK8 = 3
    PSK16 = 4
    QAM16 = 5
    QAM64 = 6
    QAM256 = 7
    FSK2 = 8
    FSK4 = 9
    MSK = 10
    GMSK = 11
    OFDM = 12
    AM = 13
    FM = 14
    SSB = 15
    NOISE = 16
    CHIRP = 17
    FHSS = 18
    DSSS = 19


@dataclass
class InferenceResult:
    """ML model inference result"""

    # Primary classification
    modulation_type: ModulationType = ModulationType.UNKNOWN
    confidence: float = 0.0

    # Full probability distribution
    class_probabilities: dict[ModulationType, float] = field(default_factory=dict)

    # Latent space embedding
    latent_vector: np.ndarray = field(default_factory=lambda: np.array([]))

    # Anomaly detection
    anomaly_score: float = 0.0
    mahalanobis_distance: float = 0.0
    is_out_of_distribution: bool = False

    # Model metadata
    model_name: str = ""
    model_version: str = ""
    model_hash: str = ""
    inference_time_ms: float = 0.0

    # Batch info
    batch_size: int = 0
    frame_indices: list[int] = field(default_factory=list)
    timestamp_ns: int = 0


# ==============================================================================
# Fusion Engine Types
# ==============================================================================


@dataclass
class FusionWeights:
    """Configurable weights for decision fusion"""

    hos_weight: float = 0.35
    scd_weight: float = 0.30
    ml_weight: float = 0.35

    # Confidence thresholds
    min_confidence: float = 0.6
    anomaly_threshold: float = 2.5  # Mahalanobis distance

    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = self.hos_weight + self.scd_weight + self.ml_weight
        self.hos_weight /= total
        self.scd_weight /= total
        self.ml_weight /= total


@dataclass
class FusionResult:
    """Combined decision from all analysis pipelines"""

    # Final classification
    classification: ModulationType = ModulationType.UNKNOWN
    fused_confidence: float = 0.0

    # Individual component scores
    hos_confidence: float = 0.0
    scd_coherence: float = 0.0
    ml_probability: float = 0.0

    # Anomaly assessment
    anomaly_score: float = 0.0
    is_anomalous: bool = False
    anomaly_reasons: list[str] = field(default_factory=list)

    # Signatures for forensic record
    hos_signature: CumulantResult = field(default_factory=CumulantResult)
    scd_signature: SCDResult = field(default_factory=SCDResult)
    latent_embedding: np.ndarray = field(default_factory=lambda: np.array([]))

    # Fusion metadata
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    processing_time_ms: float = 0.0
    timestamp_ns: int = 0

    # Evidence tracking
    capture_id: str = ""
    sequence_number: int = 0


# ==============================================================================
# Forensic Types
# ==============================================================================


@dataclass
class EvidenceMetadata:
    """Forensic evidence metadata for chain of custody"""

    evidence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = ""
    capture_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Device identification
    device_serial: str = ""
    device_model: str = ""
    firmware_version: str = ""

    # Capture parameters
    center_freq_hz: int = 0
    sample_rate_hz: int = 0
    bandwidth_hz: int = 0
    gain_db: float = 0.0
    antenna: str = ""

    # Location (if available)
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    gps_fix_quality: int = 0

    # Integrity
    sha3_256_hash: str = ""
    sha256_hash: str = ""
    ed25519_signature: str = ""
    signer_public_key: str = ""

    # Chain of custody
    collected_by: str = ""
    collection_notes: str = ""
    custody_transfers: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BagItManifest:
    """BagIt bag manifest structure"""

    bag_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    # File checksums
    payload_files: dict[str, str] = field(default_factory=dict)  # path -> SHA3-256
    tag_files: dict[str, str] = field(default_factory=dict)

    # Bag metadata
    bag_size_bytes: int = 0
    payload_oxum: str = ""  # "octetcount.streamcount"

    # Validation
    is_valid: bool = False
    validation_errors: list[str] = field(default_factory=list)


# ==============================================================================
# Telemetry Types
# ==============================================================================


@dataclass
class SDRTelemetry:
    """Real-time SDR health and status telemetry"""

    timestamp_ns: int = 0
    device_id: str = ""

    # Clock status
    pps_locked: bool = False
    gpsdo_locked: bool = False
    ref_locked: bool = False
    pll_locked: bool = False

    # Frequency accuracy
    lo_offset_hz: float = 0.0
    ppm_error: float = 0.0
    drift_rate_hz_per_sec: float = 0.0

    # Temperature
    fpga_temp_c: float = 0.0
    board_temp_c: float = 0.0

    # Gain/power
    rx_gain_db: float = 0.0
    agc_enabled: bool = False
    adc_overload_count: int = 0
    rssi_dbm: float = -100.0

    # Buffer status
    buffer_fill_percent: float = 0.0
    overflow_count: int = 0
    underflow_count: int = 0
    dropped_samples: int = 0

    # GPS
    gps_latitude: float = 0.0
    gps_longitude: float = 0.0
    gps_satellites: int = 0
    gps_fix_type: int = 0  # 0=none, 1=2D, 2=3D


# ==============================================================================
# Pipeline Types
# ==============================================================================


@dataclass
class ProcessingContext:
    """Context passed through the processing pipeline"""

    capture_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = ""

    # Timing
    start_time_ns: int = 0
    end_time_ns: int = 0

    # Sample info
    sample_format: SampleFormat = SampleFormat.CF32
    sample_rate_hz: int = 0
    center_freq_hz: int = 0

    # Processing stages completed
    ingested: bool = False
    hos_computed: bool = False
    scd_computed: bool = False
    ml_inferred: bool = False
    fused: bool = False
    sealed: bool = False

    # Results
    hos_result: CumulantResult | None = None
    scd_result: SCDResult | None = None
    inference_result: InferenceResult | None = None
    fusion_result: FusionResult | None = None

    # Telemetry snapshot
    telemetry: SDRTelemetry | None = None

    # Evidence
    evidence_metadata: EvidenceMetadata | None = None
    bag_manifest: BagItManifest | None = None
