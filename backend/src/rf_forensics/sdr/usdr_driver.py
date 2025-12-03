"""
GPU RF Forensics Engine - USDR Driver

Python wrapper for libusdr.so to control uSDR DevBoard hardware.
Uses ctypes to interface with the native C library.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Library path (configurable via environment)
USDR_LIB_PATH = os.environ.get("SDR_LIB_PATH", "/usr/local/lib/libusdr.so.0.9.9")


@dataclass
class USDRDevice:
    """Discovered USDR device info."""

    id: str
    model: str = "uSDR DevBoard"
    serial: str = ""
    status: str = "available"


@dataclass
class USDRGain:
    """RX gain configuration (3 stages)."""

    lna_db: int = 15  # 0-30 dB
    tia_db: int = 9  # 0, 3, 9, 12 dB
    pga_db: int = 12  # 0-32 dB

    @property
    def total_db(self) -> int:
        return self.lna_db + self.tia_db + self.pga_db


@dataclass
class DuplexerBand:
    """Duplexer band configuration."""

    name: str
    aliases: list[str]
    trx_mode: str  # TRX_BAND2-8 or TRX_BYPASS
    rx_filter: str  # RX_LPF1200, RX_LPF2100, RX_BPF2100_3000, RX_BPF3000_4200
    tx_filter: str  # TX_LPF400, TX_LPF1200, TX_LPF2100, TX_BYPASS
    pa_enable: bool
    lna_enable: bool
    category: str  # "cellular", "tx_only", "rx_only", "tdd"
    freq_range_mhz: tuple = (0, 0)  # (min, max) MHz
    description: str = ""


# Predefined duplexer bands for uSDR DevBoard
DUPLEXER_BANDS: dict[str, DuplexerBand] = {
    # Cellular FDD bands (duplexer active)
    "band2": DuplexerBand(
        name="band2",
        aliases=["pcs", "gsm1900"],
        trx_mode="TRX_BAND2",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF400",
        pa_enable=True,
        lna_enable=True,
        category="cellular",
        freq_range_mhz=(1850, 1990),
        description="PCS / GSM 1900",
    ),
    "band3": DuplexerBand(
        name="band3",
        aliases=["dcs", "gsm1800"],
        trx_mode="TRX_BAND3",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF400",
        pa_enable=True,
        lna_enable=True,
        category="cellular",
        freq_range_mhz=(1710, 1880),
        description="DCS / GSM 1800",
    ),
    "band5": DuplexerBand(
        name="band5",
        aliases=["gsm850"],
        trx_mode="TRX_BAND5",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF400",
        pa_enable=True,
        lna_enable=True,
        category="cellular",
        freq_range_mhz=(824, 894),
        description="GSM 850",
    ),
    "band7": DuplexerBand(
        name="band7",
        aliases=["imte"],
        trx_mode="TRX_BAND7",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF400",
        pa_enable=True,
        lna_enable=True,
        category="cellular",
        freq_range_mhz=(2500, 2690),
        description="IMT-E / LTE Band 7",
    ),
    "band8": DuplexerBand(
        name="band8",
        aliases=["gsm900"],
        trx_mode="TRX_BAND8",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF400",
        pa_enable=True,
        lna_enable=True,
        category="cellular",
        freq_range_mhz=(880, 960),
        description="GSM 900",
    ),
    # TX-only paths (duplexer bypass)
    "txlpf400": DuplexerBand(
        name="txlpf400",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF400",
        pa_enable=True,
        lna_enable=False,
        category="tx_only",
        freq_range_mhz=(0, 400),
        description="TX only, LPF 400 MHz",
    ),
    "txlpf1200": DuplexerBand(
        name="txlpf1200",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF1200",
        pa_enable=True,
        lna_enable=False,
        category="tx_only",
        freq_range_mhz=(0, 1200),
        description="TX only, LPF 1200 MHz",
    ),
    "txlpf2100": DuplexerBand(
        name="txlpf2100",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF2100",
        pa_enable=True,
        lna_enable=False,
        category="tx_only",
        freq_range_mhz=(0, 2100),
        description="TX only, LPF 2100 MHz",
    ),
    "txlpf4200": DuplexerBand(
        name="txlpf4200",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF1200",
        tx_filter="TX_BYPASS",
        pa_enable=True,
        lna_enable=False,
        category="tx_only",
        freq_range_mhz=(0, 4200),
        description="TX only, bypass (full range)",
    ),
    # RX-only paths (duplexer bypass)
    "rxlpf1200": DuplexerBand(
        name="rxlpf1200",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF400",
        pa_enable=False,
        lna_enable=True,
        category="rx_only",
        freq_range_mhz=(0, 1200),
        description="RX only, LPF 1200 MHz",
    ),
    "rxlpf2100": DuplexerBand(
        name="rxlpf2100",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF2100",
        tx_filter="TX_LPF400",
        pa_enable=False,
        lna_enable=True,
        category="rx_only",
        freq_range_mhz=(0, 2100),
        description="RX only, LPF 2100 MHz",
    ),
    "rxbpf2100_3000": DuplexerBand(
        name="rxbpf2100_3000",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_BPF2100_3000",
        tx_filter="TX_LPF400",
        pa_enable=False,
        lna_enable=True,
        category="rx_only",
        freq_range_mhz=(2100, 3000),
        description="RX only, BPF 2.1-3.0 GHz",
    ),
    "rxbpf3000_4200": DuplexerBand(
        name="rxbpf3000_4200",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_BPF3000_4200",
        tx_filter="TX_LPF400",
        pa_enable=False,
        lna_enable=True,
        category="rx_only",
        freq_range_mhz=(3000, 4200),
        description="RX only, BPF 3.0-4.2 GHz",
    ),
    # TDD / Half-duplex modes
    "trx0_400": DuplexerBand(
        name="trx0_400",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF400",
        pa_enable=True,
        lna_enable=True,
        category="tdd",
        freq_range_mhz=(0, 400),
        description="TDD 0-400 MHz",
    ),
    "trx400_1200": DuplexerBand(
        name="trx400_1200",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF1200",
        tx_filter="TX_LPF1200",
        pa_enable=True,
        lna_enable=True,
        category="tdd",
        freq_range_mhz=(400, 1200),
        description="TDD 400-1200 MHz",
    ),
    "trx1200_2100": DuplexerBand(
        name="trx1200_2100",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_LPF2100",
        tx_filter="TX_LPF2100",
        pa_enable=True,
        lna_enable=True,
        category="tdd",
        freq_range_mhz=(1200, 2100),
        description="TDD 1.2-2.1 GHz",
    ),
    "trx2100_3000": DuplexerBand(
        name="trx2100_3000",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_BPF2100_3000",
        tx_filter="TX_BYPASS",
        pa_enable=True,
        lna_enable=True,
        category="tdd",
        freq_range_mhz=(2100, 3000),
        description="TDD 2.1-3.0 GHz",
    ),
    "trx3000_4200": DuplexerBand(
        name="trx3000_4200",
        aliases=[],
        trx_mode="TRX_BYPASS",
        rx_filter="RX_BPF3000_4200",
        tx_filter="TX_BYPASS",
        pa_enable=True,
        lna_enable=True,
        category="tdd",
        freq_range_mhz=(3000, 4200),
        description="TDD 3.0-4.2 GHz",
    ),
}


def get_band_by_name(name: str) -> DuplexerBand | None:
    """Look up band by name or alias."""
    name_lower = name.lower()
    # Direct lookup
    if name_lower in DUPLEXER_BANDS:
        return DUPLEXER_BANDS[name_lower]
    # Check aliases
    for band in DUPLEXER_BANDS.values():
        if name_lower in [a.lower() for a in band.aliases]:
            return band
    return None


def list_bands_by_category(category: str = None) -> list[DuplexerBand]:
    """List bands, optionally filtered by category."""
    if category is None:
        return list(DUPLEXER_BANDS.values())
    return [b for b in DUPLEXER_BANDS.values() if b.category == category]


@dataclass
class USDRConfig:
    """USDR configuration (see docs.wsdr.io/hardware/devboard.html)."""

    center_freq_hz: int = 915_000_000
    sample_rate_hz: int = 10_000_000
    bandwidth_hz: int = 10_000_000
    gain: USDRGain = field(default_factory=USDRGain)
    rx_path: str = "LNAL"  # LNAH, LNAL, LNAW
    # DevBoard controls (docs.wsdr.io option names in comments)
    lna_enable: bool = True  # lna_: RX LNA +19.5dB
    pa_enable: bool = False  # pa_: TX PA +19.5dB (usually set via band)
    attenuator_db: int = 0  # attn_: 0-18 dB RX attenuation
    vctcxo_dac: int = 32768  # dac_: VCTCXO tuning 0-65535 (default=32768)
    gps_enable: bool = False  # gps_: GPS module
    osc_enable: bool = True  # osc_: Reference clock oscillator
    loopback_enable: bool = False  # lb_: RX->TX loopback
    uart_enable: bool = False  # uart_: UART interface
    # Duplexer band selection
    duplexer_band: str = ""  # path_: Empty = manual/auto


@dataclass
class USDRStatus:
    """Real-time USDR status."""

    connected: bool = False
    device_id: str | None = None
    temperature_c: float = 0.0
    actual_freq_hz: int = 0
    actual_sample_rate_hz: int = 0
    actual_bandwidth_hz: int = 0
    rx_path: str = ""
    streaming: bool = False


class USDRLibrary:
    """
    Low-level ctypes wrapper for libusdr.so.

    Maps C API functions to Python callables.
    """

    def __init__(self, lib_path: str = USDR_LIB_PATH):
        """Load the USDR library."""
        self._lib = None
        self._loaded = False

        if not Path(lib_path).exists():
            # Try finding in standard locations
            alt_path = ctypes.util.find_library("usdr")
            if alt_path:
                lib_path = alt_path
            else:
                return  # Library not available

        try:
            self._lib = ctypes.CDLL(lib_path)
            self._setup_functions()
            self._loaded = True
        except OSError:
            pass  # Library load failed

    def _setup_functions(self):
        """Define C function signatures."""
        if not self._lib:
            return

        # Device discovery
        # int usdr_dmd_discovery(const char* filter, unsigned max_buf, char* devlist)
        self._lib.usdr_dmd_discovery.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_char_p]
        self._lib.usdr_dmd_discovery.restype = ctypes.c_int

        # Device creation
        # int usdr_dmd_create_string(const char* conn_str, pdm_dev_t* odev)
        self._lib.usdr_dmd_create_string.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.usdr_dmd_create_string.restype = ctypes.c_int

        # Device close
        # int usdr_dmd_close(pdm_dev_t dev)
        self._lib.usdr_dmd_close.argtypes = [ctypes.c_void_p]
        self._lib.usdr_dmd_close.restype = ctypes.c_int

        # Parameter get/set (uint64)
        # int usdr_dme_get_uint(pdm_dev_t dev, const char* path, uint64_t* oval)
        self._lib.usdr_dme_get_uint.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.usdr_dme_get_uint.restype = ctypes.c_int

        # int usdr_dme_set_uint(pdm_dev_t dev, const char* path, uint64_t val)
        self._lib.usdr_dme_set_uint.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
        self._lib.usdr_dme_set_uint.restype = ctypes.c_int

        # Parameter get/set (u32)
        # int usdr_dme_get_u32(pdm_dev_t dev, const char* path, uint32_t* oval)
        self._lib.usdr_dme_get_u32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._lib.usdr_dme_get_u32.restype = ctypes.c_int

        # String parameter
        # int usdr_dme_set_string(pdm_dev_t dev, const char* path, const char* val)
        self._lib.usdr_dme_set_string.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self._lib.usdr_dme_set_string.restype = ctypes.c_int

        # Sample rate
        # int usdr_dmr_rate_set(pdm_dev_t dev, const char* rate_name, unsigned rate)
        self._lib.usdr_dmr_rate_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        self._lib.usdr_dmr_rate_set.restype = ctypes.c_int

        # Stream creation
        # int usdr_dms_create(pdm_dev_t dev, const char* sobj, const char* dformat,
        #                     unsigned channels, unsigned pktsyms, pusdr_dms_t* outu)
        self._lib.usdr_dms_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.usdr_dms_create.restype = ctypes.c_int

        # Stream destroy
        # int usdr_dms_destroy(pusdr_dms_t stream)
        self._lib.usdr_dms_destroy.argtypes = [ctypes.c_void_p]
        self._lib.usdr_dms_destroy.restype = ctypes.c_int

        # Stream operation (start/stop)
        # int usdr_dms_op(pusdr_dms_t stream, unsigned command, dm_time_t tm)
        self._lib.usdr_dms_op.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64]
        self._lib.usdr_dms_op.restype = ctypes.c_int

        # Stream sync
        # int usdr_dms_sync(pdm_dev_t dev, const char* synctype, unsigned scount, pusdr_dms_t* pstream)
        self._lib.usdr_dms_sync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.usdr_dms_sync.restype = ctypes.c_int

        # Stream receive
        # int usdr_dms_recv(pusdr_dms_t stream, void** buffs, unsigned timeout_ms, usdr_dms_recv_nfo_t* nfo)
        self._lib.usdr_dms_recv.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_void_p,  # nfo struct pointer
        ]
        self._lib.usdr_dms_recv.restype = ctypes.c_int

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def lib(self):
        return self._lib


# Recv info structure
class USDRRecvNfo(ctypes.Structure):
    _fields_ = [
        ("fsymtime", ctypes.c_uint64),
        ("totsyms", ctypes.c_uint),
        ("totlost", ctypes.c_uint),
        ("max_parts", ctypes.c_uint),
        ("extra", ctypes.c_uint64),
    ]


# Stream operation constants
USDR_DMS_START = 0
USDR_DMS_STOP = 1
USDR_DMS_START_AT = 2
USDR_DMS_STOP_AT = 3


class USDRDriver:
    """
    High-level USDR driver for RF Forensics pipeline.

    Provides:
    - Device discovery and connection
    - RF configuration (frequency, gain, bandwidth, path)
    - IQ sample streaming with format conversion
    - Real-time status monitoring
    """

    # VFS paths (see docs.wsdr.io/hardware/devboard.html)
    VFS_RX_FREQ = b"/dm/sdr/0/rx/freqency"  # Note: typo in firmware
    VFS_RX_FREQ_ALT = b"/dm/sdr/0/rx/frequency"
    VFS_RX_BW = b"/dm/sdr/0/rx/bandwidth"
    VFS_RX_PATH = b"/dm/sdr/0/rx/path"
    VFS_RX_GAIN_LNA = b"/dm/sdr/0/rx/gain/lna"
    VFS_RX_GAIN_VGA = b"/dm/sdr/0/rx/gain/vga"  # TIA
    VFS_RX_GAIN_PGA = b"/dm/sdr/0/rx/gain/pga"
    VFS_SENSOR_TEMP = b"/dm/sensor/temp"
    VFS_RATE = b"/dm/rate/rxtxadcdac"
    # DevBoard frontend controls (attn_, lna_, pa_, dac_, path_)
    VFS_FE_ATTN = b"/dm/sdr/0/fe/attn"  # attn_: 0-18 dB RX attenuator
    VFS_FE_LNA = b"/dm/sdr/0/fe/lna"  # lna_: RX LNA +19.5dB (QPL9547TR7)
    VFS_FE_PA = b"/dm/sdr/0/fe/pa"  # pa_: TX PA +19.5dB (QPL9547TR7)
    VFS_DAC_VCTCXO = b"/dm/sdr/0/dac_vctcxo"  # dac_: VCTCXO tuning 0-65535
    VFS_FE_PATH = b"/dm/sdr/0/fe/path"  # path_: Duplexer band selection
    # Additional DevBoard controls (gps_, osc_, lb_, uart_)
    VFS_FE_GPS = b"/dm/sdr/0/fe/gps"  # gps_: GPS module on/off
    VFS_FE_OSC = b"/dm/sdr/0/fe/osc"  # osc_: Reference clock oscillator on/off
    VFS_FE_LB = b"/dm/sdr/0/fe/lb"  # lb_: RX->TX loopback on/off
    VFS_FE_UART = b"/dm/sdr/0/fe/uart"  # uart_: UART interface on/off

    def __init__(self, lib_path: str = USDR_LIB_PATH):
        """Initialize driver."""
        self._usdr = USDRLibrary(lib_path)
        self._device = None
        self._stream = None
        self._config = USDRConfig()
        self._status = USDRStatus()
        self._device_id: str | None = None

        # Streaming
        self._streaming = False
        self._stream_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._sample_callback: Callable | None = None

        # Buffer for receiving samples (larger = less callback overhead)
        self._packet_size = 131072  # 13.1ms @ 10 MSPS
        self._rx_buffer = np.zeros(self._packet_size, dtype=np.complex64)

        # Overflow tracking
        self._overflow_count = 0
        self._last_overflow_time: float | None = None
        self._total_samples_received = 0

    @property
    def is_available(self) -> bool:
        """Check if USDR library is available."""
        return self._usdr.is_loaded

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._device is not None

    @property
    def is_streaming(self) -> bool:
        """Check if streaming is active."""
        return self._streaming

    def discover(self) -> list[USDRDevice]:
        """Discover available USDR devices."""
        if not self.is_available:
            logger.warning(
                "SDR discovery failed: libusdr not available. "
                "Check SDR_LIB_PATH or install libusdr.so"
            )
            return []

        devlist = ctypes.create_string_buffer(4096)
        result = self._usdr.lib.usdr_dmd_discovery(b"", 4096, devlist)

        if result < 0:
            logger.warning(
                f"SDR discovery failed: usdr_dmd_discovery returned {result}. "
                "Check USB connection and permissions."
            )
            return []

        devices = []
        raw = devlist.value.decode("utf-8", errors="ignore")
        logger.debug(f"Raw discovery result: '{raw}'")

        # Parse device list (format: devices separated by \v, fields by \t)
        if raw:
            for dev_str in raw.split("\v"):
                if dev_str.strip():
                    devices.append(
                        USDRDevice(id=dev_str.strip(), model="uSDR DevBoard", status="available")
                    )

        logger.info(f"SDR discovery found {len(devices)} device(s)")
        # Return actual devices only - no placeholders
        return devices

    def connect(self, device_id: str = "") -> bool:
        """
        Connect to USDR device.

        Args:
            device_id: Device connection string (empty for auto-select).

        Returns:
            True if connected successfully.
        """
        if not self.is_available:
            return False

        if self._device is not None:
            self.disconnect()

        dev_ptr = ctypes.c_void_p()
        result = self._usdr.lib.usdr_dmd_create_string(
            device_id.encode() if device_id else b"", ctypes.byref(dev_ptr)
        )

        if result < 0:
            return False

        self._device = dev_ptr
        self._device_id = device_id or "usb://0"
        self._status.connected = True
        self._status.device_id = self._device_id

        # Apply initial configuration
        self._apply_config()

        return True

    def disconnect(self) -> bool:
        """Disconnect from device."""
        if self._device is None:
            return True

        # Stop streaming first
        if self._streaming:
            self.stop_streaming()

        # Destroy stream if exists
        if self._stream is not None:
            self._usdr.lib.usdr_dms_destroy(self._stream)
            self._stream = None

        # Close device
        result = self._usdr.lib.usdr_dmd_close(self._device)
        self._device = None
        self._device_id = None
        self._status.connected = False
        self._status.device_id = None

        return result >= 0

    def configure(self, config: USDRConfig) -> bool:
        """
        Apply configuration to device.

        Args:
            config: USDR configuration.

        Returns:
            True if configuration applied successfully.
        """
        self._config = config
        if self._device is not None:
            return self._apply_config()
        return True

    def _apply_config(self) -> bool:
        """Apply current config to hardware."""
        if self._device is None:
            return False

        lib = self._usdr.lib
        dev = self._device

        # Set sample rate FIRST (before stream creation)
        rates = (ctypes.c_uint * 4)(self._config.sample_rate_hz, self._config.sample_rate_hz, 0, 0)
        lib.usdr_dme_set_uint(dev, self.VFS_RATE, ctypes.cast(rates, ctypes.c_void_p).value)

        # Set frequency (try both spellings for firmware compatibility)
        result = lib.usdr_dme_set_uint(dev, self.VFS_RX_FREQ, self._config.center_freq_hz)
        if result < 0:
            lib.usdr_dme_set_uint(dev, self.VFS_RX_FREQ_ALT, self._config.center_freq_hz)

        # Set bandwidth
        lib.usdr_dme_set_uint(dev, self.VFS_RX_BW, self._config.bandwidth_hz)

        # Set gains
        lib.usdr_dme_set_uint(dev, self.VFS_RX_GAIN_LNA, self._config.gain.lna_db)
        lib.usdr_dme_set_uint(dev, self.VFS_RX_GAIN_VGA, self._config.gain.tia_db)
        lib.usdr_dme_set_uint(dev, self.VFS_RX_GAIN_PGA, self._config.gain.pga_db)

        # Set RX path
        lib.usdr_dme_set_string(dev, self.VFS_RX_PATH, self._config.rx_path.encode())

        # DevBoard controls (see docs.wsdr.io/hardware/devboard.html)
        lib.usdr_dme_set_uint(dev, self.VFS_FE_LNA, 1 if self._config.lna_enable else 0)
        lib.usdr_dme_set_uint(dev, self.VFS_FE_PA, 1 if self._config.pa_enable else 0)
        lib.usdr_dme_set_uint(dev, self.VFS_FE_ATTN, self._config.attenuator_db)
        lib.usdr_dme_set_uint(dev, self.VFS_DAC_VCTCXO, self._config.vctcxo_dac)
        lib.usdr_dme_set_uint(dev, self.VFS_FE_GPS, 1 if self._config.gps_enable else 0)
        lib.usdr_dme_set_uint(dev, self.VFS_FE_OSC, 1 if self._config.osc_enable else 0)
        lib.usdr_dme_set_uint(dev, self.VFS_FE_LB, 1 if self._config.loopback_enable else 0)
        lib.usdr_dme_set_uint(dev, self.VFS_FE_UART, 1 if self._config.uart_enable else 0)

        # Update status
        self._update_status()

        return True

    def _update_status(self):
        """Read current status from hardware."""
        if self._device is None:
            return

        lib = self._usdr.lib
        dev = self._device

        # Read temperature
        temp_raw = ctypes.c_uint32()
        if lib.usdr_dme_get_u32(dev, self.VFS_SENSOR_TEMP, ctypes.byref(temp_raw)) >= 0:
            self._status.temperature_c = temp_raw.value / 256.0

        # Read actual frequency
        freq = ctypes.c_uint64()
        if lib.usdr_dme_get_uint(dev, self.VFS_RX_FREQ, ctypes.byref(freq)) >= 0:
            self._status.actual_freq_hz = freq.value

        self._status.actual_sample_rate_hz = self._config.sample_rate_hz
        self._status.actual_bandwidth_hz = self._config.bandwidth_hz
        self._status.rx_path = self._config.rx_path
        self._status.streaming = self._streaming

    def get_status(self) -> USDRStatus:
        """Get current device status."""
        if self._device is not None:
            self._update_status()
        return self._status

    def get_config(self) -> USDRConfig:
        """Get current configuration."""
        return self._config

    def start_streaming(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        """
        Start RX streaming.

        Args:
            callback: Function called with (samples: np.ndarray[complex64], timestamp: float)

        Returns:
            True if streaming started successfully.
        """
        if self._device is None:
            return False

        if self._streaming:
            return True

        lib = self._usdr.lib
        dev = self._device

        # Create RX stream
        stream_ptr = ctypes.c_void_p()
        result = lib.usdr_dms_create(
            dev,
            b"/ll/srx/0",
            b"cf32@ci16",  # Host: complex float32, Wire: complex int16
            0x1,  # Channel 0
            self._packet_size,
            ctypes.byref(stream_ptr),
        )

        if result < 0:
            return False

        self._stream = stream_ptr

        # Sync before start
        streams = (ctypes.c_void_p * 1)(stream_ptr)
        lib.usdr_dms_sync(dev, b"off", 1, streams)

        # Start stream
        result = lib.usdr_dms_op(stream_ptr, USDR_DMS_START, 0)
        if result < 0:
            lib.usdr_dms_destroy(stream_ptr)
            self._stream = None
            return False

        # Sync after start
        lib.usdr_dms_sync(dev, b"all", 1, streams)

        # Start receive thread
        self._sample_callback = callback
        self._stop_event.clear()
        self._streaming = True
        self._status.streaming = True

        self._stream_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._stream_thread.start()

        return True

    def stop_streaming(self) -> bool:
        """Stop RX streaming."""
        if not self._streaming:
            return True

        self._stop_event.set()

        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None

        if self._stream is not None:
            self._usdr.lib.usdr_dms_op(self._stream, USDR_DMS_STOP, 0)
            self._usdr.lib.usdr_dms_destroy(self._stream)
            self._stream = None

        self._streaming = False
        self._status.streaming = False
        self._sample_callback = None

        return True

    def _receive_loop(self):
        """Background thread for receiving samples."""
        lib = self._usdr.lib
        stream = self._stream

        # Allocate buffer
        buffer = np.zeros(self._packet_size, dtype=np.complex64)
        buffer_ptr = buffer.ctypes.data_as(ctypes.c_void_p)
        buffers = (ctypes.c_void_p * 1)(buffer_ptr)

        nfo = USDRRecvNfo()

        while not self._stop_event.is_set():
            # Receive samples
            result = lib.usdr_dms_recv(
                stream,
                buffers,
                100,  # 100ms timeout
                ctypes.byref(nfo),
            )

            if result >= 0 and nfo.totsyms > 0:
                # Track samples received
                self._total_samples_received += nfo.totsyms

                # Call callback with samples
                if self._sample_callback:
                    timestamp = nfo.fsymtime / self._config.sample_rate_hz
                    samples = buffer[: nfo.totsyms].copy()
                    try:
                        self._sample_callback(samples, timestamp)
                    except Exception:
                        pass  # Don't crash receive thread

                # Track lost samples (overflow)
                if nfo.totlost > 0:
                    self._overflow_count += nfo.totlost
                    self._last_overflow_time = time.time()
                    # Warning logged by caller if they subscribe to overflow events

    def set_frequency(self, freq_hz: int) -> bool:
        """Set center frequency."""
        self._config.center_freq_hz = freq_hz
        if self._device:
            result = self._usdr.lib.usdr_dme_set_uint(self._device, self.VFS_RX_FREQ, freq_hz)
            return result >= 0
        return True

    def set_gain(self, lna_db: int = None, tia_db: int = None, pga_db: int = None) -> bool:
        """Set gain stages."""
        if lna_db is not None:
            self._config.gain.lna_db = max(0, min(30, lna_db))
        if tia_db is not None:
            self._config.gain.tia_db = max(0, min(12, tia_db))
        if pga_db is not None:
            self._config.gain.pga_db = max(0, min(32, pga_db))

        if self._device:
            lib = self._usdr.lib
            lib.usdr_dme_set_uint(self._device, self.VFS_RX_GAIN_LNA, self._config.gain.lna_db)
            lib.usdr_dme_set_uint(self._device, self.VFS_RX_GAIN_VGA, self._config.gain.tia_db)
            lib.usdr_dme_set_uint(self._device, self.VFS_RX_GAIN_PGA, self._config.gain.pga_db)

        return True

    def set_rx_path(self, path: str) -> bool:
        """Set RX antenna path (LNAH, LNAL, LNAW)."""
        if path not in ("LNAH", "LNAL", "LNAW"):
            return False

        self._config.rx_path = path
        if self._device:
            self._usdr.lib.usdr_dme_set_string(self._device, self.VFS_RX_PATH, path.encode())
        return True

    def set_duplexer_band(self, band_name: str) -> bool:
        """
        Set duplexer band configuration.

        Args:
            band_name: Band name (e.g., "band2", "gsm900", "rxlpf1200", "trx0_400")

        Returns:
            True if band was set successfully.
        """
        band = get_band_by_name(band_name)
        if band is None:
            return False

        self._config.duplexer_band = band.name
        self._config.lna_enable = band.lna_enable

        if self._device:
            lib = self._usdr.lib

            # Set the frontend path (duplexer band)
            lib.usdr_dme_set_string(self._device, self.VFS_FE_PATH, band.name.encode())

            # Set PA/LNA enables based on band config
            lib.usdr_dme_set_uint(self._device, self.VFS_FE_PA, 1 if band.pa_enable else 0)
            lib.usdr_dme_set_uint(self._device, self.VFS_FE_LNA, 1 if band.lna_enable else 0)

        return True

    def get_duplexer_bands(self, category: str = None) -> list[dict[str, Any]]:
        """
        Get list of available duplexer bands.

        Args:
            category: Optional filter ("cellular", "tx_only", "rx_only", "tdd")

        Returns:
            List of band info dicts.
        """
        bands = list_bands_by_category(category)
        return [
            {
                "name": b.name,
                "aliases": b.aliases,
                "category": b.category,
                "freq_range_mhz": list(b.freq_range_mhz),
                "description": b.description,
                "pa_enable": b.pa_enable,
                "lna_enable": b.lna_enable,
                "trx_mode": b.trx_mode,
                "rx_filter": b.rx_filter,
                "tx_filter": b.tx_filter,
            }
            for b in bands
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dict for API response."""
        result = {
            "device_type": "usdr",
            "center_freq_hz": self._config.center_freq_hz,
            "sample_rate_hz": self._config.sample_rate_hz,
            "bandwidth_hz": self._config.bandwidth_hz,
            "gain": {
                "lna_db": self._config.gain.lna_db,
                "tia_db": self._config.gain.tia_db,
                "pga_db": self._config.gain.pga_db,
                "total_db": self._config.gain.total_db,
            },
            "rx_path": self._config.rx_path,
            "devboard": {
                "lna_enable": self._config.lna_enable,
                "pa_enable": self._config.pa_enable,
                "attenuator_db": self._config.attenuator_db,
                "vctcxo_dac": self._config.vctcxo_dac,
                "gps_enable": self._config.gps_enable,
                "osc_enable": self._config.osc_enable,
                "loopback_enable": self._config.loopback_enable,
                "uart_enable": self._config.uart_enable,
            },
            "duplexer_band": self._config.duplexer_band,
        }

        # Add band details if a band is selected
        if self._config.duplexer_band:
            band = get_band_by_name(self._config.duplexer_band)
            if band:
                result["duplexer_band_info"] = {
                    "name": band.name,
                    "category": band.category,
                    "freq_range_mhz": list(band.freq_range_mhz),
                    "description": band.description,
                    "pa_enable": band.pa_enable,
                    "lna_enable": band.lna_enable,
                }

        return result

    def status_to_dict(self) -> dict[str, Any]:
        """Convert status to dict for API response."""
        status = self.get_status()
        return {
            "connected": status.connected,
            "device_id": status.device_id,
            "temperature_c": status.temperature_c,
            "actual_freq_hz": status.actual_freq_hz,
            "actual_sample_rate_hz": status.actual_sample_rate_hz,
            "actual_bandwidth_hz": status.actual_bandwidth_hz,
            "rx_path": status.rx_path,
            "streaming": status.streaming,
        }

    def get_overflow_stats(self) -> dict[str, Any]:
        """Get overflow/sample loss statistics."""
        return {
            "total_overflows": self._overflow_count,
            "last_overflow_time": self._last_overflow_time,
            "total_samples_received": self._total_samples_received,
        }

    def reset_overflow_stats(self) -> None:
        """Reset overflow statistics."""
        self._overflow_count = 0
        self._last_overflow_time = None
        self._total_samples_received = 0


# Note: The old get_usdr_driver() singleton has been replaced by SDRManager.
# Use: from rf_forensics.sdr.manager import get_sdr_manager
# See: rf_forensics/sdr/manager.py
