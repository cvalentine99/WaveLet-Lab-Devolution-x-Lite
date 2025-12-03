"""
GPU RF Forensics Engine - SDR Interface

Abstract SDR interface and concrete implementations for various SDR hardware.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# Import SDRConfig from centralized config schema (Pydantic model with validation)
from rf_forensics.config.schema import SDRConfig


@dataclass
class SDRDeviceInfo:
    """Information about an SDR device."""

    device_type: str
    serial: str = ""
    firmware_version: str = ""
    hardware_version: str = ""
    supported_sample_rates: list = field(default_factory=list)
    supported_freq_range: tuple = (0, 6e9)
    supported_gain_range: tuple = (0, 76)
    num_rx_channels: int = 1
    num_tx_channels: int = 1


@dataclass
class StreamMetadata:
    """Metadata for a stream buffer."""

    timestamp: float
    center_freq_hz: float
    sample_rate_hz: float
    num_samples: int
    overflow: bool = False
    sequence_num: int = 0


class SDRInterface(ABC):
    """
    Abstract base class for SDR device interfaces.

    All SDR implementations must:
    - Support configuration of center frequency, sample rate, bandwidth, gain
    - Write directly to pinned memory buffers
    - Support async callback pattern for streaming
    - Handle device errors gracefully
    """

    def __init__(self, config: SDRConfig | None = None):
        """
        Initialize the SDR interface.

        Args:
            config: Optional SDR configuration.
        """
        self._config = config or SDRConfig()
        self._streaming = False
        self._callback: Callable | None = None
        self._device_info: SDRDeviceInfo | None = None

    @property
    def config(self) -> SDRConfig:
        """Current configuration."""
        return self._config

    @property
    def is_streaming(self) -> bool:
        """Whether the device is currently streaming."""
        return self._streaming

    @abstractmethod
    def configure(
        self,
        center_freq_hz: float | None = None,
        sample_rate_hz: float | None = None,
        bandwidth_hz: float | None = None,
        gain_db: float | None = None,
        antenna: str | None = None,
    ) -> dict[str, float]:
        """
        Configure the SDR device.

        Args:
            center_freq_hz: Center frequency in Hz.
            sample_rate_hz: Sample rate in Hz.
            bandwidth_hz: Bandwidth in Hz.
            gain_db: Gain in dB.
            antenna: Antenna port name.

        Returns:
            Dictionary of actual (achieved) parameters.
        """
        pass

    @abstractmethod
    def start_streaming(self, callback: Callable[[np.ndarray, StreamMetadata], None]) -> None:
        """
        Start streaming data from the SDR.

        Args:
            callback: Function called with (samples, metadata) for each buffer.
        """
        pass

    @abstractmethod
    def stop_streaming(self) -> None:
        """Stop streaming data."""
        pass

    @abstractmethod
    def get_device_info(self) -> SDRDeviceInfo:
        """
        Get device information.

        Returns:
            SDRDeviceInfo with device details.
        """
        pass

    def read_samples(self, num_samples: int) -> tuple[np.ndarray, StreamMetadata]:
        """
        Read samples synchronously (blocking).

        Args:
            num_samples: Number of samples to read.

        Returns:
            Tuple of (samples, metadata).
        """
        raise NotImplementedError("Synchronous read not supported")

    def close(self) -> None:
        """Close the device and release resources."""
        if self._streaming:
            self.stop_streaming()


class USRPInterface(SDRInterface):
    """
    USRP SDR interface using UHD Python bindings.

    Supports USRP B200/B210/X300/X310 series.
    """

    def __init__(self, config: SDRConfig | None = None, device_args: str = ""):
        super().__init__(config)
        self._device_args = device_args
        self._usrp = None
        self._streamer = None

        try:
            import uhd

            self._uhd = uhd
        except ImportError:
            raise ImportError(
                "UHD Python bindings not found. Install with: conda install -c conda-forge uhd"
            )

    def _init_device(self) -> None:
        """Initialize the USRP device."""
        if self._usrp is not None:
            return

        self._usrp = self._uhd.usrp.MultiUSRP(self._device_args)

        # Get device info
        info = self._usrp.get_usrp_rx_info()
        self._device_info = SDRDeviceInfo(
            device_type=info.get("mboard_id", "USRP"),
            serial=info.get("mboard_serial", ""),
            firmware_version=info.get("fpga_version", ""),
            hardware_version=info.get("mboard_name", ""),
            supported_sample_rates=[1e6, 2e6, 5e6, 10e6, 20e6, 40e6, 61.44e6],
            supported_freq_range=(
                self._usrp.get_rx_freq_range().start(),
                self._usrp.get_rx_freq_range().stop(),
            ),
            supported_gain_range=(
                self._usrp.get_rx_gain_range().start(),
                self._usrp.get_rx_gain_range().stop(),
            ),
        )

    def configure(
        self,
        center_freq_hz: float | None = None,
        sample_rate_hz: float | None = None,
        bandwidth_hz: float | None = None,
        gain_db: float | None = None,
        antenna: str | None = None,
    ) -> dict[str, float]:
        """Configure the USRP."""
        self._init_device()

        if sample_rate_hz is not None:
            self._usrp.set_rx_rate(sample_rate_hz)
            self._config.sample_rate_hz = self._usrp.get_rx_rate()

        if center_freq_hz is not None:
            tune_req = self._uhd.types.TuneRequest(center_freq_hz)
            self._usrp.set_rx_freq(tune_req)
            self._config.center_freq_hz = self._usrp.get_rx_freq()

        if bandwidth_hz is not None:
            self._usrp.set_rx_bandwidth(bandwidth_hz)
            self._config.bandwidth_hz = self._usrp.get_rx_bandwidth()

        if gain_db is not None:
            self._usrp.set_rx_gain(gain_db)
            self._config.gain_db = self._usrp.get_rx_gain()

        if antenna is not None:
            self._usrp.set_rx_antenna(antenna)
            self._config.antenna = self._usrp.get_rx_antenna()

        return {
            "center_freq_hz": self._config.center_freq_hz,
            "sample_rate_hz": self._config.sample_rate_hz,
            "bandwidth_hz": self._config.bandwidth_hz,
            "gain_db": self._config.gain_db,
        }

    def start_streaming(self, callback: Callable[[np.ndarray, StreamMetadata], None]) -> None:
        """Start streaming from USRP."""
        self._init_device()

        if self._streaming:
            raise RuntimeError("Already streaming")

        # Create streamer
        st_args = self._uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self._streamer = self._usrp.get_rx_stream(st_args)

        self._callback = callback
        self._streaming = True
        self._stop_event = threading.Event()

        # Start streaming thread
        self._stream_thread = threading.Thread(target=self._usrp_stream_loop, daemon=True)
        self._stream_thread.start()

    def _usrp_stream_loop(self) -> None:
        """USRP streaming loop."""
        # Issue stream command
        stream_cmd = self._uhd.types.StreamCMD(self._uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self._streamer.issue_stream_cmd(stream_cmd)

        num_samps = int(self._config.sample_rate_hz * 0.01)  # 10ms
        recv_buffer = np.zeros(num_samps, dtype=np.complex64)
        metadata = self._uhd.types.RXMetadata()

        while not self._stop_event.is_set():
            num_rx = self._streamer.recv(recv_buffer, metadata)

            if num_rx > 0 and self._callback:
                stream_meta = StreamMetadata(
                    timestamp=metadata.time_spec.get_real_secs(),
                    center_freq_hz=self._config.center_freq_hz,
                    sample_rate_hz=self._config.sample_rate_hz,
                    num_samples=num_rx,
                    overflow=metadata.error_code == self._uhd.types.RXMetadataErrorCode.overflow,
                )
                self._callback(recv_buffer[:num_rx].copy(), stream_meta)

        # Stop streaming
        stream_cmd = self._uhd.types.StreamCMD(self._uhd.types.StreamMode.stop_cont)
        self._streamer.issue_stream_cmd(stream_cmd)

    def stop_streaming(self) -> None:
        """Stop USRP streaming."""
        if not self._streaming:
            return

        self._stop_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
        self._streaming = False

    def get_device_info(self) -> SDRDeviceInfo:
        """Get USRP device info."""
        self._init_device()
        return self._device_info

    def close(self) -> None:
        """Close USRP device."""
        self.stop_streaming()
        self._usrp = None


class HackRFInterface(SDRInterface):
    """
    HackRF One SDR interface.

    Note: Requires hackrf library.
    """

    def __init__(self, config: SDRConfig | None = None):
        super().__init__(config)
        self._device = None

        try:
            import hackrf

            self._hackrf = hackrf
        except ImportError:
            raise ImportError("hackrf library not found")

    def configure(
        self,
        center_freq_hz: float | None = None,
        sample_rate_hz: float | None = None,
        bandwidth_hz: float | None = None,
        gain_db: float | None = None,
        antenna: str | None = None,
    ) -> dict[str, float]:
        """Configure HackRF."""
        # HackRF configuration implementation
        raise NotImplementedError("HackRF support not yet implemented")

    def start_streaming(self, callback: Callable[[np.ndarray, StreamMetadata], None]) -> None:
        """Start HackRF streaming."""
        raise NotImplementedError("HackRF support not yet implemented")

    def stop_streaming(self) -> None:
        """Stop HackRF streaming."""
        raise NotImplementedError("HackRF support not yet implemented")

    def get_device_info(self) -> SDRDeviceInfo:
        """Get HackRF device info."""
        return SDRDeviceInfo(
            device_type="HackRF One",
            supported_sample_rates=[2e6, 4e6, 8e6, 10e6, 12.5e6, 16e6, 20e6],
            supported_freq_range=(1e6, 6e9),
            supported_gain_range=(0, 62),
        )


class USDRInterface(SDRInterface):
    """
    SDR interface adapter for USDR DevBoard.

    Uses SDRManager singleton for unified driver ownership.
    This adapter provides SDRInterface compatibility while
    ensuring only one driver instance exists system-wide.
    """

    def __init__(self, config: SDRConfig | None = None):
        super().__init__(config)
        from rf_forensics.sdr.manager import get_sdr_manager
        from rf_forensics.sdr.usdr_driver import USDRGain

        self._manager = get_sdr_manager()
        self._USDRGain = USDRGain

    def configure(
        self,
        center_freq_hz: float | None = None,
        sample_rate_hz: float | None = None,
        bandwidth_hz: float | None = None,
        gain_db: float | None = None,
        antenna: str | None = None,
    ) -> dict[str, float]:
        """Configure the USDR device via SDRManager."""
        if center_freq_hz is not None:
            self._config.center_freq_hz = center_freq_hz
        if sample_rate_hz is not None:
            self._config.sample_rate_hz = sample_rate_hz
        if bandwidth_hz is not None:
            self._config.bandwidth_hz = bandwidth_hz
        if gain_db is not None:
            self._config.gain_db = gain_db

        # Connect via manager if not already
        if not self._manager.is_connected:
            devices = self._manager.discover()
            if devices:
                self._manager.connect(devices[0].id)

        # Build USDR config and apply via manager
        from rf_forensics.sdr.usdr_driver import USDRConfig

        freq_hz = int(self._config.center_freq_hz)
        if freq_hz < 0:
            raise ValueError(f"Frequency must be positive, got {freq_hz} Hz")

        # Split gain across LNA/TIA/PGA stages
        total_gain = int(self._config.gain_db)
        lna_gain = min(30, total_gain)
        remaining = total_gain - lna_gain
        tia_gain = min(12, remaining)
        pga_gain = min(32, remaining - tia_gain)

        usdr_config = USDRConfig(
            center_freq_hz=freq_hz,
            sample_rate_hz=int(self._config.sample_rate_hz),
            bandwidth_hz=int(self._config.bandwidth_hz),
            gain=self._USDRGain(lna_db=lna_gain, tia_db=tia_gain, pga_db=pga_gain),
        )
        self._manager.configure(usdr_config)

        return {
            "center_freq_hz": self._config.center_freq_hz,
            "sample_rate_hz": self._config.sample_rate_hz,
            "bandwidth_hz": self._config.bandwidth_hz,
            "gain_db": self._config.gain_db,
        }

    def start_streaming(self, callback: Callable[[np.ndarray, StreamMetadata], None]):
        """Start streaming from USDR via SDRManager."""
        self._callback = callback
        self._streaming = True

        def sample_handler(samples: np.ndarray, timestamp: float):
            if self._callback:
                metadata = StreamMetadata(
                    timestamp=timestamp,
                    center_freq_hz=self._config.center_freq_hz,
                    sample_rate_hz=self._config.sample_rate_hz,
                    num_samples=len(samples),
                )
                self._callback(samples, metadata)

        self._manager.start_streaming(sample_handler)

    def stop_streaming(self):
        """Stop streaming via SDRManager."""
        self._streaming = False
        self._manager.stop_streaming()

    def read_samples(self, num_samples: int) -> tuple[np.ndarray, StreamMetadata]:
        """Read samples (blocking) - not supported with SDRManager."""
        raise NotImplementedError("Synchronous read not supported - use streaming")

    def get_device_info(self) -> SDRDeviceInfo:
        """Get USDR device info from SDRManager capabilities."""
        caps = self._manager.get_capabilities()
        return SDRDeviceInfo(
            device_type="USDR DevBoard",
            supported_sample_rates=[1e6, 2e6, 4e6, 8e6, 10e6, 20e6, 40e6, 61.44e6],
            supported_freq_range=caps.freq_range_hz,
            supported_gain_range=caps.gain_range_db,
        )

    def close(self) -> None:
        """
        Close the USDR interface.

        Note: Does NOT close the SDR hardware - that's owned by SDRManager.
        Just stops streaming if active and clears local state.
        """
        if self._streaming:
            self.stop_streaming()
        self._callback = None
        # Don't close manager - it's a singleton that other components may use


def create_sdr(device_type: str, **config) -> SDRInterface:
    """
    Factory function to create an SDR interface."""
    sdr_config = SDRConfig(**{k: v for k, v in config.items() if k != "device_args"})
    device_args = config.get("device_args", "")

    if device_type.lower() == "usrp":
        return USRPInterface(sdr_config, device_args)
    elif device_type.lower() == "hackrf":
        return HackRFInterface(sdr_config)
    elif device_type.lower() == "usdr":
        return USDRInterface(sdr_config)
    else:
        raise ValueError(f"Unknown SDR type: {device_type}")
