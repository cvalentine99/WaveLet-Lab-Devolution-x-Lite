"""
GPU RF Forensics Engine - BLE GFSK Demodulator

Bluetooth Low Energy Gaussian FSK demodulator.
"""

from __future__ import annotations

from dataclasses import dataclass

from rf_forensics.core.gpu_compat import CUPY_AVAILABLE, cp, np


@dataclass
class BLEPacket:
    """Decoded BLE packet."""

    access_address: int
    pdu_header: int
    payload: bytes
    crc: int
    crc_valid: bool
    channel: int = 37
    phy_mode: str = "1M"
    packet_id: int = 0
    rssi_dbm: float = -100.0

    def to_frontend_dict(self) -> dict:
        """Convert to frontend-compatible JSON format."""
        # Determine PDU type from header
        pdu_type_val = self.pdu_header & 0x0F
        pdu_types = {
            0x00: "ADV_IND",
            0x01: "ADV_DIRECT_IND",
            0x02: "ADV_NONCONN_IND",
            0x03: "SCAN_REQ",
            0x04: "SCAN_RSP",
            0x05: "CONNECT_IND",
            0x06: "ADV_SCAN_IND",
        }
        pdu_type = pdu_types.get(pdu_type_val, f"UNKNOWN_{pdu_type_val:02X}")

        return {
            "packet_id": self.packet_id,
            "channel": self.channel,
            "access_address": f"{self.access_address:08X}",
            "pdu_type": pdu_type,
            "payload_hex": self.payload.hex(),
            "crc_valid": self.crc_valid,
            "rssi_dbm": self.rssi_dbm,
        }


@dataclass
class BLEConfig:
    """BLE PHY configuration."""

    phy_mode: str = "1M"  # "1M", "2M", "Coded"
    sample_rate: float = 4e6  # Recommended 4× symbol rate
    access_address: int = 0x8E89BED6  # Advertising AA


class BLEDemodulator:
    """
    GPU-accelerated BLE GFSK demodulator.

    BLE uses Gaussian FSK with:
    - 1M PHY: 1 Msym/s, ±250 kHz deviation
    - 2M PHY: 2 Msym/s, ±500 kHz deviation
    - BT product = 0.5

    Demodulation approach:
    1. Frequency discrimination: Δφ[n] = arg(x[n] × x*[n-1])
    2. Gaussian matched filtering
    3. Symbol timing recovery
    4. Bit decision via threshold
    """

    # BLE constants
    ADVERTISING_AA = 0x8E89BED6
    ADVERTISING_CHANNELS = [37, 38, 39]  # Channels 37-39
    CRC_INIT = 0x555555

    def __init__(self, phy_mode: str = "1M", sample_rate: float = 4e6):
        """
        Initialize BLE demodulator.

        Args:
            phy_mode: PHY mode ("1M" or "2M").
            sample_rate: Sample rate in Hz.

        Raises:
            ValueError: If parameters are invalid.
        """
        self._phy_mode = phy_mode
        self._sample_rate = sample_rate

        # PHY parameters
        if phy_mode == "1M":
            self._symbol_rate = 1e6
            self._deviation = 250e3
        elif phy_mode == "2M":
            self._symbol_rate = 2e6
            self._deviation = 500e3
        else:
            raise ValueError(f"Unsupported PHY mode: {phy_mode}")

        # Validate sample rate is sufficient
        if sample_rate < self._symbol_rate:
            raise ValueError(
                f"Sample rate ({sample_rate} Hz) must be >= symbol rate ({self._symbol_rate} Hz)"
            )

        self._samples_per_symbol = int(sample_rate / self._symbol_rate)
        if self._samples_per_symbol < 1:
            raise ValueError(
                f"Sample rate too low: samples_per_symbol = {self._samples_per_symbol}"
            )

        self._bt_product = 0.5

        # Generate Gaussian filter
        self._gaussian_filter = self._design_gaussian_filter()

    def _design_gaussian_filter(self) -> cp.ndarray:
        """Design Gaussian matched filter for BLE demodulation."""
        # Filter spans 3 symbol periods
        num_taps = max(3 * self._samples_per_symbol, 1)
        t = cp.arange(num_taps, dtype=cp.float64) / self._sample_rate
        t = t - (num_taps - 1) / (2 * self._sample_rate)

        # Gaussian filter
        alpha = cp.sqrt(cp.log(cp.array(2.0))) / (2 * cp.pi * self._bt_product / self._symbol_rate)
        h = cp.sqrt(cp.pi) / alpha * cp.exp(-((cp.pi * t / alpha) ** 2))

        # Normalize - protect against zero sum
        h_sum = float(cp.sum(h))
        if h_sum == 0 or not np.isfinite(h_sum):
            # Fallback to simple averaging filter
            h = cp.ones(num_taps, dtype=cp.float32) / num_taps
        else:
            h = h / h_sum

        return h.astype(cp.float32)

    def demodulate_bits(self, signal: cp.ndarray) -> np.ndarray:
        """
        Demodulate signal to bits.

        Args:
            signal: Complex input signal.

        Returns:
            Array of demodulated bits.
        """
        signal = cp.asarray(signal)

        # Step 1: Frequency discrimination
        # Instantaneous frequency via phase difference
        phase_diff = cp.angle(signal[1:] * cp.conj(signal[:-1]))

        # Normalize to [-1, 1] range
        # Protect against division by zero (shouldn't happen with valid config)
        normalization_factor = 2 * cp.pi * self._deviation / self._sample_rate
        if abs(float(normalization_factor)) < 1e-12:
            normalization_factor = 1e-12
        freq_norm = phase_diff / normalization_factor

        # Step 2: Gaussian matched filtering
        filtered = self._convolve(freq_norm, self._gaussian_filter)

        # Step 3: Sample at symbol rate (vectorized)
        num_symbols = len(filtered) // self._samples_per_symbol
        if num_symbols < 1:
            return np.array([], dtype=np.uint8)

        # Vectorized sampling at symbol centers
        sample_offset = self._samples_per_symbol // 2
        indices = cp.arange(num_symbols) * self._samples_per_symbol + sample_offset
        indices = indices[indices < len(filtered)]
        symbols = filtered[indices.astype(cp.int64)]

        # Step 4: Binary decision
        bits = (symbols > 0).astype(cp.uint8)

        if CUPY_AVAILABLE:
            return cp.asnumpy(bits)
        return bits

    def _convolve(self, signal: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
        """GPU-accelerated convolution."""
        n = len(signal) + len(kernel) - 1
        n_fft = 1 << (n - 1).bit_length()

        signal_fft = cp.fft.fft(signal, n_fft)
        kernel_fft = cp.fft.fft(kernel, n_fft)

        result = cp.fft.ifft(signal_fft * kernel_fft).real

        # Compensate for filter delay
        delay = len(kernel) // 2
        return result[delay : delay + len(signal)]

    def detect_access_address(
        self, signal: cp.ndarray, aa: int = None, threshold: float = 0.7
    ) -> list[int]:
        """
        Detect access address in signal.

        Args:
            signal: Input signal.
            aa: Access address to search for (default: advertising).
            threshold: Correlation threshold.

        Returns:
            List of detected start indices.
        """
        if aa is None:
            aa = self.ADVERTISING_AA

        # Demodulate to bits
        bits = self.demodulate_bits(signal)

        # Convert AA to bit sequence (LSB first)
        aa_bits = np.array([(aa >> i) & 1 for i in range(32)], dtype=np.uint8)

        # Correlate
        detections = []
        for i in range(len(bits) - 32):
            match_count = np.sum(bits[i : i + 32] == aa_bits)
            if match_count / 32.0 >= threshold:
                detections.append(i)

        return detections

    def decode_packet(self, bits: np.ndarray, start_idx: int = 0) -> BLEPacket | None:
        """
        Decode BLE packet from bits.

        Args:
            bits: Demodulated bit array.
            start_idx: Start index of access address.

        Returns:
            BLEPacket if successful, None otherwise.
        """
        if len(bits) < start_idx + 80:  # Minimum packet size
            return None

        # Extract fields (all LSB first) - vectorized bit extraction
        idx = start_idx

        # Access Address (32 bits) - vectorized
        aa = self._bits_to_int(bits[idx : idx + 32], 32)
        idx += 32

        # PDU Header (16 bits) - vectorized
        pdu_header = self._bits_to_int(bits[idx : idx + 16], 16)
        idx += 16

        # Payload length from header
        payload_len = (pdu_header >> 8) & 0xFF

        if len(bits) < idx + payload_len * 8 + 24:  # +24 for CRC
            return None

        # Payload
        payload_bits = bits[idx : idx + payload_len * 8]
        payload = self._bits_to_bytes(payload_bits)
        idx += payload_len * 8

        # CRC (24 bits) - must have all 24 bits available
        if idx + 24 > len(bits):
            return None  # Incomplete CRC

        # CRC (24 bits) - vectorized
        crc = self._bits_to_int(bits[idx : idx + 24], 24)

        # Validate CRC
        crc_valid = self._check_crc(pdu_header, payload, crc)

        return BLEPacket(
            access_address=aa,
            pdu_header=pdu_header,
            payload=payload,
            crc=crc,
            crc_valid=crc_valid,
            phy_mode=self._phy_mode,
        )

    def _bits_to_int(self, bits: np.ndarray, num_bits: int) -> int:
        """Convert bit array to integer (LSB first) - vectorized."""
        if len(bits) < num_bits:
            num_bits = len(bits)
        if num_bits == 0:
            return 0

        # Create bit position weights: [1, 2, 4, 8, ...]
        weights = 1 << np.arange(num_bits, dtype=np.uint64)

        # Vectorized dot product
        return int(np.dot(bits[:num_bits].astype(np.uint64), weights))

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes (LSB first) - vectorized."""
        num_bytes = len(bits) // 8
        if num_bytes == 0:
            return b""

        # Reshape bits into (num_bytes, 8) matrix
        bits_matrix = bits[: num_bytes * 8].reshape(num_bytes, 8)

        # Create bit position weights for LSB-first: [1, 2, 4, 8, 16, 32, 64, 128]
        weights = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

        # Vectorized bit-to-byte conversion: dot product of each row with weights
        byte_values = np.dot(bits_matrix.astype(np.uint8), weights)

        return bytes(byte_values.astype(np.uint8))

    def _check_crc(self, header: int, payload: bytes, crc: int) -> bool:
        """
        Verify BLE CRC-24.

        BLE uses CRC-24 with polynomial x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1
        Polynomial in reversed bit order (LSB-first): 0x00065B with MSB set = 0x100065B
        Note: For LFSR implementation, we use the reversed polynomial 0xDA6000
        """
        # CRC polynomial (reversed for LSB-first LFSR implementation)
        # x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1 = 0x100065B
        # Reversed for right-shifting LFSR: 0xDA6000
        poly = 0xDA6000

        # Initialize with CRC_INIT
        crc_calc = self.CRC_INIT

        # Process header
        for i in range(16):
            bit = (header >> i) & 1
            if (crc_calc ^ bit) & 1:
                crc_calc = (crc_calc >> 1) ^ poly
            else:
                crc_calc >>= 1

        # Process payload
        for byte in payload:
            for i in range(8):
                bit = (byte >> i) & 1
                if (crc_calc ^ bit) & 1:
                    crc_calc = (crc_calc >> 1) ^ poly
                else:
                    crc_calc >>= 1

        return crc_calc == crc

    @property
    def phy_mode(self) -> str:
        return self._phy_mode

    @property
    def symbol_rate(self) -> float:
        return self._symbol_rate

    @property
    def samples_per_symbol(self) -> int:
        return self._samples_per_symbol
