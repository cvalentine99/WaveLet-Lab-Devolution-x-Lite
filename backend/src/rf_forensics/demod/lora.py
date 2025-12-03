"""
GPU RF Forensics Engine - LoRa CSS Demodulator

Chirp Spread Spectrum demodulator for LoRa signals.
"""

from __future__ import annotations

from dataclasses import dataclass

from rf_forensics.core.gpu_compat import CUPY_AVAILABLE, cp, np


@dataclass
class LoRaFrame:
    """Decoded LoRa frame."""

    symbols: np.ndarray
    sf: int  # Spreading Factor
    bw: float  # Bandwidth in Hz
    sync_word: int
    snr_estimate: float
    timestamp: float = 0.0
    preamble_start: int = 0
    frame_id: int = 0
    center_freq_hz: float = 0.0
    coding_rate: str = "4/5"
    crc_valid: bool = True
    rssi_dbm: float = -100.0

    def to_frontend_dict(self) -> dict:
        """Convert to frontend-compatible JSON format."""
        # Convert symbols to payload hex
        payload_bytes = bytes(self.symbols.astype(np.uint8).tolist())
        payload_hex = payload_bytes.hex()

        return {
            "frame_id": self.frame_id,
            "center_freq_hz": self.center_freq_hz,
            "spreading_factor": self.sf,
            "bandwidth_hz": self.bw,
            "coding_rate": self.coding_rate,
            "payload_hex": payload_hex,
            "crc_valid": self.crc_valid,
            "snr_db": self.snr_estimate,
            "rssi_dbm": self.rssi_dbm,
        }


@dataclass
class LoRaConfig:
    """LoRa modulation parameters."""

    spreading_factor: int = 7  # 7-12
    bandwidth: float = 125000  # 125kHz, 250kHz, 500kHz
    coding_rate: str = "4/5"  # 4/5, 4/6, 4/7, 4/8
    implicit_header: bool = False
    crc_enabled: bool = True


class LoRaDemodulator:
    """
    GPU-accelerated LoRa Chirp Spread Spectrum demodulator.

    LoRa uses CSS (Chirp Spread Spectrum) modulation where:
    - Symbols are encoded as cyclic shifts of a base chirp
    - Spreading Factor (SF) determines bits per symbol (SF = log2(2^SF) chips)
    - Demodulation: multiply by down-chirp, FFT, find peak

    Algorithm:
    1. Generate reference down-chirp for configured SF/BW
    2. Multiply received signal by down-chirp (de-chirping)
    3. FFT of de-chirped signal
    4. Peak frequency maps to symbol value
    """

    # Maximum samples per symbol to prevent memory exhaustion
    MAX_SAMPLES_PER_SYMBOL = 1_000_000  # 1M samples max

    def __init__(self, sf: int = 7, bw: float = 125000, sample_rate: float | None = None):
        """
        Initialize LoRa demodulator.

        Args:
            sf: Spreading factor (7-12).
            bw: Bandwidth in Hz (125kHz, 250kHz, 500kHz).
            sample_rate: Sample rate (defaults to 2× bandwidth).

        Raises:
            ValueError: If parameters are out of valid range.
        """
        # Validate spreading factor (allow 5-12 for experimental configurations)
        if sf < 5 or sf > 12:
            raise ValueError("Spreading factor must be 5-12")

        # Validate bandwidth
        valid_bandwidths = [7800, 10400, 15600, 20800, 31250, 41700, 62500, 125000, 250000, 500000]
        if bw <= 0:
            raise ValueError(f"Bandwidth must be positive, got {bw}")

        self._sf = sf
        self._bw = bw
        self._sample_rate = sample_rate or (2 * bw)

        # Validate sample rate
        if self._sample_rate < bw:
            raise ValueError(f"Sample rate ({self._sample_rate} Hz) must be >= bandwidth ({bw} Hz)")

        # Derived parameters
        self._num_chips = 2**sf
        self._symbol_time = self._num_chips / bw
        self._samples_per_symbol = int(self._symbol_time * self._sample_rate)

        # Validate samples per symbol doesn't exceed memory limits
        if self._samples_per_symbol > self.MAX_SAMPLES_PER_SYMBOL:
            raise ValueError(
                f"Configuration would require {self._samples_per_symbol} samples per symbol, "
                f"exceeding maximum of {self.MAX_SAMPLES_PER_SYMBOL}. "
                f"Reduce sample rate or increase bandwidth."
            )

        if self._samples_per_symbol < 1:
            raise ValueError(
                f"Sample rate too low: need at least 1 sample per symbol, got {self._samples_per_symbol}"
            )

        # Generate reference chirps
        self._down_chirp = self._generate_chirp(down=True)
        self._up_chirp = self._generate_chirp(down=False)

    def _generate_chirp(self, down: bool = True) -> cp.ndarray:
        """
        Generate reference chirp signal.

        LoRa chirp: f(t) = f0 + (BW/T_sym) × t  (up-chirp)
        Down-chirp is conjugate of up-chirp.
        """
        t = cp.arange(self._samples_per_symbol, dtype=cp.float64) / self._sample_rate

        # Chirp parameters
        f0 = -self._bw / 2  # Start frequency
        chirp_rate = self._bw / self._symbol_time

        # Generate chirp phase
        phase = 2 * cp.pi * (f0 * t + 0.5 * chirp_rate * t**2)

        if down:
            return cp.exp(-1j * phase).astype(cp.complex64)
        else:
            return cp.exp(1j * phase).astype(cp.complex64)

    def detect_preamble(
        self, signal: cp.ndarray, threshold: float = 0.7, num_preamble_symbols: int = 8
    ) -> list[int]:
        """
        Detect LoRa preamble (sequence of up-chirps) - GPU-accelerated.

        Uses vectorized local maximum detection on GPU to avoid
        CPU fallback for peak finding.

        Args:
            signal: Input signal.
            threshold: Correlation threshold (0-1).
            num_preamble_symbols: Expected preamble length.

        Returns:
            List of detected preamble start indices.
        """
        signal = cp.asarray(signal, copy=False)

        # Correlate with up-chirp (GPU)
        correlation = self._correlate_chirp(signal, self._up_chirp)

        # Find peaks - all on GPU
        min_spacing = self._samples_per_symbol * (num_preamble_symbols + 2)

        # Compute correlation magnitude (GPU)
        corr_mag = cp.abs(correlation)
        thresh_value = float(cp.max(corr_mag)) * threshold

        # Vectorized local maximum detection on GPU
        # A point is local max if it's >= both neighbors
        n = len(corr_mag)
        if n < 3:
            return []

        # Create shifted views for neighbor comparison (GPU-side)
        left = corr_mag[:-2]  # i-1 values
        center = corr_mag[1:-1]  # i values
        right = corr_mag[2:]  # i+1 values

        # Local maximum mask: center >= left AND center >= right AND center > threshold
        local_max_mask = (center >= left) & (center >= right) & (center > thresh_value)

        # Get indices where local maxima occur (adjust for the offset from slicing)
        peak_indices = cp.where(local_max_mask)[0] + 1  # +1 to account for [1:-1] slice

        # Transfer just the indices to CPU (small array)
        if CUPY_AVAILABLE:
            peak_indices_np = cp.asnumpy(peak_indices)
        else:
            peak_indices_np = peak_indices

        # Apply minimum spacing filter (on CPU since it's sequential decision)
        detections = []
        for idx in peak_indices_np:
            if not detections or (idx - detections[-1]) > min_spacing:
                detections.append(int(idx))

        return detections

    def _correlate_chirp(self, signal: cp.ndarray, chirp: cp.ndarray) -> cp.ndarray:
        """Cross-correlate signal with chirp reference."""
        # Use FFT-based correlation
        n = len(signal) + len(chirp) - 1
        n_fft = 1 << (n - 1).bit_length()

        signal_fft = cp.fft.fft(signal, n_fft)
        chirp_fft = cp.fft.fft(chirp, n_fft)

        correlation = cp.fft.ifft(signal_fft * cp.conj(chirp_fft))

        return correlation[: len(signal)]

    def demodulate_symbols(self, signal: cp.ndarray, start_offset: int = 0) -> cp.ndarray:
        """
        Demodulate LoRa symbols from signal using batched FFT.

        Args:
            signal: Input signal (should start at symbol boundary).
            start_offset: Sample offset to start of first symbol.

        Returns:
            Array of demodulated symbol values (0 to 2^SF - 1).
        """
        signal = cp.asarray(signal)

        # Calculate number of complete symbols
        effective_length = len(signal) - start_offset
        num_symbols = effective_length // self._samples_per_symbol

        if num_symbols < 1:
            return cp.array([], dtype=cp.int32)

        # Reshape signal into 2D array of symbols (vectorized)
        # Extract all symbol samples at once
        symbol_length = self._samples_per_symbol
        symbol_data = cp.zeros((num_symbols, symbol_length), dtype=cp.complex64)

        for i in range(num_symbols):
            start = start_offset + i * symbol_length
            symbol_data[i, :] = signal[start : start + symbol_length]

        # Batched de-chirp (multiply each row by down_chirp)
        dechirped = symbol_data * self._down_chirp[cp.newaxis, :]

        # Batched FFT along axis 1 (each symbol independently)
        spectra = cp.fft.fft(dechirped, axis=1)

        # Find peak bin for each symbol (vectorized)
        # Only look at first num_chips bins
        spectra_mag = cp.abs(spectra[:, : self._num_chips])
        symbols = cp.argmax(spectra_mag, axis=1).astype(cp.int32)

        return symbols

    def decode_frame(self, signal: cp.ndarray, start_idx: int = 0) -> LoRaFrame | None:
        """
        Decode a complete LoRa frame.

        Args:
            signal: Input signal.
            start_idx: Start index of preamble.

        Returns:
            LoRaFrame if successful, None otherwise.
        """
        signal = cp.asarray(signal)

        # Skip preamble (8 up-chirps)
        payload_start = start_idx + 8 * self._samples_per_symbol

        # Skip sync word (2 symbols) and SFD (2.25 down-chirps)
        payload_start += int(4.25 * self._samples_per_symbol)

        # Demodulate payload
        symbols = self.demodulate_symbols(signal, payload_start)

        if len(symbols) < 1:
            return None

        # Estimate SNR from de-chirped spectrum
        sample = signal[payload_start : payload_start + self._samples_per_symbol]
        dechirped = sample * self._down_chirp
        spectrum = cp.abs(cp.fft.fft(dechirped))

        # Calculate SNR safely, handling edge cases
        signal_bins = spectrum[: self._num_chips]
        noise_bins = spectrum[self._num_chips :]

        # Protect against empty arrays and zero values
        if len(signal_bins) > 0:
            peak_power = float(cp.max(signal_bins))
        else:
            peak_power = 0.0

        if len(noise_bins) > 0:
            noise_power = float(cp.mean(noise_bins))
        else:
            # Fallback: estimate noise from lower portion of signal bins
            noise_power = float(cp.mean(signal_bins)) * 0.1 if len(signal_bins) > 0 else 1e-12

        # Protect against log(0) and division by zero
        if peak_power <= 0:
            snr_db = -np.inf
        else:
            snr_db = 10 * np.log10(peak_power / max(noise_power, 1e-12))

        # Convert symbols to NumPy
        if CUPY_AVAILABLE:
            symbols_np = cp.asnumpy(symbols)
        else:
            symbols_np = symbols

        return LoRaFrame(
            symbols=symbols_np,
            sf=self._sf,
            bw=self._bw,
            sync_word=0x12,  # Default public sync word
            snr_estimate=snr_db,
            preamble_start=start_idx,
        )

    def estimate_frequency_offset(self, signal: cp.ndarray) -> float:
        """
        Estimate carrier frequency offset from preamble.

        Args:
            signal: Signal containing preamble.

        Returns:
            Estimated frequency offset in Hz.
        """
        signal = cp.asarray(signal)

        # De-chirp one symbol
        if len(signal) < self._samples_per_symbol:
            return 0.0

        sample = signal[: self._samples_per_symbol]
        dechirped = sample * self._down_chirp

        # High-resolution FFT
        n_fft = self._samples_per_symbol * 4
        spectrum = cp.fft.fft(dechirped, n_fft)
        peak_bin = int(cp.argmax(cp.abs(spectrum)))

        # Convert bin to frequency
        if peak_bin > n_fft // 2:
            peak_bin -= n_fft

        freq_offset = peak_bin * self._sample_rate / n_fft

        return float(freq_offset)

    @property
    def spreading_factor(self) -> int:
        return self._sf

    @property
    def bandwidth(self) -> float:
        return self._bw

    @property
    def samples_per_symbol(self) -> int:
        return self._samples_per_symbol

    @property
    def symbol_time(self) -> float:
        return self._symbol_time
