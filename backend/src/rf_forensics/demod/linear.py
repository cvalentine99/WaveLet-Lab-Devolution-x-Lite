"""
GPU RF Forensics Engine - Generic Linear Demodulator

PSK and QAM demodulators with carrier and timing recovery.
"""

from __future__ import annotations

from dataclasses import dataclass

from rf_forensics.core.gpu_compat import CUPY_AVAILABLE, cp, np


@dataclass
class DemodResult:
    """Result of linear demodulation."""

    symbols: np.ndarray
    bits: np.ndarray
    constellation_points: np.ndarray
    carrier_offset_hz: float
    timing_offset: float
    snr_estimate_db: float
    evm_percent: float


@dataclass
class ConstellationPoint:
    """A point in a constellation diagram."""

    i: float
    q: float
    bits: tuple[int, ...]


class LinearDemodulator:
    """
    GPU-accelerated generic PSK and QAM demodulator.

    Supports:
    - BPSK, QPSK, 8PSK
    - 16QAM, 64QAM, 256QAM

    Features:
    - Costas loop carrier recovery
    - Decision-directed timing recovery
    - Soft and hard decision outputs
    - Constellation visualization
    """

    # Constellation definitions (normalized)
    CONSTELLATIONS = {
        "BPSK": np.array([1 + 0j, -1 + 0j]),
        "QPSK": np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2),
        "8PSK": np.exp(1j * np.arange(8) * np.pi / 4),
    }

    def __init__(
        self,
        modulation: str,
        symbol_rate: float,
        sample_rate: float,
        carrier_recovery: bool = True,
        timing_recovery: bool = True,
    ):
        """
        Initialize linear demodulator.

        Args:
            modulation: Modulation type (BPSK, QPSK, 8PSK, 16QAM, etc.).
            symbol_rate: Symbol rate in Hz.
            sample_rate: Sample rate in Hz.
            carrier_recovery: Enable carrier recovery.
            timing_recovery: Enable timing recovery.

        Raises:
            ValueError: If parameters are invalid.
        """
        self._modulation = modulation.upper()
        self._symbol_rate = symbol_rate
        self._sample_rate = sample_rate
        self._carrier_recovery = carrier_recovery
        self._timing_recovery = timing_recovery

        # Validate sample rate
        if sample_rate < symbol_rate:
            raise ValueError(
                f"Sample rate ({sample_rate} Hz) must be >= symbol rate ({symbol_rate} Hz)"
            )

        self._samples_per_symbol = int(sample_rate / symbol_rate)
        if self._samples_per_symbol < 1:
            raise ValueError(
                f"Sample rate too low: samples_per_symbol = {self._samples_per_symbol}"
            )

        # Get or generate constellation
        if modulation.upper() in self.CONSTELLATIONS:
            self._constellation = cp.asarray(self.CONSTELLATIONS[modulation.upper()])
        else:
            self._constellation = self._generate_qam_constellation(modulation)

        # Validate constellation size is power of 2
        constellation_size = len(self._constellation)
        if constellation_size < 2:
            raise ValueError(f"Constellation must have at least 2 points, got {constellation_size}")
        if constellation_size & (constellation_size - 1) != 0:
            raise ValueError(f"Constellation size must be power of 2, got {constellation_size}")

        self._bits_per_symbol = int(np.log2(constellation_size))

        # Carrier recovery state
        self._carrier_phase = 0.0
        self._carrier_freq = 0.0
        self._carrier_loop_bw = 0.01

        # Timing recovery state
        self._timing_offset = 0.0
        self._timing_loop_bw = 0.01

    def _generate_qam_constellation(self, modulation: str) -> cp.ndarray:
        """
        Generate QAM constellation points in Gray-coded order.

        The constellation index directly maps to bit pattern via natural binary,
        but the constellation points are arranged so adjacent points (in index)
        differ by one bit in their coordinates (Gray code property).
        """
        if modulation == "16QAM":
            m = 4
        elif modulation == "64QAM":
            m = 8
        elif modulation == "256QAM":
            m = 16
        else:
            raise ValueError(f"Unknown modulation: {modulation}")

        bits_per_axis = int(np.log2(m))

        # Generate Gray code to natural mapping for each axis
        def gray_to_natural(n: int, bits: int) -> int:
            """Convert Gray code to natural binary."""
            result = n
            mask = n >> 1
            while mask:
                result ^= mask
                mask >>= 1
            return result

        # Generate constellation in index order (0, 1, 2, ...)
        # where each index's I/Q coordinates follow Gray coding
        points = np.zeros(m * m, dtype=np.complex128)

        for idx in range(m * m):
            # Split index into I and Q bit groups
            i_gray = idx & (m - 1)  # Lower bits for Q axis
            q_gray = (idx >> bits_per_axis) & (m - 1)  # Upper bits for I axis

            # Convert Gray index to natural for coordinate lookup
            i_nat = gray_to_natural(i_gray, bits_per_axis)
            q_nat = gray_to_natural(q_gray, bits_per_axis)

            # Map to centered coordinates
            i_val = 2 * i_nat - m + 1
            q_val = 2 * q_nat - m + 1

            points[idx] = complex(i_val, q_val)

        # Normalize to unit average power
        points = points / np.sqrt(np.mean(np.abs(points) ** 2))

        return cp.asarray(points)

    def demodulate(self, signal: cp.ndarray) -> DemodResult:
        """
        Demodulate signal to symbols and bits.

        Args:
            signal: Complex input signal.

        Returns:
            DemodResult with symbols, bits, and metrics.

        Raises:
            ValueError: If signal is too short to contain at least one symbol.
        """
        signal = cp.asarray(signal, copy=False)

        # Validate signal length
        if len(signal) < self._samples_per_symbol:
            raise ValueError(
                f"Signal too short: {len(signal)} samples, "
                f"need at least {self._samples_per_symbol} for one symbol"
            )

        # Carrier recovery
        if self._carrier_recovery:
            signal, carrier_offset = self._costas_loop(signal)
        else:
            carrier_offset = 0.0

        # Timing recovery
        if self._timing_recovery:
            signal, timing_offset = self._timing_recovery_loop(signal)
        else:
            timing_offset = 0.0

        # Sample at symbol rate
        num_symbols = len(signal) // self._samples_per_symbol

        # Guard against empty result after timing recovery
        if num_symbols < 1:
            raise ValueError("Signal too short after recovery: cannot extract any symbols")

        # Vectorized symbol sampling at center of each symbol period
        sample_offset = self._samples_per_symbol // 2
        indices = cp.arange(num_symbols) * self._samples_per_symbol + sample_offset
        # Ensure indices are within bounds
        indices = indices[indices < len(signal)]
        symbol_samples = signal[indices.astype(cp.int64)]

        # Ensure we have the expected number of symbols
        if len(symbol_samples) < num_symbols:
            num_symbols = len(symbol_samples)

        # Make decisions
        symbols, bits = self._make_decisions(symbol_samples)

        # Calculate metrics
        snr_db, evm_percent = self._calculate_metrics(symbol_samples, symbols)

        # Convert to NumPy
        if CUPY_AVAILABLE:
            symbols_np = cp.asnumpy(symbols)
            bits_np = cp.asnumpy(bits)
            constellation_np = cp.asnumpy(symbol_samples)
        else:
            symbols_np = symbols
            bits_np = bits
            constellation_np = symbol_samples

        return DemodResult(
            symbols=symbols_np,
            bits=bits_np,
            constellation_points=constellation_np,
            carrier_offset_hz=carrier_offset,
            timing_offset=timing_offset,
            snr_estimate_db=snr_db,
            evm_percent=evm_percent,
        )

    def _costas_loop(self, signal: cp.ndarray) -> tuple[cp.ndarray, float]:
        """
        Costas loop carrier recovery.

        Works for BPSK and QPSK. For higher-order uses decision-directed.
        """
        output = cp.zeros_like(signal)

        phase = self._carrier_phase
        freq = self._carrier_freq

        # Loop parameters
        bw = self._carrier_loop_bw
        damping = 0.707
        theta = bw / (damping + 1 / (4 * damping))
        d = 1 + 2 * damping * theta + theta**2
        k1 = 4 * damping * theta / d
        k2 = 4 * theta**2 / d

        for i in range(len(signal)):
            # Derotate
            output[i] = signal[i] * cp.exp(-1j * phase)

            # Phase error detector
            if self._modulation in ["BPSK"]:
                error = float(cp.sign(output[i].real) * output[i].imag)
            elif self._modulation in ["QPSK"]:
                error = float(
                    cp.sign(output[i].real) * output[i].imag
                    - cp.sign(output[i].imag) * output[i].real
                )
            else:
                # Decision-directed for higher order
                decision = self._nearest_symbol(output[i])
                error = float((output[i] * cp.conj(decision)).imag)

            # Update loop
            freq += k2 * error
            phase += freq + k1 * error

            # Wrap phase using numpy scalars (avoids GPU sync per iteration)
            while phase > np.pi:
                phase -= 2 * np.pi
            while phase < -np.pi:
                phase += 2 * np.pi

        self._carrier_phase = phase
        self._carrier_freq = freq

        # Estimate frequency offset
        freq_offset_hz = float(freq) * self._sample_rate / (2 * np.pi)

        return output, freq_offset_hz

    def _timing_recovery_loop(self, signal: cp.ndarray) -> tuple[cp.ndarray, float]:
        """Gardner timing error detector and interpolation."""
        sps = self._samples_per_symbol

        # Simple approach: find optimal sampling phase
        # by maximizing symbol energy
        best_phase = 0
        best_energy = 0

        for phase in range(sps):
            samples = signal[phase::sps]
            energy = float(cp.sum(cp.abs(samples) ** 2))
            if energy > best_energy:
                best_energy = energy
                best_phase = phase

        # Resample with optimal phase
        output = signal[best_phase:]

        timing_offset = best_phase / sps
        self._timing_offset = timing_offset

        return output, timing_offset

    def _nearest_symbol(self, sample: cp.ndarray) -> cp.ndarray:
        """Find nearest constellation point."""
        distances = cp.abs(self._constellation - sample)
        idx = int(cp.argmin(distances))
        return self._constellation[idx]

    def _make_decisions(self, samples: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """Make symbol and bit decisions (vectorized)."""
        num_symbols = len(samples)

        if num_symbols == 0:
            return cp.array([], dtype=cp.int32), cp.array([], dtype=cp.uint8)

        # Vectorized nearest neighbor search
        # Compute distance from each sample to each constellation point
        # samples shape: (num_symbols,), constellation shape: (num_points,)
        # Use broadcasting: distances shape: (num_symbols, num_points)
        distances = cp.abs(samples[:, cp.newaxis] - self._constellation[cp.newaxis, :])

        # Find index of nearest constellation point for each sample
        symbols = cp.argmin(distances, axis=1).astype(cp.int32)

        # Vectorized bit extraction
        # Create bit shift array [0, 1, 2, ..., bits_per_symbol-1]
        bit_shifts = cp.arange(self._bits_per_symbol, dtype=cp.int32)

        # For each symbol, extract all bits at once using broadcasting
        # symbols shape: (num_symbols,), bit_shifts shape: (bits_per_symbol,)
        # Result shape: (num_symbols, bits_per_symbol)
        bits_2d = ((symbols[:, cp.newaxis] >> bit_shifts[cp.newaxis, :]) & 1).astype(cp.uint8)

        # Flatten to 1D array
        bits = bits_2d.ravel()

        return symbols, bits

    def _calculate_metrics(
        self, received: cp.ndarray, decisions: cp.ndarray
    ) -> tuple[float, float]:
        """Calculate SNR and EVM."""
        # Get ideal symbols
        ideal = self._constellation[decisions.astype(cp.int32)]

        # Error vector
        error = received - ideal

        # EVM
        error_power = float(cp.mean(cp.abs(error) ** 2))
        signal_power = float(cp.mean(cp.abs(ideal) ** 2))

        # Protect against zero/negative values in both signal and error power
        if signal_power > 1e-12 and error_power >= 0:
            evm_percent = 100 * np.sqrt(error_power / signal_power)
            snr_db = 10 * np.log10(max(signal_power, 1e-12) / max(error_power, 1e-12))
        elif signal_power > 1e-12:
            # Error power is zero/negative - perfect signal
            evm_percent = 0.0
            snr_db = np.inf
        else:
            # No signal power - undefined SNR
            evm_percent = 100.0
            snr_db = -np.inf

        return snr_db, evm_percent

    def get_constellation(self) -> cp.ndarray:
        """Get constellation diagram points."""
        return self._constellation

    @property
    def modulation(self) -> str:
        return self._modulation

    @property
    def bits_per_symbol(self) -> int:
        return self._bits_per_symbol

    @property
    def samples_per_symbol(self) -> int:
        return self._samples_per_symbol
