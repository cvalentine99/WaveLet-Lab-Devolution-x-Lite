"""
GPU RF Forensics Engine - CFAR Detection

GPU-optimized Cell-Averaging CFAR (Constant False Alarm Rate) detector
with custom Numba CUDA kernels using shared memory optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rf_forensics.core.gpu_compat import CUPY_AVAILABLE, cp, np, to_numpy

# Import Numba CUDA for optimized kernels
try:
    import numba
    from numba import cuda

    CUDA_AVAILABLE = CUPY_AVAILABLE  # Both CuPy and Numba CUDA needed
except ImportError:
    CUDA_AVAILABLE = False
    cuda = None
    numba = None


@dataclass
class CFARResult:
    """Result of CFAR detection."""

    detection_mask: np.ndarray  # Boolean mask of detections
    snr_estimate: np.ndarray  # Estimated SNR for each bin
    threshold_map: np.ndarray  # Adaptive threshold at each bin
    num_detections: int


class CFARDetector:
    """
    GPU-accelerated CFAR (Constant False Alarm Rate) detector.

    Implements multiple CFAR variants optimized for GPU:
    - CA-CFAR: Cell Averaging (standard)
    - GO-CFAR: Greatest Of (best for edges)
    - SO-CFAR: Smallest Of (best for multiple targets)
    - OS-CFAR: Order Statistic (robust to interference)

    Uses custom Numba CUDA kernels with shared memory optimization
    for maximum performance on RTX 4090.
    """

    def __init__(
        self,
        num_reference: int = 32,
        num_guard: int = 4,
        pfa: float = 1e-6,
        variant: Literal["CA", "GO", "SO", "OS"] = "CA",
    ):
        """
        Initialize CFAR detector.

        Args:
            num_reference: Number of reference (training) cells on each side.
            num_guard: Number of guard cells on each side.
            pfa: Probability of false alarm.
            variant: CFAR variant to use.
        """
        self._num_reference = num_reference
        self._num_guard = num_guard
        self._pfa = pfa
        self._variant = variant

        # Calculate alpha (scaling factor) from Pfa
        # For CA-CFAR: α = N × (Pfa^(-1/N) - 1) where N = 2 × num_reference
        n_cells = 2 * num_reference
        self._alpha = n_cells * (pfa ** (-1.0 / n_cells) - 1)

        # Total window size
        self._window_size = 2 * (num_reference + num_guard) + 1

        # Pre-allocated output buffers - reused each frame to avoid allocation
        self._detection_mask = None
        self._snr_out = None
        self._threshold_out = None
        self._last_n_bins = 0

        # Pre-allocated kernel for CuPy fallback path
        # (CUDA kernel path uses shared memory, no kernel array needed)
        window_half = num_reference + num_guard
        kernel_size = 2 * window_half + 1
        self._kernel = cp.ones(kernel_size, dtype=cp.float32)
        # Zero out center (CUT) and guard cells
        guard_start = window_half - num_guard
        guard_end = window_half + num_guard + 1
        self._kernel[guard_start:guard_end] = 0
        # Normalize by number of reference cells
        num_ref_cells = 2 * num_reference
        self._kernel /= num_ref_cells

        # Pre-allocated padding buffer for _cfar_cupy
        self._psd_padded = None
        self._padded_size = 0

        # Compile CUDA kernels if available
        if CUDA_AVAILABLE:
            self._compile_kernels()

    def _compile_kernels(self):
        """Compile optimized CUDA kernels."""

        # CFAR kernel using shared memory
        @cuda.jit
        def ca_cfar_kernel(
            psd, detection_mask, snr_out, threshold_out, num_ref, num_guard, alpha, n_bins
        ):
            """
            CA-CFAR kernel with shared memory optimization.

            Shared memory layout:
            [guard_left][reference_left][CUT][reference_right][guard_right]

            Each thread processes one Cell Under Test (CUT).
            """
            # Shared memory for PSD data (declared dynamically)
            shared = cuda.shared.array(shape=0, dtype=numba.float32)

            # Thread and block indices
            tx = cuda.threadIdx.x
            bx = cuda.blockIdx.x
            block_size = cuda.blockDim.x

            # Global index (CUT position)
            gid = bx * block_size + tx

            # Window parameters
            window_half = num_ref + num_guard
            window_size = 2 * window_half + 1

            # Calculate shared memory load indices
            # Each thread loads one element plus halo regions
            local_start = bx * block_size - window_half
            shared_size = block_size + 2 * window_half

            # Cooperative loading into shared memory
            for i in range(tx, shared_size, block_size):
                global_idx = local_start + i
                if 0 <= global_idx < n_bins:
                    shared[i] = psd[global_idx]
                else:
                    shared[i] = 0.0

            # Synchronize after loading
            cuda.syncthreads()

            # Process if within bounds
            if gid < n_bins:
                # Local index in shared memory
                local_idx = tx + window_half

                # Get CUT power
                cut_power = shared[local_idx]

                # Sum reference cells (excluding guard cells)
                noise_sum = 0.0
                count = 0

                # Left reference cells
                for i in range(num_ref):
                    ref_idx = local_idx - window_half + i
                    if ref_idx >= 0:
                        noise_sum += shared[ref_idx]
                        count += 1

                # Right reference cells
                for i in range(num_ref):
                    ref_idx = local_idx + num_guard + 1 + i
                    if ref_idx < shared_size:
                        noise_sum += shared[ref_idx]
                        count += 1

                # Calculate threshold
                if count > 0:
                    noise_estimate = noise_sum / count
                    threshold = alpha * noise_estimate
                else:
                    noise_estimate = 0.0
                    threshold = 0.0

                # Detection decision
                detection_mask[gid] = cut_power > threshold

                # SNR estimate (in linear scale)
                if noise_estimate > 0:
                    snr_out[gid] = cut_power / noise_estimate
                else:
                    snr_out[gid] = 0.0

                threshold_out[gid] = threshold

        self._ca_cfar_kernel = ca_cfar_kernel

    def detect(self, psd: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Perform CFAR detection on 1D PSD.

        Uses pre-allocated output buffers to avoid per-frame GPU allocation.

        Args:
            psd: Power spectral density array (linear scale).

        Returns:
            Tuple of (detection_mask, snr_estimate).
        """
        # Ensure psd is a cupy array on GPU (handle numpy input gracefully)
        if CUPY_AVAILABLE:
            if not hasattr(psd, "__cuda_array_interface__"):
                # Input is numpy - convert to cupy
                psd = cp.asarray(psd, dtype=cp.float32)
            elif psd.dtype != cp.float32:
                psd = psd.astype(cp.float32)
        else:
            psd = np.asarray(psd, dtype=np.float32)

        n_bins = len(psd)

        # Lazy initialization / resize of pre-allocated output buffers
        if self._detection_mask is None or self._last_n_bins != n_bins:
            self._detection_mask = cp.zeros(n_bins, dtype=cp.bool_)
            self._snr_out = cp.zeros(n_bins, dtype=cp.float32)
            self._threshold_out = cp.zeros(n_bins, dtype=cp.float32)
            self._last_n_bins = n_bins
        else:
            # Reuse existing buffers - just reset values
            self._detection_mask.fill(False)
            self._snr_out.fill(0)
            self._threshold_out.fill(0)

        if CUDA_AVAILABLE and self._variant == "CA":
            # Use optimized CUDA kernel
            threads_per_block = 256
            blocks = (n_bins + threads_per_block - 1) // threads_per_block

            # Calculate shared memory size
            shared_size = threads_per_block + 2 * (self._num_reference + self._num_guard)
            shared_mem_bytes = shared_size * 4  # float32 = 4 bytes

            # Convert cupy arrays to numba device arrays for kernel
            psd_device = cuda.as_cuda_array(psd)
            mask_device = cuda.as_cuda_array(self._detection_mask)
            snr_device = cuda.as_cuda_array(self._snr_out)
            thresh_device = cuda.as_cuda_array(self._threshold_out)

            self._ca_cfar_kernel[blocks, threads_per_block, 0, shared_mem_bytes](
                psd_device,
                mask_device,
                snr_device,
                thresh_device,
                self._num_reference,
                self._num_guard,
                self._alpha,
                n_bins,
            )
        else:
            # Fallback to CuPy implementation (updates pre-allocated buffers in place)
            self._cfar_cupy(psd)

        return self._detection_mask, self._snr_out

    def _cfar_cupy(self, psd: cp.ndarray) -> None:
        """
        Vectorized CuPy-based CFAR implementation using convolution.

        Uses pre-allocated kernel and writes directly to pre-allocated output buffers.
        """
        n_bins = len(psd)
        window_half = self._num_reference + self._num_guard

        # Allocate/resize padded buffer if needed
        required_padded_size = n_bins + 2 * window_half
        if self._psd_padded is None or self._padded_size != required_padded_size:
            self._psd_padded = cp.zeros(required_padded_size, dtype=cp.float32)
            self._padded_size = required_padded_size

        # Manual edge padding into pre-allocated buffer (avoids cp.pad allocation)
        self._psd_padded[window_half : window_half + n_bins] = psd
        self._psd_padded[:window_half] = psd[0]  # Left edge
        self._psd_padded[window_half + n_bins :] = psd[-1]  # Right edge

        if self._variant == "CA":
            # Cell Averaging: convolve for noise estimate using pre-allocated kernel
            noise_estimate = cp.convolve(self._psd_padded, self._kernel, mode="valid")

        elif self._variant in ("GO", "SO"):
            # Greatest Of / Smallest Of: need separate left/right windows
            kernel_size = 2 * window_half + 1
            left_kernel = cp.zeros(kernel_size, dtype=cp.float32)
            left_kernel[: self._num_reference] = 1.0 / self._num_reference

            right_kernel = cp.zeros(kernel_size, dtype=cp.float32)
            right_kernel[kernel_size - self._num_reference :] = 1.0 / self._num_reference

            left_avg = cp.convolve(self._psd_padded, left_kernel, mode="valid")
            right_avg = cp.convolve(self._psd_padded, right_kernel, mode="valid")

            if self._variant == "GO":
                noise_estimate = cp.maximum(left_avg, right_avg)
            else:  # SO
                noise_estimate = cp.minimum(left_avg, right_avg)

        else:  # OS - Order Statistic (median) - still needs loop but vectorized where possible
            noise_estimate = cp.zeros(n_bins, dtype=cp.float32)
            for i in range(n_bins):
                center = i + window_half
                left_ref = self._psd_padded[center - window_half : center - self._num_guard]
                right_ref = self._psd_padded[
                    center + self._num_guard + 1 : center + window_half + 1
                ]
                all_ref = cp.concatenate([left_ref, right_ref])
                noise_estimate[i] = cp.median(all_ref)

        # NaN protection: replace any NaN values in noise estimate with fallback
        if cp.any(cp.isnan(noise_estimate)):
            # Use median of valid values as fallback
            valid_mask = ~cp.isnan(noise_estimate)
            if cp.any(valid_mask):
                fallback = cp.median(noise_estimate[valid_mask])
            else:
                fallback = 1e-12
            noise_estimate = cp.nan_to_num(noise_estimate, nan=float(fallback))

        # Compute threshold and detection, writing to pre-allocated buffers
        threshold = self._alpha * noise_estimate

        # Write results to pre-allocated buffers
        self._threshold_out[:] = threshold
        self._detection_mask[:] = psd > threshold
        # SNR calculation with zero-division protection
        # CuPy doesn't support 'where' parameter, use safe division instead
        safe_noise = cp.maximum(noise_estimate, 1e-12)
        cp.divide(psd, safe_noise, out=self._snr_out)

    def detect_2d(self, spectrogram: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Perform 2D CFAR detection on spectrogram (time-frequency).

        Applies 1D CFAR along frequency axis for each time bin.
        Vectorized implementation processes all time bins in a single batched operation.

        Args:
            spectrogram: 2D array with shape (time_bins, freq_bins).

        Returns:
            Tuple of (detection_mask, snr_estimate) with same shape.
        """
        spectrogram = cp.asarray(spectrogram, dtype=cp.float32)
        n_time, n_freq = spectrogram.shape

        # Use batched 1D CFAR for CA variant (most common)
        if self._variant == "CA":
            return self._detect_2d_batched_ca(spectrogram)

        # Fallback to sequential for other variants
        detection_mask = cp.zeros((n_time, n_freq), dtype=cp.bool_)
        snr_out = cp.zeros((n_time, n_freq), dtype=cp.float32)

        for t in range(n_time):
            det, snr = self.detect(spectrogram[t])
            detection_mask[t] = det
            snr_out[t] = snr

        return detection_mask, snr_out

    def _detect_2d_batched_ca(self, spectrogram: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Batched CA-CFAR along frequency axis for all time bins simultaneously.

        Uses 2D convolution with a 1D kernel to process all rows at once.
        """
        n_time, n_freq = spectrogram.shape
        window_half = self._num_reference + self._num_guard

        # Create 1D reference cell kernel with zeros in guard region
        kernel_size = 2 * window_half + 1
        kernel_1d = cp.ones(kernel_size, dtype=cp.float32)

        # Zero out center (CUT) and guard cells
        guard_start = window_half - self._num_guard
        guard_end = window_half + self._num_guard + 1
        kernel_1d[guard_start:guard_end] = 0

        # Normalize by number of reference cells
        num_ref_cells = 2 * self._num_reference
        kernel_1d /= num_ref_cells

        # Reshape to 2D kernel (1 x kernel_size) for batched 1D convolution
        kernel_2d = kernel_1d.reshape(1, -1)

        # Pad along frequency axis only
        spec_padded = cp.pad(spectrogram, ((0, 0), (window_half, window_half)), mode="edge")

        # 2D convolution with 1×N kernel = batched 1D convolution along rows
        # This processes ALL time bins simultaneously
        noise_estimate = self._convolve2d_separable(spec_padded, kernel_2d)

        # Trim to original size
        noise_estimate = noise_estimate[:, :n_freq]

        # NaN protection for 2D
        if cp.any(cp.isnan(noise_estimate)):
            valid_mask = ~cp.isnan(noise_estimate)
            if cp.any(valid_mask):
                fallback = float(cp.median(noise_estimate[valid_mask]))
            else:
                fallback = 1e-12
            noise_estimate = cp.nan_to_num(noise_estimate, nan=fallback)

        # Threshold and detection (all vectorized)
        threshold = self._alpha * noise_estimate
        detection_mask = spectrogram > threshold

        # SNR calculation with zero-division protection
        safe_noise = cp.maximum(noise_estimate, 1e-12)
        snr_out = spectrogram / safe_noise

        return detection_mask, snr_out

    def _convolve2d_separable(self, signal: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
        """
        2D convolution optimized for separable/1D kernels.

        For a 1×N kernel, this is equivalent to batched 1D convolution along rows.
        """
        # For small kernels, direct convolution is efficient
        # Use FFT for larger kernels
        if kernel.shape[1] < 64:
            # Direct convolution along rows
            result = cp.zeros_like(signal[:, : signal.shape[1] - kernel.shape[1] + 1])
            kernel_1d = kernel.ravel()
            for i in range(signal.shape[0]):
                result[i] = cp.convolve(signal[i], kernel_1d, mode="valid")
            return result
        else:
            # FFT-based for large kernels
            n_fft = 1 << (signal.shape[1] + kernel.shape[1] - 2).bit_length()

            # Batch FFT along rows
            signal_fft = cp.fft.fft(signal, n=n_fft, axis=1)
            kernel_fft = cp.fft.fft(kernel.ravel(), n=n_fft)

            # Multiply and inverse FFT
            result = cp.fft.ifft(signal_fft * kernel_fft, axis=1).real

            # Return valid portion
            return result[:, : signal.shape[1] - kernel.shape[1] + 1]

    def get_threshold_map(self, psd: cp.ndarray) -> cp.ndarray:
        """
        Compute adaptive threshold map without detection.

        Args:
            psd: Power spectral density array.

        Returns:
            Threshold array.
        """
        _, snr = self.detect(psd)
        # Reconstruct threshold from SNR
        return psd / (snr + 1e-12)

    def get_result(self, psd: cp.ndarray) -> CFARResult:
        """
        Perform detection and return structured result.

        Args:
            psd: Power spectral density array.

        Returns:
            CFARResult with detection information.
        """
        detection_mask, snr_estimate = self.detect(psd)
        threshold_map = self.get_threshold_map(psd)

        # Convert to NumPy for result
        det_np = to_numpy(detection_mask)
        snr_np = to_numpy(snr_estimate)
        thresh_np = to_numpy(threshold_map)

        return CFARResult(
            detection_mask=det_np,
            snr_estimate=snr_np,
            threshold_map=thresh_np,
            num_detections=int(det_np.sum()),
        )

    @property
    def alpha(self) -> float:
        """CFAR scaling factor."""
        return self._alpha

    @property
    def pfa(self) -> float:
        """Probability of false alarm."""
        return self._pfa

    def set_pfa(self, pfa: float) -> None:
        """Update probability of false alarm and recalculate alpha."""
        self._pfa = pfa
        n_cells = 2 * self._num_reference
        self._alpha = n_cells * (pfa ** (-1.0 / n_cells) - 1)


class CFARDetector2D(CFARDetector):
    """
    True 2D CFAR detector using square reference window.

    For time-frequency detection where targets span both dimensions.
    """

    def __init__(self, num_reference: int = 8, num_guard: int = 2, pfa: float = 1e-6):
        """Initialize 2D CFAR detector."""
        super().__init__(num_reference, num_guard, pfa, "CA")

        # 2D has more reference cells
        n_cells = (2 * num_reference + 2 * num_guard + 1) ** 2 - (2 * num_guard + 1) ** 2
        self._alpha = n_cells * (pfa ** (-1.0 / n_cells) - 1)

    def detect_2d(self, spectrogram: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        """
        True 2D CFAR detection (vectorized using 2D convolution).

        Uses 2D convolution to compute window sums for all cells simultaneously,
        eliminating the O(n²) nested Python loop.

        Args:
            spectrogram: 2D array (time x frequency).

        Returns:
            Tuple of (detection_mask, snr_estimate).
        """
        spectrogram = cp.asarray(spectrogram, dtype=cp.float32)
        n_time, n_freq = spectrogram.shape

        window_half = self._num_reference + self._num_guard
        guard_half = self._num_guard

        # Create 2D kernels for convolution
        window_size = 2 * window_half + 1
        guard_size = 2 * guard_half + 1

        # Window kernel: all ones
        window_kernel = cp.ones((window_size, window_size), dtype=cp.float32)

        # Guard kernel: ones in center (will be subtracted)
        guard_kernel = cp.zeros((window_size, window_size), dtype=cp.float32)
        # Place guard region in center of window-sized kernel
        guard_start = window_half - guard_half
        guard_end = window_half + guard_half + 1
        guard_kernel[guard_start:guard_end, guard_start:guard_end] = 1.0

        # Pad spectrogram for 'same' output size
        spec_padded = cp.pad(spectrogram, window_half, mode="edge")

        # 2D convolution for window sums (using FFT-based approach for efficiency)
        # This computes sum of all elements in window around each cell
        window_sums = self._convolve2d_fft(spec_padded, window_kernel)

        # Guard region sums
        guard_sums = self._convolve2d_fft(spec_padded, guard_kernel)

        # Reference cells = window - guard
        # Trim to original size
        ref_sums = (
            window_sums[window_half : window_half + n_time, window_half : window_half + n_freq]
            - guard_sums[window_half : window_half + n_time, window_half : window_half + n_freq]
        )

        # Number of reference cells
        ref_count = window_size * window_size - guard_size * guard_size

        # Noise estimate (vectorized)
        noise_estimate = ref_sums / ref_count

        # Threshold and detection (all vectorized)
        threshold = self._alpha * noise_estimate
        detection_mask = spectrogram > threshold

        # SNR calculation with protection against division by zero
        snr_out = cp.where(noise_estimate > 0, spectrogram / noise_estimate, 0.0)

        return detection_mask, snr_out

    def _convolve2d_fft(self, signal: cp.ndarray, kernel: cp.ndarray) -> cp.ndarray:
        """
        FFT-based 2D convolution for efficiency on large arrays.

        Args:
            signal: Input 2D array.
            kernel: Convolution kernel.

        Returns:
            Convolution result (full size).
        """
        # Compute output size
        s1 = signal.shape
        s2 = kernel.shape
        out_shape = (s1[0] + s2[0] - 1, s1[1] + s2[1] - 1)

        # FFT size (power of 2 for efficiency)
        fft_shape = (1 << (out_shape[0] - 1).bit_length(), 1 << (out_shape[1] - 1).bit_length())

        # FFT-based convolution
        signal_fft = cp.fft.fft2(signal, s=fft_shape)
        kernel_fft = cp.fft.fft2(kernel, s=fft_shape)

        result = cp.fft.ifft2(signal_fft * kernel_fft).real

        # Return valid portion
        return result[: out_shape[0], : out_shape[1]]
