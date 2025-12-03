"""
GPU RF Forensics Engine - Window Function Library

GPU-optimized window function generation using CuPy with caching support.
"""

from __future__ import annotations

from dataclasses import dataclass

from rf_forensics.core.gpu_compat import cp, np, to_numpy


@dataclass
class WindowCorrectionFactors:
    """Window correction factors for spectral analysis."""

    coherent_gain: float  # Amplitude correction for coherent signals
    noise_bandwidth: float  # Noise bandwidth in bins
    enbw: float  # Equivalent Noise Bandwidth
    processing_gain_db: float  # Processing gain vs rectangular window
    max_sidelobe_db: float  # Maximum sidelobe level in dB


class WindowGenerator:
    """
    GPU-optimized window function generator with caching.

    Generates all windows directly on GPU using CuPy. Supports caching
    to avoid regeneration overhead for frequently used window sizes.

    Supported window types:
    - hann: General purpose, good frequency resolution
    - kaiser: Configurable beta parameter for sidelobe control
    - blackman: Low sidelobes, wider main lobe
    - blackman_harris: Very low sidelobes (-92 dB)
    - hamming: Similar to Hann, slightly different sidelobes
    - flattop: Accurate amplitude measurement
    - rectangular: No windowing (maximum frequency resolution)
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize window generator.

        Args:
            cache_enabled: Whether to cache generated windows.
        """
        self._cache_enabled = cache_enabled
        self._cache: dict[tuple, cp.ndarray] = {}
        self._correction_cache: dict[tuple, WindowCorrectionFactors] = {}

        # Pre-computed correction factors for common windows
        self._correction_factors = {
            "rectangular": WindowCorrectionFactors(
                coherent_gain=1.0,
                noise_bandwidth=1.0,
                enbw=1.0,
                processing_gain_db=0.0,
                max_sidelobe_db=-13.3,
            ),
            "hann": WindowCorrectionFactors(
                coherent_gain=0.5,
                noise_bandwidth=1.5,
                enbw=1.5,
                processing_gain_db=-1.76,
                max_sidelobe_db=-31.5,
            ),
            "hamming": WindowCorrectionFactors(
                coherent_gain=0.54,
                noise_bandwidth=1.36,
                enbw=1.36,
                processing_gain_db=-1.34,
                max_sidelobe_db=-42.7,
            ),
            "blackman": WindowCorrectionFactors(
                coherent_gain=0.42,
                noise_bandwidth=1.73,
                enbw=1.73,
                processing_gain_db=-2.38,
                max_sidelobe_db=-58.1,
            ),
            "blackman_harris": WindowCorrectionFactors(
                coherent_gain=0.36,
                noise_bandwidth=2.0,
                enbw=2.0,
                processing_gain_db=-3.01,
                max_sidelobe_db=-92.0,
            ),
            "flattop": WindowCorrectionFactors(
                coherent_gain=0.22,
                noise_bandwidth=3.77,
                enbw=3.77,
                processing_gain_db=-5.77,
                max_sidelobe_db=-93.0,
            ),
        }

    def get_window(self, window_type: str, size: int, **params) -> cp.ndarray:
        """
        Get a window function, using cache if available.

        Args:
            window_type: Type of window (hann, kaiser, blackman, etc.).
            size: Window length in samples.
            **params: Additional parameters (e.g., beta for Kaiser).

        Returns:
            CuPy array containing the window.
        """
        # Create cache key
        cache_key = (window_type, size, tuple(sorted(params.items())))

        # Check cache
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # Generate window
        window = self._generate_window(window_type, size, **params)

        # Cache result
        if self._cache_enabled:
            self._cache[cache_key] = window

        return window

    def _generate_window(self, window_type: str, size: int, **params) -> cp.ndarray:
        """Generate a window function on GPU."""
        n = cp.arange(size, dtype=cp.float32)

        if window_type == "rectangular":
            return cp.ones(size, dtype=cp.float32)

        elif window_type == "hann":
            return 0.5 * (1 - cp.cos(2 * cp.pi * n / (size - 1)))

        elif window_type == "hamming":
            return 0.54 - 0.46 * cp.cos(2 * cp.pi * n / (size - 1))

        elif window_type == "blackman":
            return (
                0.42
                - 0.5 * cp.cos(2 * cp.pi * n / (size - 1))
                + 0.08 * cp.cos(4 * cp.pi * n / (size - 1))
            )

        elif window_type == "blackman_harris":
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            return (
                a0
                - a1 * cp.cos(2 * cp.pi * n / (size - 1))
                + a2 * cp.cos(4 * cp.pi * n / (size - 1))
                - a3 * cp.cos(6 * cp.pi * n / (size - 1))
            )

        elif window_type == "kaiser":
            beta = params.get("beta", 6.0)
            # Kaiser window: I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)
            alpha = (size - 1) / 2.0
            x = beta * cp.sqrt(1 - ((n - alpha) / alpha) ** 2)

            # Modified Bessel function I0 approximation
            def bessel_i0(x):
                """Approximate I0 using polynomial expansion."""
                ax = cp.abs(x)
                result = cp.zeros_like(x)

                # For small x
                mask_small = ax < 3.75
                t = (ax[mask_small] / 3.75) ** 2
                result[mask_small] = 1.0 + t * (
                    3.5156229
                    + t
                    * (
                        3.0899424
                        + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))
                    )
                )

                # For large x
                mask_large = ~mask_small
                t = 3.75 / ax[mask_large]
                result[mask_large] = (
                    cp.exp(ax[mask_large])
                    / cp.sqrt(ax[mask_large])
                    * (
                        0.39894228
                        + t
                        * (
                            0.01328592
                            + t
                            * (
                                0.00225319
                                + t
                                * (
                                    -0.00157565
                                    + t
                                    * (
                                        0.00916281
                                        + t
                                        * (
                                            -0.02057706
                                            + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))
                                        )
                                    )
                                )
                            )
                        )
                    )
                )

                return result

            return bessel_i0(x) / bessel_i0(cp.array([beta]))[0]

        elif window_type == "flattop":
            a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
            return (
                a0
                - a1 * cp.cos(2 * cp.pi * n / (size - 1))
                + a2 * cp.cos(4 * cp.pi * n / (size - 1))
                - a3 * cp.cos(6 * cp.pi * n / (size - 1))
                + a4 * cp.cos(8 * cp.pi * n / (size - 1))
            )

        else:
            raise ValueError(f"Unknown window type: {window_type}")

    def get_correction_factors(
        self, window_type: str, size: int, **params
    ) -> WindowCorrectionFactors:
        """
        Get correction factors for a window.

        Args:
            window_type: Type of window.
            size: Window size.
            **params: Additional parameters.

        Returns:
            WindowCorrectionFactors dataclass.
        """
        # Return pre-computed factors for known windows
        if window_type in self._correction_factors and not params:
            return self._correction_factors[window_type]

        # Compute factors for custom windows (like Kaiser with specific beta)
        cache_key = (window_type, size, tuple(sorted(params.items())))
        if cache_key in self._correction_cache:
            return self._correction_cache[cache_key]

        # Generate window and compute factors
        window = self.get_window(window_type, size, **params)
        window_np = to_numpy(window)

        # Coherent gain
        coherent_gain = float(np.mean(window_np))

        # Noise bandwidth (in bins)
        noise_bandwidth = float(np.sum(window_np**2) / (np.sum(window_np) ** 2) * size)

        # ENBW
        enbw = noise_bandwidth

        # Processing gain relative to rectangular window
        processing_gain_db = float(10 * np.log10(1.0 / noise_bandwidth))

        # Max sidelobe (approximate for custom windows)
        max_sidelobe_db = self._estimate_sidelobe_level(window_np)

        factors = WindowCorrectionFactors(
            coherent_gain=coherent_gain,
            noise_bandwidth=noise_bandwidth,
            enbw=enbw,
            processing_gain_db=processing_gain_db,
            max_sidelobe_db=max_sidelobe_db,
        )

        self._correction_cache[cache_key] = factors
        return factors

    def _estimate_sidelobe_level(self, window: np.ndarray) -> float:
        """Estimate maximum sidelobe level via FFT."""
        # Zero-pad for better frequency resolution
        n_fft = len(window) * 8
        spectrum = np.abs(np.fft.fft(window, n_fft))
        spectrum_db = 20 * np.log10(spectrum / spectrum.max() + 1e-12)

        # Find main lobe width (first null)
        half_spec = spectrum_db[: n_fft // 2]
        main_lobe_end = np.argmax(half_spec[1:] > half_spec[:-1]) + 1

        # Find max sidelobe
        if main_lobe_end < len(half_spec) - 1:
            max_sidelobe = np.max(half_spec[main_lobe_end:])
        else:
            max_sidelobe = -100.0

        return float(max_sidelobe)

    def precompute(self, window_types: list[str], sizes: list[int], **params) -> None:
        """
        Precompute and cache multiple windows.

        Args:
            window_types: List of window types to precompute.
            sizes: List of sizes for each window type.
            **params: Additional parameters for windows.
        """
        for wtype in window_types:
            for size in sizes:
                self.get_window(wtype, size, **params)
                self.get_correction_factors(wtype, size, **params)

    def clear_cache(self) -> None:
        """Clear all cached windows."""
        self._cache.clear()
        self._correction_cache.clear()

    def cache_stats(self) -> dict:
        """Get cache statistics."""
        total_bytes = sum(w.nbytes if hasattr(w, "nbytes") else 0 for w in self._cache.values())
        return {
            "num_cached_windows": len(self._cache),
            "num_cached_corrections": len(self._correction_cache),
            "cache_size_mb": total_bytes / (1024 * 1024),
        }


# Module-level default generator
_default_generator = None


def get_window(window_type: str, size: int, **params) -> cp.ndarray:
    """
    Get a window function using the default generator.

    Args:
        window_type: Type of window.
        size: Window size.
        **params: Additional parameters.

    Returns:
        Window array.
    """
    global _default_generator
    if _default_generator is None:
        _default_generator = WindowGenerator()
    return _default_generator.get_window(window_type, size, **params)


def get_correction_factors(window_type: str, size: int, **params) -> WindowCorrectionFactors:
    """Get correction factors using the default generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = WindowGenerator()
    return _default_generator.get_correction_factors(window_type, size, **params)
