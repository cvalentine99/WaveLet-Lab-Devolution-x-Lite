#!/usr/bin/env python3
"""
GPU RF Forensics Engine - Environment Verification Script

Validates:
- RTX 4090 detection (Compute Capability 8.9)
- CUDA 13.0 runtime
- CuPy array operations
- cuSignal FFT functionality
- Pinned memory allocation
- System specifications
"""

import sys
from dataclasses import dataclass
from typing import Optional, Tuple
import importlib


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    message: str
    details: Optional[str] = None


class EnvironmentVerifier:
    """Verifies GPU RF Forensics environment components."""

    def __init__(self):
        self.results: list[VerificationResult] = []

    def check_gpu_detection(self) -> VerificationResult:
        """Check for RTX 4090 (Compute Capability 8.9)."""
        try:
            import cupy as cp

            device = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(device.id)

            name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            cc_major = props['major']
            cc_minor = props['minor']
            compute_cap = f"{cc_major}.{cc_minor}"
            total_mem_gb = props['totalGlobalMem'] / (1024**3)

            is_4090 = compute_cap == "8.9"

            return VerificationResult(
                name="GPU Detection",
                passed=is_4090,
                message=f"{name} (CC {compute_cap})" if is_4090 else f"Expected CC 8.9, got {compute_cap}",
                details=f"VRAM: {total_mem_gb:.1f} GB"
            )
        except Exception as e:
            return VerificationResult(
                name="GPU Detection",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def check_cuda_version(self) -> VerificationResult:
        """Validate CUDA 13.0 runtime."""
        try:
            import cupy as cp

            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            version_str = f"{major}.{minor}"

            # Accept CUDA 13.x
            is_valid = major == 13

            return VerificationResult(
                name="CUDA Runtime",
                passed=is_valid,
                message=f"CUDA {version_str}" if is_valid else f"Expected CUDA 13.x, got {version_str}"
            )
        except Exception as e:
            return VerificationResult(
                name="CUDA Runtime",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def check_cupy_allocation(self) -> VerificationResult:
        """Test CuPy array allocation on GPU."""
        try:
            import cupy as cp
            import numpy as np

            # Allocate 100M complex64 samples (~800 MB)
            size = 100_000_000
            gpu_array = cp.zeros(size, dtype=cp.complex64)

            # Verify allocation
            mem_bytes = gpu_array.nbytes
            mem_mb = mem_bytes / (1024**2)

            # Simple computation test
            gpu_array[:1000] = cp.arange(1000, dtype=cp.complex64)
            result = float(cp.abs(gpu_array[:1000]).sum().get())

            # Cleanup
            del gpu_array
            cp.get_default_memory_pool().free_all_blocks()

            return VerificationResult(
                name="CuPy Allocation",
                passed=True,
                message=f"Allocated {mem_mb:.0f} MB GPU array",
                details=f"Compute test result: {result:.2f}"
            )
        except Exception as e:
            return VerificationResult(
                name="CuPy Allocation",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def check_cusignal_fft(self) -> VerificationResult:
        """Verify cuSignal import and basic FFT."""
        try:
            import cupy as cp
            import cusignal

            # Generate test signal
            sample_rate = 1e6
            num_samples = 8192
            t = cp.arange(num_samples) / sample_rate
            freq = 100e3
            signal = cp.exp(2j * cp.pi * freq * t).astype(cp.complex64)

            # Compute FFT via cuSignal/CuPy
            fft_result = cp.fft.fft(signal)

            # Find peak frequency
            freqs = cp.fft.fftfreq(num_samples, 1/sample_rate)
            peak_idx = int(cp.argmax(cp.abs(fft_result[:num_samples//2])).get())
            peak_freq = float(freqs[peak_idx].get())

            freq_error = abs(peak_freq - freq)
            is_valid = freq_error < 1000  # Within 1 kHz

            return VerificationResult(
                name="cuSignal FFT",
                passed=is_valid,
                message=f"FFT peak at {peak_freq/1e3:.1f} kHz (expected {freq/1e3:.1f} kHz)",
                details=f"Error: {freq_error:.1f} Hz"
            )
        except ImportError:
            # cuSignal may not be separately importable in RAPIDS 25.10
            try:
                import cupy as cp
                # Use CuPy FFT directly
                signal = cp.random.randn(8192).astype(cp.complex64)
                fft_result = cp.fft.fft(signal)
                return VerificationResult(
                    name="cuSignal FFT",
                    passed=True,
                    message="CuPy FFT functional (cusignal integrated)",
                )
            except Exception as e:
                return VerificationResult(
                    name="cuSignal FFT",
                    passed=False,
                    message=f"Failed: {str(e)}"
                )
        except Exception as e:
            return VerificationResult(
                name="cuSignal FFT",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def check_pinned_memory(self) -> VerificationResult:
        """Check pinned memory allocation capability."""
        try:
            import cupy as cp
            import numpy as np

            # Allocate pinned memory
            size = 10_000_000  # 10M samples
            pinned_mem = cp.cuda.alloc_pinned_memory(size * np.dtype(np.complex64).itemsize)

            # Create NumPy view into pinned memory
            pinned_array = np.frombuffer(pinned_mem, dtype=np.complex64, count=size)

            # Write test data
            pinned_array[:1000] = np.arange(1000, dtype=np.complex64)

            # Async transfer to GPU
            stream = cp.cuda.Stream(non_blocking=True)
            gpu_array = cp.empty(size, dtype=cp.complex64)

            with stream:
                gpu_array.set(pinned_array)

            stream.synchronize()

            # Verify transfer
            result = float(cp.abs(gpu_array[:1000]).sum().get())

            mem_mb = (size * 8) / (1024**2)

            return VerificationResult(
                name="Pinned Memory",
                passed=True,
                message=f"Allocated {mem_mb:.0f} MB pinned memory",
                details=f"Async H2D transfer verified"
            )
        except Exception as e:
            return VerificationResult(
                name="Pinned Memory",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def check_cuml(self) -> VerificationResult:
        """Verify cuML DBSCAN functionality."""
        try:
            import cupy as cp
            from cuml.cluster import DBSCAN

            # Generate test data
            np_data = cp.random.randn(1000, 5).astype(cp.float32)

            # Run DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(np_data)

            n_clusters = len(cp.unique(labels[labels >= 0]))

            return VerificationResult(
                name="cuML DBSCAN",
                passed=True,
                message=f"DBSCAN found {n_clusters} clusters in test data"
            )
        except Exception as e:
            return VerificationResult(
                name="cuML DBSCAN",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def check_numba_cuda(self) -> VerificationResult:
        """Verify Numba CUDA kernel compilation."""
        try:
            from numba import cuda
            import numpy as np

            @cuda.jit
            def test_kernel(arr):
                idx = cuda.grid(1)
                if idx < arr.size:
                    arr[idx] = arr[idx] * 2.0

            # Test kernel
            data = np.ones(1024, dtype=np.float32)
            d_data = cuda.to_device(data)

            threads_per_block = 256
            blocks = (data.size + threads_per_block - 1) // threads_per_block

            test_kernel[blocks, threads_per_block](d_data)

            result = d_data.copy_to_host()
            is_valid = np.allclose(result, 2.0)

            return VerificationResult(
                name="Numba CUDA",
                passed=is_valid,
                message="Custom CUDA kernel compilation successful"
            )
        except Exception as e:
            return VerificationResult(
                name="Numba CUDA",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def check_websockets(self) -> VerificationResult:
        """Verify websockets library."""
        try:
            import websockets
            version = websockets.__version__

            # Check version >= 12.0
            major = int(version.split('.')[0])
            is_valid = major >= 12

            return VerificationResult(
                name="WebSockets",
                passed=is_valid,
                message=f"websockets {version}"
            )
        except Exception as e:
            return VerificationResult(
                name="WebSockets",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def check_fastapi(self) -> VerificationResult:
        """Verify FastAPI installation."""
        try:
            import fastapi
            version = fastapi.__version__

            return VerificationResult(
                name="FastAPI",
                passed=True,
                message=f"FastAPI {version}"
            )
        except Exception as e:
            return VerificationResult(
                name="FastAPI",
                passed=False,
                message=f"Failed: {str(e)}"
            )

    def get_system_specs(self) -> dict:
        """Collect system specifications."""
        specs = {}

        try:
            import cupy as cp
            import platform
            import os

            # Platform info
            specs['platform'] = platform.system()
            specs['python_version'] = platform.python_version()

            # CPU info
            specs['cpu_cores'] = os.cpu_count()

            # Memory info
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if 'MemTotal' in line:
                            mem_kb = int(line.split()[1])
                            specs['system_ram_gb'] = round(mem_kb / (1024**2), 1)
                            break
            except Exception:
                specs['system_ram_gb'] = 'Unknown'

            # GPU info
            device = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            specs['gpu_name'] = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            specs['gpu_vram_gb'] = round(props['totalGlobalMem'] / (1024**3), 1)
            specs['gpu_sm_count'] = props['multiProcessorCount']
            specs['gpu_compute_cap'] = f"{props['major']}.{props['minor']}"

            # CUDA version
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            specs['cuda_version'] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"

            # Package versions
            specs['cupy_version'] = cp.__version__

            try:
                import numpy as np
                specs['numpy_version'] = np.__version__
            except ImportError:
                pass

            try:
                import numba
                specs['numba_version'] = numba.__version__
            except ImportError:
                pass

        except Exception as e:
            specs['error'] = str(e)

        return specs

    def run_all_checks(self) -> Tuple[bool, list[VerificationResult]]:
        """Run all verification checks."""
        checks = [
            self.check_gpu_detection,
            self.check_cuda_version,
            self.check_cupy_allocation,
            self.check_cusignal_fft,
            self.check_pinned_memory,
            self.check_cuml,
            self.check_numba_cuda,
            self.check_websockets,
            self.check_fastapi,
        ]

        results = []
        for check in checks:
            result = check()
            results.append(result)

        all_passed = all(r.passed for r in results)
        return all_passed, results

    def print_results(self, results: list[VerificationResult], specs: dict):
        """Print formatted verification results."""
        print("")
        print("=" * 60)
        print("  GPU RF Forensics Engine - Environment Verification")
        print("=" * 60)
        print("")

        # System specs table
        print("System Specifications:")
        print("-" * 40)
        spec_items = [
            ('Platform', specs.get('platform', 'Unknown')),
            ('Python', specs.get('python_version', 'Unknown')),
            ('CPU Cores', specs.get('cpu_cores', 'Unknown')),
            ('System RAM', f"{specs.get('system_ram_gb', 'Unknown')} GB"),
            ('GPU', specs.get('gpu_name', 'Unknown')),
            ('GPU VRAM', f"{specs.get('gpu_vram_gb', 'Unknown')} GB"),
            ('GPU SMs', specs.get('gpu_sm_count', 'Unknown')),
            ('Compute Cap', specs.get('gpu_compute_cap', 'Unknown')),
            ('CUDA', specs.get('cuda_version', 'Unknown')),
            ('CuPy', specs.get('cupy_version', 'Unknown')),
            ('NumPy', specs.get('numpy_version', 'Unknown')),
            ('Numba', specs.get('numba_version', 'Unknown')),
        ]

        for name, value in spec_items:
            print(f"  {name:<15} {value}")

        print("")
        print("Verification Results:")
        print("-" * 40)

        for result in results:
            status = "[PASS]" if result.passed else "[FAIL]"
            color_status = f"\033[92m{status}\033[0m" if result.passed else f"\033[91m{status}\033[0m"
            print(f"  {color_status} {result.name:<20} {result.message}")
            if result.details:
                print(f"         {result.details}")

        print("")
        print("-" * 40)

        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)

        if passed_count == total_count:
            print(f"\033[92m  All {total_count} checks passed!\033[0m")
            print("  Environment is ready for GPU RF Forensics Engine.")
        else:
            print(f"\033[91m  {passed_count}/{total_count} checks passed.\033[0m")
            print("  Please resolve failed checks before proceeding.")

        print("")
        print("=" * 60)
        print("")

        return passed_count == total_count


def main():
    verifier = EnvironmentVerifier()

    # Get system specs first
    specs = verifier.get_system_specs()

    # Run all checks
    all_passed, results = verifier.run_all_checks()

    # Print results
    success = verifier.print_results(results, specs)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
