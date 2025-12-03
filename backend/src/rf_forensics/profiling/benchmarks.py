"""
GPU Benchmarks for RF Forensics Pipeline

Comprehensive benchmarks for:
- cuFFT plan reuse and warmup
- CFAR/peak kernel performance
- End-to-end pipeline latency
- Clustering GPU vs CPU
- Backpressure behavior
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.fft import get_fft_plan

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    get_fft_plan = None
    CUPY_AVAILABLE = False

from rf_forensics.profiling.nvtx_markers import NVTXProfiler


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    iterations: int
    total_ms: float
    avg_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    throughput: float | None = None  # Operations/sec or samples/sec
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        s = f"{self.name}: avg={self.avg_ms:.3f}ms, min={self.min_ms:.3f}ms, max={self.max_ms:.3f}ms"
        if self.throughput:
            s += f", throughput={self.throughput:.2f}"
        return s


class FFTBenchmark:
    """
    Benchmarks for cuFFT plan reuse and warmup behavior.

    Tests:
    - Cold vs warm FFT performance
    - Plan reuse efficiency
    - Batched FFT scaling
    - Memory allocation patterns
    """

    def __init__(self, profiler: NVTXProfiler | None = None):
        if not CUPY_AVAILABLE:
            raise RuntimeError("FFTBenchmark requires CuPy")
        self._profiler = profiler or NVTXProfiler()
        self._results: list[BenchmarkResult] = []

    def benchmark_plan_warmup(
        self, fft_size: int = 2048, num_warmup: int = 5, num_timed: int = 100
    ) -> BenchmarkResult:
        """
        Measure FFT performance with and without warmup.

        Shows the performance difference between first FFT (cold)
        and subsequent FFTs (warm) due to plan creation overhead.
        """
        # Generate test signal
        signal = cp.random.randn(fft_size) + 1j * cp.random.randn(fft_size)
        signal = signal.astype(cp.complex64)

        # Cold start (no plan cache)
        cp.fft.config.clear_plan_cache()
        cp.cuda.Device().synchronize()

        cold_times = []
        for i in range(3):
            cp.fft.config.clear_plan_cache()
            cp.cuda.Device().synchronize()

            start = time.perf_counter()
            _ = cp.fft.fft(signal)
            cp.cuda.Device().synchronize()
            cold_times.append((time.perf_counter() - start) * 1000)

        # Warm start (with plan cache)
        _ = cp.fft.fft(signal)  # Ensure plan is cached
        cp.cuda.Device().synchronize()

        warm_times = []
        for i in range(num_timed):
            start = time.perf_counter()
            _ = cp.fft.fft(signal)
            cp.cuda.Device().synchronize()
            warm_times.append((time.perf_counter() - start) * 1000)

        result = BenchmarkResult(
            name=f"fft_warmup_{fft_size}",
            iterations=num_timed,
            total_ms=sum(warm_times),
            avg_ms=statistics.mean(warm_times),
            min_ms=min(warm_times),
            max_ms=max(warm_times),
            std_ms=statistics.stdev(warm_times) if len(warm_times) > 1 else 0,
            extra={
                "cold_avg_ms": statistics.mean(cold_times),
                "warm_avg_ms": statistics.mean(warm_times),
                "speedup": statistics.mean(cold_times) / statistics.mean(warm_times),
                "fft_size": fft_size,
            },
        )
        self._results.append(result)
        return result

    def benchmark_plan_reuse(
        self, fft_size: int = 2048, num_segments: int = 16, iterations: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark explicit FFT plan reuse for batched operations.

        Compares:
        - Implicit plan (CuPy auto-caches)
        - Explicit plan with get_fft_plan
        """
        # Create batched signal
        signal = cp.random.randn(num_segments, fft_size) + 1j * cp.random.randn(
            num_segments, fft_size
        )
        signal = signal.astype(cp.complex64)

        # Create explicit plan
        plan = get_fft_plan(signal, axes=(1,))

        # Warmup
        with plan:
            _ = cp.fft.fft(signal, axis=1)
        cp.cuda.Device().synchronize()

        # Benchmark with explicit plan
        times_explicit = []
        for _ in range(iterations):
            start = time.perf_counter()
            with plan:
                _ = cp.fft.fft(signal, axis=1)
            cp.cuda.Device().synchronize()
            times_explicit.append((time.perf_counter() - start) * 1000)

        # Benchmark with implicit plan (cache)
        times_implicit = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = cp.fft.fft(signal, axis=1)
            cp.cuda.Device().synchronize()
            times_implicit.append((time.perf_counter() - start) * 1000)

        result = BenchmarkResult(
            name=f"fft_plan_reuse_{fft_size}x{num_segments}",
            iterations=iterations,
            total_ms=sum(times_explicit),
            avg_ms=statistics.mean(times_explicit),
            min_ms=min(times_explicit),
            max_ms=max(times_explicit),
            std_ms=statistics.stdev(times_explicit) if len(times_explicit) > 1 else 0,
            throughput=(num_segments * fft_size)
            / (statistics.mean(times_explicit) / 1000)
            / 1e6,  # MSamples/s
            extra={
                "explicit_avg_ms": statistics.mean(times_explicit),
                "implicit_avg_ms": statistics.mean(times_implicit),
                "fft_size": fft_size,
                "num_segments": num_segments,
            },
        )
        self._results.append(result)
        return result

    def benchmark_batched_scaling(
        self, fft_size: int = 2048, batch_sizes: list[int] = None, iterations: int = 50
    ) -> list[BenchmarkResult]:
        """
        Measure FFT throughput scaling with batch size.
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

        results = []
        for batch_size in batch_sizes:
            signal = cp.random.randn(batch_size, fft_size).astype(cp.complex64)
            plan = get_fft_plan(signal, axes=(1,))

            # Warmup
            with plan:
                _ = cp.fft.fft(signal, axis=1)
            cp.cuda.Device().synchronize()

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                with plan:
                    _ = cp.fft.fft(signal, axis=1)
                cp.cuda.Device().synchronize()
                times.append((time.perf_counter() - start) * 1000)

            throughput = (batch_size * fft_size) / (statistics.mean(times) / 1000) / 1e6

            result = BenchmarkResult(
                name=f"fft_batch_{batch_size}",
                iterations=iterations,
                total_ms=sum(times),
                avg_ms=statistics.mean(times),
                min_ms=min(times),
                max_ms=max(times),
                std_ms=statistics.stdev(times) if len(times) > 1 else 0,
                throughput=throughput,
                extra={"batch_size": batch_size, "fft_size": fft_size},
            )
            results.append(result)

        self._results.extend(results)
        return results

    def get_results(self) -> list[BenchmarkResult]:
        return self._results


class CFARBenchmark:
    """
    Benchmarks for CFAR detection kernel performance.

    Tests:
    - Kernel occupancy
    - Memory bandwidth utilization
    - Shared memory efficiency
    - Different CFAR variants
    """

    def __init__(self, profiler: NVTXProfiler | None = None):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CFARBenchmark requires CuPy")
        self._profiler = profiler or NVTXProfiler()
        self._results: list[BenchmarkResult] = []

    def benchmark_cfar_variants(
        self, n_bins: int = 2048, iterations: int = 100
    ) -> list[BenchmarkResult]:
        """
        Benchmark different CFAR variants (CA, GO, SO).
        """
        from rf_forensics.detection.cfar import CFARDetector

        # Generate test PSD
        psd = cp.random.exponential(1.0, n_bins).astype(cp.float32)

        variants = ["CA", "GO", "SO"]
        results = []

        for variant in variants:
            detector = CFARDetector(num_reference=32, num_guard=4, pfa=1e-6, variant=variant)

            # Warmup
            _ = detector.detect(psd)
            cp.cuda.Device().synchronize()

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = detector.detect(psd)
                cp.cuda.Device().synchronize()
                times.append((time.perf_counter() - start) * 1000)

            result = BenchmarkResult(
                name=f"cfar_{variant}_{n_bins}",
                iterations=iterations,
                total_ms=sum(times),
                avg_ms=statistics.mean(times),
                min_ms=min(times),
                max_ms=max(times),
                std_ms=statistics.stdev(times) if len(times) > 1 else 0,
                throughput=n_bins / (statistics.mean(times) / 1000) / 1e6,  # MBins/s
                extra={"variant": variant, "n_bins": n_bins},
            )
            results.append(result)

        self._results.extend(results)
        return results

    def benchmark_cfar_scaling(
        self, sizes: list[int] = None, iterations: int = 50
    ) -> list[BenchmarkResult]:
        """
        Benchmark CFAR performance scaling with input size.
        """
        from rf_forensics.detection.cfar import CFARDetector

        if sizes is None:
            sizes = [512, 1024, 2048, 4096, 8192, 16384]

        results = []
        for n_bins in sizes:
            psd = cp.random.exponential(1.0, n_bins).astype(cp.float32)
            detector = CFARDetector(num_reference=32, num_guard=4, pfa=1e-6)

            # Warmup
            _ = detector.detect(psd)
            cp.cuda.Device().synchronize()

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = detector.detect(psd)
                cp.cuda.Device().synchronize()
                times.append((time.perf_counter() - start) * 1000)

            result = BenchmarkResult(
                name=f"cfar_scaling_{n_bins}",
                iterations=iterations,
                total_ms=sum(times),
                avg_ms=statistics.mean(times),
                min_ms=min(times),
                max_ms=max(times),
                std_ms=statistics.stdev(times) if len(times) > 1 else 0,
                throughput=n_bins / (statistics.mean(times) / 1000) / 1e6,
                extra={"n_bins": n_bins},
            )
            results.append(result)

        self._results.extend(results)
        return results

    def get_results(self) -> list[BenchmarkResult]:
        return self._results


class LatencyBenchmark:
    """
    End-to-end pipeline latency benchmarks.

    Measures latency from sample arrival to detection output.
    """

    def __init__(self, profiler: NVTXProfiler | None = None):
        if not CUPY_AVAILABLE:
            raise RuntimeError("LatencyBenchmark requires CuPy")
        self._profiler = profiler or NVTXProfiler()
        self._results: list[BenchmarkResult] = []

    def benchmark_pipeline_latency(
        self, fft_size: int = 2048, sample_rate: float = 10e6, iterations: int = 100
    ) -> BenchmarkResult:
        """
        Measure full pipeline latency: H2D -> PSD -> CFAR -> D2H.
        """
        from rf_forensics.detection.cfar import CFARDetector
        from rf_forensics.dsp.psd import WelchPSD

        # Create components
        psd_estimator = WelchPSD(fft_size=fft_size, sample_rate=sample_rate)
        cfar = CFARDetector(num_reference=32, num_guard=4, pfa=1e-6)

        # Create test signal (simulates one buffer)
        buffer_size = fft_size * 4  # 4 segments
        host_signal = (np.random.randn(buffer_size) + 1j * np.random.randn(buffer_size)).astype(
            np.complex64
        )

        # Pinned memory for staging
        pinned = cp.cuda.alloc_pinned_memory(host_signal.nbytes)
        pinned_arr = np.frombuffer(pinned, dtype=np.complex64)
        np.copyto(pinned_arr, host_signal)

        # Pre-allocate GPU buffer
        gpu_signal = cp.empty(buffer_size, dtype=cp.complex64)

        # Warmup
        gpu_signal.set(pinned_arr)
        _, psd = psd_estimator.compute_psd(gpu_signal)
        psd_db = 10 * cp.log10(psd + 1e-12)
        mask, snr = cfar.detect(psd)
        psd_np = cp.asnumpy(psd_db)
        mask_np = cp.asnumpy(mask)
        cp.cuda.Device().synchronize()

        # Benchmark
        times = []
        stage_times = {"h2d": [], "psd": [], "cfar": [], "d2h": []}

        for _ in range(iterations):
            start = time.perf_counter()

            # H2D
            t0 = time.perf_counter()
            gpu_signal.set(pinned_arr)
            cp.cuda.Device().synchronize()
            stage_times["h2d"].append((time.perf_counter() - t0) * 1000)

            # PSD
            t0 = time.perf_counter()
            _, psd = psd_estimator.compute_psd(gpu_signal)
            psd_db = 10 * cp.log10(psd + 1e-12)
            cp.cuda.Device().synchronize()
            stage_times["psd"].append((time.perf_counter() - t0) * 1000)

            # CFAR
            t0 = time.perf_counter()
            mask, snr = cfar.detect(psd)
            cp.cuda.Device().synchronize()
            stage_times["cfar"].append((time.perf_counter() - t0) * 1000)

            # D2H
            t0 = time.perf_counter()
            psd_np = cp.asnumpy(psd_db)
            mask_np = cp.asnumpy(mask)
            stage_times["d2h"].append((time.perf_counter() - t0) * 1000)

            times.append((time.perf_counter() - start) * 1000)

        result = BenchmarkResult(
            name=f"pipeline_latency_{fft_size}",
            iterations=iterations,
            total_ms=sum(times),
            avg_ms=statistics.mean(times),
            min_ms=min(times),
            max_ms=max(times),
            std_ms=statistics.stdev(times) if len(times) > 1 else 0,
            throughput=buffer_size / (statistics.mean(times) / 1000) / 1e6,  # MSamples/s
            extra={
                "h2d_avg_ms": statistics.mean(stage_times["h2d"]),
                "psd_avg_ms": statistics.mean(stage_times["psd"]),
                "cfar_avg_ms": statistics.mean(stage_times["cfar"]),
                "d2h_avg_ms": statistics.mean(stage_times["d2h"]),
                "fft_size": fft_size,
                "buffer_size": buffer_size,
                "p99_ms": sorted(times)[int(len(times) * 0.99)],
            },
        )
        self._results.append(result)
        return result

    def benchmark_async_overlap(
        self, fft_size: int = 2048, iterations: int = 50
    ) -> BenchmarkResult:
        """
        Measure latency with async D2H overlap.

        Compares sync vs async D2H transfer with compute overlap.
        """
        from rf_forensics.detection.cfar import CFARDetector
        from rf_forensics.dsp.psd import WelchPSD

        psd_estimator = WelchPSD(fft_size=fft_size)
        cfar = CFARDetector()

        buffer_size = fft_size * 4
        gpu_signal = cp.random.randn(buffer_size).astype(cp.complex64)

        # Create streams
        compute_stream = cp.cuda.Stream(non_blocking=True)
        d2h_stream = cp.cuda.Stream(non_blocking=True)

        # Pinned buffer for D2H
        pinned = cp.cuda.alloc_pinned_memory(fft_size * 4)  # float32
        pinned_arr = np.frombuffer(pinned, dtype=np.float32, count=fft_size)

        # Sync benchmark
        sync_times = []
        for _ in range(iterations):
            start = time.perf_counter()

            # Compute
            _, psd = psd_estimator.compute_psd(gpu_signal)
            psd_db = 10 * cp.log10(psd + 1e-12)
            mask, snr = cfar.detect(psd)
            cp.cuda.Device().synchronize()

            # Sync D2H
            psd_np = cp.asnumpy(psd_db)

            sync_times.append((time.perf_counter() - start) * 1000)

        # Async benchmark (overlap D2H with next frame compute)
        async_times = []
        prev_psd_db = None

        for i in range(iterations + 1):
            start = time.perf_counter()

            # Compute current frame on compute stream
            with compute_stream:
                _, psd = psd_estimator.compute_psd(gpu_signal)
                psd_db = 10 * cp.log10(psd + 1e-12)
                mask, snr = cfar.detect(psd)

            # Async D2H of previous frame on d2h stream
            if prev_psd_db is not None:
                src_ptr = prev_psd_db.data.ptr
                dst_ptr = pinned_arr.ctypes.data
                cp.cuda.runtime.memcpyAsync(
                    dst_ptr,
                    src_ptr,
                    fft_size * 4,
                    cp.cuda.runtime.memcpyDeviceToHost,
                    d2h_stream.ptr,
                )

            # Sync both streams
            compute_stream.synchronize()
            d2h_stream.synchronize()

            if i > 0:  # Skip first iteration (warmup)
                async_times.append((time.perf_counter() - start) * 1000)

            prev_psd_db = psd_db

        result = BenchmarkResult(
            name=f"async_overlap_{fft_size}",
            iterations=iterations,
            total_ms=sum(async_times),
            avg_ms=statistics.mean(async_times),
            min_ms=min(async_times),
            max_ms=max(async_times),
            std_ms=statistics.stdev(async_times) if len(async_times) > 1 else 0,
            extra={
                "sync_avg_ms": statistics.mean(sync_times),
                "async_avg_ms": statistics.mean(async_times),
                "speedup": statistics.mean(sync_times) / statistics.mean(async_times),
                "fft_size": fft_size,
            },
        )
        self._results.append(result)
        return result

    def get_results(self) -> list[BenchmarkResult]:
        return self._results


class ClusteringBenchmark:
    """
    Benchmarks for clustering GPU vs CPU performance.
    """

    def __init__(self):
        self._results: list[BenchmarkResult] = []

    def benchmark_dbscan_gpu_vs_cpu(
        self, n_samples: int = 1000, n_features: int = 9, iterations: int = 20
    ) -> dict[str, BenchmarkResult]:
        """
        Compare cuML DBSCAN vs sklearn DBSCAN.
        """
        # Generate test features
        np.random.seed(42)
        features_np = np.random.randn(n_samples, n_features).astype(np.float32)

        results = {}

        # CPU (sklearn)
        try:
            from sklearn.cluster import DBSCAN as skDBSCAN

            cpu_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                clusterer = skDBSCAN(eps=0.5, min_samples=5)
                labels = clusterer.fit_predict(features_np)
                cpu_times.append((time.perf_counter() - start) * 1000)

            results["cpu"] = BenchmarkResult(
                name=f"dbscan_cpu_{n_samples}",
                iterations=iterations,
                total_ms=sum(cpu_times),
                avg_ms=statistics.mean(cpu_times),
                min_ms=min(cpu_times),
                max_ms=max(cpu_times),
                std_ms=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0,
                throughput=n_samples / (statistics.mean(cpu_times) / 1000),
                extra={"n_samples": n_samples, "n_features": n_features},
            )
        except ImportError:
            pass

        # GPU (cuML)
        if CUPY_AVAILABLE:
            try:
                from cuml.cluster import DBSCAN as cuDBSCAN

                features_gpu = cp.asarray(features_np)

                # Warmup
                clusterer = cuDBSCAN(eps=0.5, min_samples=5)
                _ = clusterer.fit_predict(features_gpu)
                cp.cuda.Device().synchronize()

                gpu_times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    clusterer = cuDBSCAN(eps=0.5, min_samples=5)
                    labels = clusterer.fit_predict(features_gpu)
                    cp.cuda.Device().synchronize()
                    gpu_times.append((time.perf_counter() - start) * 1000)

                results["gpu"] = BenchmarkResult(
                    name=f"dbscan_gpu_{n_samples}",
                    iterations=iterations,
                    total_ms=sum(gpu_times),
                    avg_ms=statistics.mean(gpu_times),
                    min_ms=min(gpu_times),
                    max_ms=max(gpu_times),
                    std_ms=statistics.stdev(gpu_times) if len(gpu_times) > 1 else 0,
                    throughput=n_samples / (statistics.mean(gpu_times) / 1000),
                    extra={"n_samples": n_samples, "n_features": n_features},
                )
            except ImportError:
                pass

        if "cpu" in results and "gpu" in results:
            results["speedup"] = results["cpu"].avg_ms / results["gpu"].avg_ms

        self._results.extend(
            results.values() if isinstance(list(results.values())[0], BenchmarkResult) else []
        )
        return results

    def get_results(self) -> list[BenchmarkResult]:
        return self._results


def run_all_benchmarks(
    fft_size: int = 2048, iterations: int = 50, verbose: bool = True
) -> dict[str, list[BenchmarkResult]]:
    """
    Run complete benchmark suite.

    Args:
        fft_size: FFT size to use
        iterations: Number of iterations per benchmark
        verbose: Print results as they complete

    Returns:
        Dict of benchmark category -> results
    """
    results = {}

    if verbose:
        print("\n" + "=" * 70)
        print("RF Forensics GPU Benchmark Suite")
        print("=" * 70)

    # FFT Benchmarks
    if verbose:
        print("\n[FFT Benchmarks]")
    fft_bench = FFTBenchmark()
    warmup = fft_bench.benchmark_plan_warmup(fft_size, iterations=iterations)
    if verbose:
        print(
            f"  Plan warmup: cold={warmup.extra['cold_avg_ms']:.3f}ms, "
            f"warm={warmup.extra['warm_avg_ms']:.3f}ms, "
            f"speedup={warmup.extra['speedup']:.1f}x"
        )

    reuse = fft_bench.benchmark_plan_reuse(fft_size, iterations=iterations)
    if verbose:
        print(f"  Plan reuse: {reuse.throughput:.1f} MSamples/s")

    scaling = fft_bench.benchmark_batched_scaling(fft_size, iterations=iterations // 2)
    if verbose:
        for r in scaling:
            print(f"  Batch {r.extra['batch_size']:>3}: {r.throughput:.1f} MSamples/s")

    results["fft"] = fft_bench.get_results()

    # CFAR Benchmarks
    if verbose:
        print("\n[CFAR Benchmarks]")
    cfar_bench = CFARBenchmark()
    variants = cfar_bench.benchmark_cfar_variants(fft_size, iterations=iterations)
    if verbose:
        for r in variants:
            print(f"  {r.extra['variant']}: {r.avg_ms:.3f}ms, {r.throughput:.1f} MBins/s")

    results["cfar"] = cfar_bench.get_results()

    # Latency Benchmarks
    if verbose:
        print("\n[Pipeline Latency]")
    latency_bench = LatencyBenchmark()
    pipeline = latency_bench.benchmark_pipeline_latency(fft_size, iterations=iterations)
    if verbose:
        print(f"  Total: {pipeline.avg_ms:.3f}ms (p99: {pipeline.extra['p99_ms']:.3f}ms)")
        print(f"    H2D:  {pipeline.extra['h2d_avg_ms']:.3f}ms")
        print(f"    PSD:  {pipeline.extra['psd_avg_ms']:.3f}ms")
        print(f"    CFAR: {pipeline.extra['cfar_avg_ms']:.3f}ms")
        print(f"    D2H:  {pipeline.extra['d2h_avg_ms']:.3f}ms")

    overlap = latency_bench.benchmark_async_overlap(fft_size, iterations=iterations // 2)
    if verbose:
        print(f"  Async overlap: {overlap.extra['speedup']:.2f}x speedup")

    results["latency"] = latency_bench.get_results()

    # Clustering Benchmarks
    if verbose:
        print("\n[Clustering]")
    cluster_bench = ClusteringBenchmark()
    cluster_results = cluster_bench.benchmark_dbscan_gpu_vs_cpu(iterations=iterations // 2)
    if verbose:
        if "cpu" in cluster_results:
            print(f"  CPU: {cluster_results['cpu'].avg_ms:.3f}ms")
        if "gpu" in cluster_results:
            print(f"  GPU: {cluster_results['gpu'].avg_ms:.3f}ms")
        if "speedup" in cluster_results:
            print(f"  Speedup: {cluster_results['speedup']:.1f}x")

    results["clustering"] = cluster_bench.get_results()

    if verbose:
        print("\n" + "=" * 70)
        print("Benchmark Complete")
        print("=" * 70)

    return results


if __name__ == "__main__":
    run_all_benchmarks(verbose=True)
