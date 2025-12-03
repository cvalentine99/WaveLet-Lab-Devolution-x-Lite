#!/usr/bin/env python3
"""
Benchmark: Holoscan Pipeline vs Python/CuPy Baseline

Compares throughput, latency, and efficiency between:
1. Python/CuPy baseline (current implementation)
2. NVIDIA Holoscan C++ pipeline (new implementation)

Target metrics:
- Throughput: 9.5+ MSPS (vs 6.57 MSPS baseline)
- Latency: <1ms (vs 1.60ms baseline)
- Efficiency: 95%+ (vs 66% baseline)
"""

import sys
import time
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Add parent to path
sys.path.insert(0, '/home/cvalentine/GPU Forensics /rf_forensics')


@dataclass
class BenchmarkResult:
    """Benchmark results for a single test."""
    name: str
    throughput_msps: float
    avg_latency_ms: float
    efficiency_pct: float
    total_samples: int
    total_frames: int
    duration_s: float


def benchmark_python_baseline(
    duration_s: float = 10.0,
    sample_rate: float = 10e6,
    buffer_size: int = 500_000
) -> BenchmarkResult:
    """Benchmark the Python/CuPy baseline implementation."""
    print("\n" + "=" * 60)
    print("Benchmarking Python/CuPy Baseline...")
    print("=" * 60)

    try:
        import cupy as cp
        from rf_forensics.dsp.psd import WelchPSD
        from rf_forensics.detection.cfar import CFARDetector
    except ImportError as e:
        print(f"  ERROR: {e}")
        return BenchmarkResult(
            name="Python/CuPy",
            throughput_msps=0.0,
            avg_latency_ms=0.0,
            efficiency_pct=0.0,
            total_samples=0,
            total_frames=0,
            duration_s=0.0
        )

    # Initialize components
    psd = WelchPSD(fft_size=1024, overlap=0.5, sample_rate=sample_rate)
    cfar = CFARDetector(num_reference=32, num_guard=4, pfa=1e-6)

    # Warm-up
    print("  Warming up GPU...")
    for _ in range(5):
        test_signal = cp.random.randn(buffer_size, dtype=cp.complex64)
        psd_result = psd.compute(test_signal)
        cfar.detect(psd_result)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    print(f"  Running benchmark for {duration_s}s...")
    total_samples = 0
    total_frames = 0
    latencies = []

    start_time = time.perf_counter()
    while time.perf_counter() - start_time < duration_s:
        frame_start = time.perf_counter()

        # Generate test signal (simulates SDR receive)
        signal = cp.random.randn(buffer_size, dtype=cp.complex64)

        # Process: PSD -> CFAR
        psd_result = psd.compute(signal)
        detections = cfar.detect(psd_result)

        # Sync to measure actual completion
        cp.cuda.Stream.null.synchronize()

        frame_end = time.perf_counter()
        latencies.append((frame_end - frame_start) * 1000)

        total_samples += buffer_size
        total_frames += 1

    elapsed = time.perf_counter() - start_time

    throughput = total_samples / elapsed / 1e6
    avg_latency = np.mean(latencies)
    efficiency = throughput / (sample_rate / 1e6) * 100

    print(f"  Throughput: {throughput:.2f} MSPS")
    print(f"  Latency:    {avg_latency:.2f} ms")
    print(f"  Efficiency: {efficiency:.1f}%")

    return BenchmarkResult(
        name="Python/CuPy",
        throughput_msps=throughput,
        avg_latency_ms=avg_latency,
        efficiency_pct=efficiency,
        total_samples=total_samples,
        total_frames=total_frames,
        duration_s=elapsed
    )


def benchmark_holoscan(
    duration_s: float = 10.0,
    sample_rate: float = 10e6,
    buffer_size: int = 500_000
) -> BenchmarkResult:
    """Benchmark the Holoscan C++ pipeline."""
    print("\n" + "=" * 60)
    print("Benchmarking Holoscan C++ Pipeline...")
    print("=" * 60)

    try:
        from rf_forensics.holoscan.python import HoloscanPipeline
    except (ImportError, NotImplementedError) as e:
        print(f"  ERROR: Holoscan module not available: {e}")
        print("  Build with: cd holoscan && ./build.sh")
        return BenchmarkResult(
            name="Holoscan",
            throughput_msps=0.0,
            avg_latency_ms=0.0,
            efficiency_pct=0.0,
            total_samples=0,
            total_frames=0,
            duration_s=0.0
        )

    # Configure pipeline
    pipeline = HoloscanPipeline()
    pipeline.configure(
        center_freq=915e6,
        sample_rate=sample_rate,
        buffer_size=buffer_size,
        simulate=True,  # Use simulated data
        fft_size=1024,
        overlap=0.5,
        num_reference=32,
        num_guard=4,
        pfa=1e-6,
        min_snr=6.0
    )

    # Start pipeline
    print(f"  Running benchmark for {duration_s}s...")
    pipeline.start()

    start_time = time.perf_counter()
    total_detections = 0
    latencies = []

    try:
        while time.perf_counter() - start_time < duration_s:
            det_start = time.perf_counter()
            detections = pipeline.get_detections(timeout_ms=100)
            det_end = time.perf_counter()

            if detections:
                total_detections += len(detections)
                latencies.append((det_end - det_start) * 1000)

    finally:
        pipeline.stop()

    elapsed = time.perf_counter() - start_time
    stats = pipeline.get_stats()

    total_frames = stats.get('total_frames', 0)
    total_samples = total_frames * buffer_size

    if elapsed > 0 and total_frames > 0:
        throughput = total_samples / elapsed / 1e6
        avg_latency = np.mean(latencies) if latencies else 0.0
        efficiency = throughput / (sample_rate / 1e6) * 100
    else:
        throughput = 0.0
        avg_latency = 0.0
        efficiency = 0.0

    print(f"  Throughput: {throughput:.2f} MSPS")
    print(f"  Latency:    {avg_latency:.2f} ms")
    print(f"  Efficiency: {efficiency:.1f}%")
    print(f"  Detections: {total_detections}")

    return BenchmarkResult(
        name="Holoscan",
        throughput_msps=throughput,
        avg_latency_ms=avg_latency,
        efficiency_pct=efficiency,
        total_samples=total_samples,
        total_frames=total_frames,
        duration_s=elapsed
    )


def print_comparison(baseline: BenchmarkResult, holoscan: BenchmarkResult):
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<20} {'Python/CuPy':<15} {'Holoscan':<15} {'Improvement':<15}")
    print("-" * 65)

    # Throughput
    if baseline.throughput_msps > 0:
        throughput_gain = (holoscan.throughput_msps / baseline.throughput_msps - 1) * 100
        throughput_str = f"+{throughput_gain:.1f}%" if throughput_gain > 0 else f"{throughput_gain:.1f}%"
    else:
        throughput_str = "N/A"
    print(f"{'Throughput (MSPS)':<20} {baseline.throughput_msps:<15.2f} {holoscan.throughput_msps:<15.2f} {throughput_str:<15}")

    # Latency
    if baseline.avg_latency_ms > 0:
        latency_gain = (1 - holoscan.avg_latency_ms / baseline.avg_latency_ms) * 100
        latency_str = f"-{latency_gain:.1f}%" if latency_gain > 0 else f"+{-latency_gain:.1f}%"
    else:
        latency_str = "N/A"
    print(f"{'Latency (ms)':<20} {baseline.avg_latency_ms:<15.2f} {holoscan.avg_latency_ms:<15.2f} {latency_str:<15}")

    # Efficiency
    eff_gain = holoscan.efficiency_pct - baseline.efficiency_pct
    eff_str = f"+{eff_gain:.1f}pp" if eff_gain > 0 else f"{eff_gain:.1f}pp"
    print(f"{'Efficiency (%)':<20} {baseline.efficiency_pct:<15.1f} {holoscan.efficiency_pct:<15.1f} {eff_str:<15}")

    print("-" * 65)

    # Target comparison
    print("\nTARGET METRICS:")
    target_throughput = 9.5
    target_latency = 1.0
    target_efficiency = 95.0

    print(f"  Throughput: {holoscan.throughput_msps:.2f} / {target_throughput:.1f} MSPS "
          f"({'PASS' if holoscan.throughput_msps >= target_throughput else 'FAIL'})")
    print(f"  Latency:    {holoscan.avg_latency_ms:.2f} / {target_latency:.1f} ms "
          f"({'PASS' if holoscan.avg_latency_ms <= target_latency else 'FAIL'})")
    print(f"  Efficiency: {holoscan.efficiency_pct:.1f} / {target_efficiency:.1f}% "
          f"({'PASS' if holoscan.efficiency_pct >= target_efficiency else 'FAIL'})")


def main():
    """Run benchmarks and compare."""
    print("=" * 60)
    print("RF Forensics Pipeline Benchmark")
    print("Python/CuPy vs NVIDIA Holoscan")
    print("=" * 60)
    print("\nConfiguration:")
    print("  Sample rate:  10 MSPS")
    print("  Buffer size:  500,000 samples")
    print("  FFT size:     1024")
    print("  CFAR:         32 ref, 4 guard, Pfa=1e-6")

    # Run benchmarks
    baseline_result = benchmark_python_baseline(duration_s=10.0)
    holoscan_result = benchmark_holoscan(duration_s=10.0)

    # Print comparison
    print_comparison(baseline_result, holoscan_result)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
