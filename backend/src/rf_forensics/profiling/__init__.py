"""
RF Forensics GPU Profiling Module

Provides comprehensive GPU profiling and benchmarking tools:

- nvtx_markers: NVTX instrumentation for Nsight Systems
- gpu_metrics: GPU memory and bandwidth measurement
- benchmarks: Performance benchmark suite
- backpressure_test: Pipeline stress testing
- run_profiling: Main profiling runner

Usage:
    # Quick benchmark
    from rf_forensics.profiling.benchmarks import run_all_benchmarks
    results = run_all_benchmarks()

    # NVTX tracing
    from rf_forensics.profiling.nvtx_markers import nvtx_range
    with nvtx_range("my_operation"):
        do_work()

    # Full profiling suite
    python -m rf_forensics.profiling.run_profiling --full

    # With Nsight Systems
    nsys profile --trace=cuda,nvtx python -m rf_forensics.profiling.run_profiling --trace
"""

from rf_forensics.profiling.backpressure_test import (
    BackpressureStressTest,
    BackpressureTestResult,
    run_backpressure_suite,
)
from rf_forensics.profiling.benchmarks import (
    BenchmarkResult,
    CFARBenchmark,
    ClusteringBenchmark,
    FFTBenchmark,
    LatencyBenchmark,
    run_all_benchmarks,
)
from rf_forensics.profiling.gpu_metrics import (
    GPUMetricsCollector,
    MemorySnapshot,
    TransferMeasurement,
    measure_pcie_bandwidth,
)
from rf_forensics.profiling.nvtx_markers import (
    NVTX_COLORS,
    NVTXProfiler,
    get_profiler,
    nvtx_gpu_range,
    nvtx_mark,
    nvtx_range,
    nvtx_trace,
)

__all__ = [
    # NVTX
    "NVTXProfiler",
    "get_profiler",
    "nvtx_range",
    "nvtx_gpu_range",
    "nvtx_mark",
    "nvtx_trace",
    "NVTX_COLORS",
    # Metrics
    "GPUMetricsCollector",
    "MemorySnapshot",
    "TransferMeasurement",
    "measure_pcie_bandwidth",
    # Benchmarks
    "BenchmarkResult",
    "FFTBenchmark",
    "CFARBenchmark",
    "LatencyBenchmark",
    "ClusteringBenchmark",
    "run_all_benchmarks",
    # Backpressure
    "BackpressureTestResult",
    "BackpressureStressTest",
    "run_backpressure_suite",
]
