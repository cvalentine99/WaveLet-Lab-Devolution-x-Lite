#!/usr/bin/env python3
"""
Engineering Benchmark Suite for RF Forensics Pipeline
Collects detailed metrics for engineering analysis
"""

import json
import time
import statistics
import requests
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

API_BASE = "http://localhost:8000"

@dataclass
class SampleMetrics:
    """Metrics collected at each sample interval"""
    timestamp: float
    elapsed_sec: float
    # SDR metrics
    samples_received: int
    samples_dropped: int
    drop_rate_instant: float  # Drops in this interval
    drop_rate_cumulative: float
    overflow_total: int
    overflow_rate: float
    # Hardware metrics
    temperature_c: float
    pll_locked: bool
    actual_sample_rate_hz: int
    # Buffer metrics
    buffer_fill_percent: float
    backpressure_events: int
    # Pipeline metrics
    pipeline_state: str
    samples_processed: int
    detections_count: int
    throughput_msps: float
    gpu_memory_gb: float
    processing_latency_ms: float

@dataclass
class TestRun:
    """Complete test run results"""
    test_id: str
    target_sample_rate_msps: float
    start_time: str
    end_time: str
    duration_sec: float
    # Aggregate statistics
    total_samples_received: int
    total_samples_dropped: int
    total_detections: int
    # Drop statistics
    drop_rate_mean: float
    drop_rate_std: float
    drop_rate_min: float
    drop_rate_max: float
    drop_free_intervals: int
    total_intervals: int
    # Throughput statistics
    throughput_mean_msps: float
    throughput_std_msps: float
    throughput_min_msps: float
    throughput_max_msps: float
    # Latency statistics
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    # Buffer statistics
    buffer_fill_mean: float
    buffer_fill_max: float
    backpressure_events_total: int
    # Hardware statistics
    temp_mean_c: float
    temp_max_c: float
    pll_lock_failures: int
    # GPU statistics
    gpu_memory_mean_gb: float
    gpu_memory_max_gb: float
    # Input/Output rates
    actual_input_rate_msps: float
    effective_throughput_msps: float  # After drops
    # Raw samples for detailed analysis
    samples: List[SampleMetrics] = field(default_factory=list)
    # Status
    status: str = "unknown"  # pass, fail, marginal
    failure_reason: Optional[str] = None

def get_sdr_metrics() -> Dict:
    """Fetch current SDR metrics from API"""
    try:
        resp = requests.get(f"{API_BASE}/api/sdr/metrics", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  Warning: Failed to get SDR metrics: {e}")
    return {}

def get_pipeline_status() -> Dict:
    """Fetch pipeline status from API"""
    try:
        resp = requests.get(f"{API_BASE}/api/status", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"  Warning: Failed to get status: {e}")
    return {}

def set_sample_rate(rate_hz: int) -> bool:
    """Update sample rate via API"""
    try:
        resp = requests.post(
            f"{API_BASE}/api/sdr/config",
            json={"sample_rate_hz": rate_hz, "bandwidth_hz": rate_hz},
            timeout=10
        )
        success = resp.status_code == 200
        if success:
            # Verify the change took effect
            time.sleep(1)
            status = requests.get(f"{API_BASE}/api/sdr/status", timeout=5).json()
            actual = status.get("actual_sample_rate_hz", 0)
            if actual != rate_hz:
                print(f"  Warning: Requested {rate_hz/1e6:.0f} MSPS but got {actual/1e6:.0f} MSPS")
        return success
    except Exception as e:
        print(f"  Warning: Failed to set sample rate: {e}")
        return False

def run_test(target_msps: float, duration_sec: int = 60, sample_interval_sec: float = 1.0) -> TestRun:
    """Run a complete benchmark test at specified sample rate"""

    test_id = f"bench_{target_msps}msps_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target_hz = int(target_msps * 1_000_000)

    print(f"\n{'='*80}")
    print(f"ENGINEERING BENCHMARK: {target_msps} MSPS")
    print(f"{'='*80}")
    print(f"Test ID: {test_id}")
    print(f"Duration: {duration_sec} seconds")
    print(f"Sample interval: {sample_interval_sec} seconds")
    print()

    # Set sample rate
    print(f"Setting sample rate to {target_msps} MSPS...")
    if not set_sample_rate(target_hz):
        print("WARNING: Failed to set sample rate via API - continuing with current rate")

    # Wait for rate change to take effect
    time.sleep(3)

    # Collect baseline
    baseline_metrics = get_sdr_metrics()
    baseline_status = get_pipeline_status()

    if not baseline_metrics or not baseline_status:
        print("ERROR: Cannot get baseline metrics")
        # Return empty result
        return TestRun(
            test_id=test_id, target_sample_rate_msps=target_msps,
            start_time=datetime.now().isoformat(), end_time=datetime.now().isoformat(),
            duration_sec=0, total_samples_received=0, total_samples_dropped=0,
            total_detections=0, drop_rate_mean=0, drop_rate_std=0, drop_rate_min=0,
            drop_rate_max=0, drop_free_intervals=0, total_intervals=0,
            throughput_mean_msps=0, throughput_std_msps=0, throughput_min_msps=0,
            throughput_max_msps=0, latency_mean_ms=0, latency_std_ms=0,
            latency_p50_ms=0, latency_p95_ms=0, latency_p99_ms=0, latency_max_ms=0,
            buffer_fill_mean=0, buffer_fill_max=0, backpressure_events_total=0,
            temp_mean_c=0, temp_max_c=0, pll_lock_failures=0,
            gpu_memory_mean_gb=0, gpu_memory_max_gb=0,
            actual_input_rate_msps=0, effective_throughput_msps=0,
            samples=[], status="ERROR", failure_reason="Cannot get baseline metrics"
        )

    prev_received = baseline_metrics.get("samples", {}).get("total_received", 0)
    prev_dropped = baseline_metrics.get("samples", {}).get("total_dropped", 0)
    prev_processed = baseline_status.get("samples_processed", 0)

    samples: List[SampleMetrics] = []
    start_time = time.time()
    start_datetime = datetime.now().isoformat()

    print(f"Collecting metrics for {duration_sec} seconds...")
    print(f"{'Time':>6} {'In Rate':>10} {'Drop%':>8} {'Thrpt':>10} {'Lat':>8} {'Buf%':>6} {'Temp':>6} {'GPU':>6}")
    print("-" * 80)

    interval = 0
    while time.time() - start_time < duration_sec:
        time.sleep(sample_interval_sec)
        interval += 1

        metrics = get_sdr_metrics()
        status = get_pipeline_status()

        if not metrics or not status:
            continue

        current_received = metrics.get("samples", {}).get("total_received", 0)
        current_dropped = metrics.get("samples", {}).get("total_dropped", 0)
        current_processed = status.get("samples_processed", 0)

        # Calculate instant rates
        recv_delta = current_received - prev_received
        drop_delta = current_dropped - prev_dropped
        proc_delta = current_processed - prev_processed

        instant_drop_rate = (drop_delta / recv_delta * 100) if recv_delta > 0 else 0
        input_rate_msps = recv_delta / sample_interval_sec / 1e6

        elapsed = time.time() - start_time

        sample = SampleMetrics(
            timestamp=time.time(),
            elapsed_sec=elapsed,
            samples_received=current_received,
            samples_dropped=current_dropped,
            drop_rate_instant=instant_drop_rate,
            drop_rate_cumulative=metrics.get("samples", {}).get("drop_rate_percent", 0),
            overflow_total=metrics.get("overflow", {}).get("total", 0),
            overflow_rate=metrics.get("overflow", {}).get("rate_per_sec", 0),
            temperature_c=metrics.get("hardware", {}).get("temperature_c", 0),
            pll_locked=metrics.get("hardware", {}).get("pll_locked", False),
            actual_sample_rate_hz=metrics.get("hardware", {}).get("actual_sample_rate_hz", 0),
            buffer_fill_percent=metrics.get("backpressure", {}).get("buffer_fill_percent", 0),
            backpressure_events=metrics.get("backpressure", {}).get("events", 0),
            pipeline_state=status.get("state", "unknown"),
            samples_processed=current_processed,
            detections_count=status.get("detections_count", 0),
            throughput_msps=status.get("current_throughput_msps", 0),
            gpu_memory_gb=status.get("gpu_memory_used_gb", 0),
            processing_latency_ms=status.get("processing_latency_ms", 0)
        )
        samples.append(sample)

        # Update previous values
        prev_received = current_received
        prev_dropped = current_dropped
        prev_processed = current_processed

        # Print progress
        print(f"{elapsed:>5.1f}s {input_rate_msps:>9.2f}M {instant_drop_rate:>7.2f}% "
              f"{sample.throughput_msps:>9.2f}M {sample.processing_latency_ms:>7.2f} "
              f"{sample.buffer_fill_percent:>5.1f}% {sample.temperature_c:>5.1f}C "
              f"{sample.gpu_memory_gb:>5.2f}G")

    end_time = time.time()
    end_datetime = datetime.now().isoformat()

    if not samples:
        return TestRun(
            test_id=test_id, target_sample_rate_msps=target_msps,
            start_time=start_datetime, end_time=end_datetime,
            duration_sec=end_time - start_time, total_samples_received=0,
            total_samples_dropped=0, total_detections=0, drop_rate_mean=0,
            drop_rate_std=0, drop_rate_min=0, drop_rate_max=0,
            drop_free_intervals=0, total_intervals=0,
            throughput_mean_msps=0, throughput_std_msps=0, throughput_min_msps=0,
            throughput_max_msps=0, latency_mean_ms=0, latency_std_ms=0,
            latency_p50_ms=0, latency_p95_ms=0, latency_p99_ms=0, latency_max_ms=0,
            buffer_fill_mean=0, buffer_fill_max=0, backpressure_events_total=0,
            temp_mean_c=0, temp_max_c=0, pll_lock_failures=0,
            gpu_memory_mean_gb=0, gpu_memory_max_gb=0,
            actual_input_rate_msps=0, effective_throughput_msps=0,
            samples=[], status="ERROR", failure_reason="No samples collected"
        )

    # Calculate statistics
    drop_rates = [s.drop_rate_instant for s in samples]
    throughputs = [s.throughput_msps for s in samples if s.throughput_msps > 0]
    latencies = [s.processing_latency_ms for s in samples if s.processing_latency_ms > 0]
    buffer_fills = [s.buffer_fill_percent for s in samples]
    temps = [s.temperature_c for s in samples if s.temperature_c > 0]
    gpu_mems = [s.gpu_memory_gb for s in samples if s.gpu_memory_gb > 0]

    # Calculate actual input rate
    total_recv = samples[-1].samples_received - samples[0].samples_received
    test_duration = samples[-1].elapsed_sec - samples[0].elapsed_sec
    actual_input_msps = (total_recv / test_duration / 1e6) if test_duration > 0 else 0

    # Calculate effective throughput (accounting for drops)
    total_drops = samples[-1].samples_dropped - samples[0].samples_dropped
    effective_recv = total_recv - total_drops
    effective_msps = (effective_recv / test_duration / 1e6) if test_duration > 0 else 0

    def safe_percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data)-1)]

    def safe_stdev(data: List[float]) -> float:
        if len(data) < 2:
            return 0.0
        return statistics.stdev(data)

    # Determine pass/fail
    drop_free = sum(1 for d in drop_rates if d < 0.1)  # Allow <0.1% as "drop free"
    avg_drop = statistics.mean(drop_rates) if drop_rates else 0
    max_drop = max(drop_rates) if drop_rates else 0

    if avg_drop < 0.5 and max_drop < 2.0:
        status = "PASS"
        failure_reason = None
    elif avg_drop < 2.0:
        status = "MARGINAL"
        failure_reason = f"Average drop rate {avg_drop:.2f}% (threshold 0.5%)"
    else:
        status = "FAIL"
        failure_reason = f"Average drop rate {avg_drop:.2f}% exceeds 2% threshold"

    result = TestRun(
        test_id=test_id,
        target_sample_rate_msps=target_msps,
        start_time=start_datetime,
        end_time=end_datetime,
        duration_sec=end_time - start_time,
        total_samples_received=samples[-1].samples_received if samples else 0,
        total_samples_dropped=samples[-1].samples_dropped if samples else 0,
        total_detections=samples[-1].detections_count if samples else 0,
        drop_rate_mean=statistics.mean(drop_rates) if drop_rates else 0,
        drop_rate_std=safe_stdev(drop_rates),
        drop_rate_min=min(drop_rates) if drop_rates else 0,
        drop_rate_max=max(drop_rates) if drop_rates else 0,
        drop_free_intervals=drop_free,
        total_intervals=len(samples),
        throughput_mean_msps=statistics.mean(throughputs) if throughputs else 0,
        throughput_std_msps=safe_stdev(throughputs),
        throughput_min_msps=min(throughputs) if throughputs else 0,
        throughput_max_msps=max(throughputs) if throughputs else 0,
        latency_mean_ms=statistics.mean(latencies) if latencies else 0,
        latency_std_ms=safe_stdev(latencies),
        latency_p50_ms=safe_percentile(latencies, 50),
        latency_p95_ms=safe_percentile(latencies, 95),
        latency_p99_ms=safe_percentile(latencies, 99),
        latency_max_ms=max(latencies) if latencies else 0,
        buffer_fill_mean=statistics.mean(buffer_fills) if buffer_fills else 0,
        buffer_fill_max=max(buffer_fills) if buffer_fills else 0,
        backpressure_events_total=samples[-1].backpressure_events if samples else 0,
        temp_mean_c=statistics.mean(temps) if temps else 0,
        temp_max_c=max(temps) if temps else 0,
        pll_lock_failures=sum(1 for s in samples if not s.pll_locked),
        gpu_memory_mean_gb=statistics.mean(gpu_mems) if gpu_mems else 0,
        gpu_memory_max_gb=max(gpu_mems) if gpu_mems else 0,
        actual_input_rate_msps=actual_input_msps,
        effective_throughput_msps=effective_msps,
        samples=[asdict(s) for s in samples],
        status=status,
        failure_reason=failure_reason
    )

    print()
    print(f"{'='*80}")
    print(f"RESULT: {status}")
    if failure_reason:
        print(f"Reason: {failure_reason}")
    print(f"Actual Input Rate: {actual_input_msps:.2f} MSPS")
    print(f"Effective Throughput: {effective_msps:.2f} MSPS (after drops)")
    print(f"{'='*80}")

    return result

def generate_report(results: List[TestRun], output_path: Path) -> Dict:
    """Generate comprehensive engineering report"""

    # Get system info from API
    try:
        health = requests.get(f"{API_BASE}/health/detailed", timeout=5).json()
    except:
        health = {}

    report = {
        "report_metadata": {
            "title": "RF Forensics Pipeline - Engineering Benchmark Report",
            "generated_at": datetime.now().isoformat(),
            "system_info": {
                "platform": "Linux (Docker + NVIDIA Container Toolkit)",
                "gpu": health.get("gpu", {}).get("name", "NVIDIA CUDA GPU"),
                "gpu_memory_total_gb": health.get("gpu", {}).get("memory_total_gb", "N/A"),
                "cuda_version": health.get("gpu", {}).get("cuda_version", "N/A"),
                "sdr": "uSDR PCIe (LMS7002M-based)",
                "max_theoretical_rate_msps": 50
            },
            "test_parameters": {
                "sample_interval_sec": 1.0,
                "pass_threshold_drop_percent": 0.5,
                "marginal_threshold_drop_percent": 2.0
            }
        },
        "executive_summary": {
            "tests_run": len(results),
            "tests_passed": sum(1 for r in results if r.status == "PASS"),
            "tests_marginal": sum(1 for r in results if r.status == "MARGINAL"),
            "tests_failed": sum(1 for r in results if r.status == "FAIL"),
            "max_sustainable_rate_msps": max(
                (r.target_sample_rate_msps for r in results if r.status == "PASS"),
                default=0
            ),
            "recommended_operating_rate_msps": round(max(
                (r.target_sample_rate_msps for r in results if r.status == "PASS"),
                default=0
            ) * 0.9, 1),  # 10% margin
            "findings": []
        },
        "test_results": []
    }

    # Add findings
    findings = []
    pass_rates = [r.target_sample_rate_msps for r in results if r.status == "PASS"]
    fail_rates = [r.target_sample_rate_msps for r in results if r.status == "FAIL"]

    if pass_rates:
        findings.append(f"Pipeline sustains {max(pass_rates)} MSPS without significant sample loss")
    if fail_rates:
        findings.append(f"Sample drops observed at {min(fail_rates)} MSPS and above")

    # Check for thermal issues
    max_temp = max((r.temp_max_c for r in results if r.temp_max_c > 0), default=0)
    if max_temp > 60:
        findings.append(f"WARNING: Peak temperature {max_temp:.1f}°C exceeds recommended 60°C")
    elif max_temp > 0:
        findings.append(f"Thermal performance nominal (max {max_temp:.1f}°C)")

    # Check latency
    max_p99_lat = max((r.latency_p99_ms for r in results if r.latency_p99_ms > 0), default=0)
    if max_p99_lat > 10:
        findings.append(f"Processing latency P99 {max_p99_lat:.2f}ms exceeds 10ms target")
    elif max_p99_lat > 0:
        findings.append(f"Processing latency within spec (P99 {max_p99_lat:.2f}ms)")

    report["executive_summary"]["findings"] = findings

    for r in results:
        test_summary = {
            "test_id": r.test_id,
            "target_rate_msps": r.target_sample_rate_msps,
            "actual_rate_msps": round(r.actual_input_rate_msps, 2),
            "status": r.status,
            "failure_reason": r.failure_reason,
            "duration_sec": round(r.duration_sec, 1),
            "metrics": {
                "samples": {
                    "total_received": r.total_samples_received,
                    "total_received_human": f"{r.total_samples_received/1e9:.2f}B",
                    "total_dropped": r.total_samples_dropped,
                    "total_detections": r.total_detections,
                    "effective_throughput_msps": round(r.effective_throughput_msps, 2)
                },
                "drop_rate": {
                    "mean_percent": round(r.drop_rate_mean, 4),
                    "std_percent": round(r.drop_rate_std, 4),
                    "min_percent": round(r.drop_rate_min, 4),
                    "max_percent": round(r.drop_rate_max, 4),
                    "drop_free_intervals": r.drop_free_intervals,
                    "total_intervals": r.total_intervals,
                    "drop_free_ratio": round(r.drop_free_intervals / r.total_intervals, 4) if r.total_intervals > 0 else 0
                },
                "gpu_processing": {
                    "throughput_mean_msps": round(r.throughput_mean_msps, 2),
                    "throughput_std_msps": round(r.throughput_std_msps, 2),
                    "throughput_min_msps": round(r.throughput_min_msps, 2),
                    "throughput_max_msps": round(r.throughput_max_msps, 2),
                    "headroom_ratio": round(r.throughput_mean_msps / r.actual_input_rate_msps, 2) if r.actual_input_rate_msps > 0 else 0
                },
                "latency": {
                    "mean_ms": round(r.latency_mean_ms, 3),
                    "std_ms": round(r.latency_std_ms, 3),
                    "p50_ms": round(r.latency_p50_ms, 3),
                    "p95_ms": round(r.latency_p95_ms, 3),
                    "p99_ms": round(r.latency_p99_ms, 3),
                    "max_ms": round(r.latency_max_ms, 3)
                },
                "buffer": {
                    "mean_fill_percent": round(r.buffer_fill_mean, 2),
                    "max_fill_percent": round(r.buffer_fill_max, 2),
                    "backpressure_events": r.backpressure_events_total
                },
                "hardware": {
                    "temp_mean_c": round(r.temp_mean_c, 2),
                    "temp_max_c": round(r.temp_max_c, 2),
                    "pll_lock_failures": r.pll_lock_failures
                },
                "gpu_memory": {
                    "mean_gb": round(r.gpu_memory_mean_gb, 3),
                    "max_gb": round(r.gpu_memory_max_gb, 3)
                }
            },
            "time_series": r.samples  # Full time series for detailed analysis
        }
        report["test_results"].append(test_summary)

    # Write JSON report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed JSON report: {output_path}")

    # Also generate markdown summary
    md_path = output_path.with_suffix('.md')
    generate_markdown_report(report, md_path)

    return report

def generate_markdown_report(report: Dict, output_path: Path):
    """Generate human-readable markdown report"""

    md = []
    md.append("# RF Forensics Pipeline - Engineering Benchmark Report\n")
    md.append(f"**Generated:** {report['report_metadata']['generated_at']}\n")

    # System Info
    md.append("## System Configuration\n")
    sys_info = report['report_metadata']['system_info']
    md.append(f"| Parameter | Value |")
    md.append(f"|-----------|-------|")
    md.append(f"| Platform | {sys_info['platform']} |")
    md.append(f"| GPU | {sys_info['gpu']} |")
    md.append(f"| GPU Memory | {sys_info['gpu_memory_total_gb']} GB |")
    md.append(f"| CUDA Version | {sys_info['cuda_version']} |")
    md.append(f"| SDR | {sys_info['sdr']} |")
    md.append(f"| Max Theoretical Rate | {sys_info['max_theoretical_rate_msps']} MSPS |")
    md.append("")

    # Executive Summary
    summary = report['executive_summary']
    md.append("## Executive Summary\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| Tests Run | {summary['tests_run']} |")
    md.append(f"| Tests Passed | {summary['tests_passed']} |")
    md.append(f"| Tests Marginal | {summary['tests_marginal']} |")
    md.append(f"| Tests Failed | {summary['tests_failed']} |")
    md.append(f"| **Max Sustainable Rate** | **{summary['max_sustainable_rate_msps']} MSPS** |")
    md.append(f"| **Recommended Operating Rate** | **{summary['recommended_operating_rate_msps']} MSPS** |")
    md.append("")

    md.append("### Key Findings\n")
    for finding in summary['findings']:
        md.append(f"- {finding}")
    md.append("")

    # Results Table
    md.append("## Test Results Summary\n")
    md.append("| Rate (MSPS) | Status | Drop Rate | Throughput | Latency P95 | Buffer Fill | Temp |")
    md.append("|-------------|--------|-----------|------------|-------------|-------------|------|")

    for result in report['test_results']:
        m = result['metrics']
        status_emoji = "✅" if result['status'] == "PASS" else "⚠️" if result['status'] == "MARGINAL" else "❌"
        md.append(
            f"| {result['target_rate_msps']} | {status_emoji} {result['status']} | "
            f"{m['drop_rate']['mean_percent']:.2f}% ± {m['drop_rate']['std_percent']:.2f}% | "
            f"{m['gpu_processing']['throughput_mean_msps']:.1f} MSPS | "
            f"{m['latency']['p95_ms']:.2f} ms | "
            f"{m['buffer']['max_fill_percent']:.1f}% | "
            f"{m['hardware']['temp_max_c']:.1f}°C |"
        )
    md.append("")

    # Detailed Results
    md.append("## Detailed Results\n")

    for result in report['test_results']:
        m = result['metrics']
        md.append(f"### {result['target_rate_msps']} MSPS Test\n")
        md.append(f"**Status:** {result['status']}")
        if result['failure_reason']:
            md.append(f"  \n**Reason:** {result['failure_reason']}")
        md.append("")

        md.append("#### Sample Processing")
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Total Received | {m['samples']['total_received_human']} samples |")
        md.append(f"| Total Dropped | {m['samples']['total_dropped']:,} samples |")
        md.append(f"| Actual Input Rate | {result['actual_rate_msps']} MSPS |")
        md.append(f"| Effective Throughput | {m['samples']['effective_throughput_msps']} MSPS |")
        md.append(f"| Total Detections | {m['samples']['total_detections']:,} |")
        md.append("")

        md.append("#### Drop Rate Analysis")
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Mean Drop Rate | {m['drop_rate']['mean_percent']:.4f}% |")
        md.append(f"| Std Dev | {m['drop_rate']['std_percent']:.4f}% |")
        md.append(f"| Min | {m['drop_rate']['min_percent']:.4f}% |")
        md.append(f"| Max | {m['drop_rate']['max_percent']:.4f}% |")
        md.append(f"| Drop-Free Intervals | {m['drop_rate']['drop_free_intervals']}/{m['drop_rate']['total_intervals']} ({m['drop_rate']['drop_free_ratio']*100:.1f}%) |")
        md.append("")

        md.append("#### GPU Processing Performance")
        md.append(f"| Metric | Value |")
        md.append(f"|--------|-------|")
        md.append(f"| Mean Throughput | {m['gpu_processing']['throughput_mean_msps']:.2f} MSPS |")
        md.append(f"| Std Dev | {m['gpu_processing']['throughput_std_msps']:.2f} MSPS |")
        md.append(f"| Min | {m['gpu_processing']['throughput_min_msps']:.2f} MSPS |")
        md.append(f"| Max | {m['gpu_processing']['throughput_max_msps']:.2f} MSPS |")
        md.append(f"| Headroom Ratio | {m['gpu_processing']['headroom_ratio']:.2f}x |")
        md.append("")

        md.append("#### Latency Distribution")
        md.append(f"| Percentile | Value |")
        md.append(f"|------------|-------|")
        md.append(f"| Mean | {m['latency']['mean_ms']:.3f} ms |")
        md.append(f"| P50 | {m['latency']['p50_ms']:.3f} ms |")
        md.append(f"| P95 | {m['latency']['p95_ms']:.3f} ms |")
        md.append(f"| P99 | {m['latency']['p99_ms']:.3f} ms |")
        md.append(f"| Max | {m['latency']['max_ms']:.3f} ms |")
        md.append("")

        md.append("#### Resource Utilization")
        md.append(f"| Resource | Mean | Max |")
        md.append(f"|----------|------|-----|")
        md.append(f"| Buffer Fill | {m['buffer']['mean_fill_percent']:.1f}% | {m['buffer']['max_fill_percent']:.1f}% |")
        md.append(f"| GPU Memory | {m['gpu_memory']['mean_gb']:.3f} GB | {m['gpu_memory']['max_gb']:.3f} GB |")
        md.append(f"| SDR Temperature | {m['hardware']['temp_mean_c']:.1f}°C | {m['hardware']['temp_max_c']:.1f}°C |")
        md.append("")

    # Write markdown
    with open(output_path, 'w') as f:
        f.write('\n'.join(md))

    print(f"Markdown report: {output_path}")

def main():
    """Run engineering benchmark suite"""

    # Test rates to benchmark
    test_rates = [30, 35, 40]  # Start from known-good, test threshold, confirm failure
    duration = 60  # 60 seconds per test

    if len(sys.argv) > 1:
        test_rates = [float(r) for r in sys.argv[1].split(',')]
    if len(sys.argv) > 2:
        duration = int(sys.argv[2])

    print("="*80)
    print("RF FORENSICS PIPELINE - ENGINEERING BENCHMARK SUITE")
    print("="*80)
    print(f"Test rates: {test_rates} MSPS")
    print(f"Duration per test: {duration} seconds")
    print(f"Total estimated time: {len(test_rates) * (duration + 8)} seconds")
    print()

    results = []

    for rate in test_rates:
        result = run_test(rate, duration_sec=duration)
        results.append(result)

        # Brief pause between tests
        if rate != test_rates[-1]:
            print("\nPausing 5 seconds before next test...")
            time.sleep(5)

    # Generate report
    output_path = Path("/home/cvalentine/GPU Forensics /rf_forensics/docs/ENGINEERING_BENCHMARK_REPORT.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, output_path)

    # Print summary
    print("\n" + "="*80)
    print("ENGINEERING BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Rate':>6} {'Status':<10} {'Drop Rate':<18} {'Throughput':<16} {'Latency P95':<12} {'Temp':<8}")
    print("-"*80)

    for r in results:
        print(f"{r.target_sample_rate_msps:>5.0f}M {r.status:<10} "
              f"{r.drop_rate_mean:>6.2f}% ± {r.drop_rate_std:>5.2f}%   "
              f"{r.throughput_mean_msps:>6.1f} ± {r.throughput_std_msps:>5.1f}M  "
              f"{r.latency_p95_ms:>6.2f} ms    "
              f"{r.temp_max_c:>5.1f}°C")

    print("\n" + "="*80)
    max_pass = max((r.target_sample_rate_msps for r in results if r.status == "PASS"), default=0)
    print(f"Maximum Sustainable Rate: {max_pass} MSPS")
    print(f"Recommended Operating Rate: {max_pass * 0.9:.1f} MSPS (10% safety margin)")
    print("="*80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
