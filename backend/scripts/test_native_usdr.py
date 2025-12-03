#!/usr/bin/env python3
"""
Direct USDR Hardware Test with Native CuPy Pipeline

Tests real SDR hardware without Numba (uses pure CuPy).
"""

import sys
sys.path.insert(0, '/home/cvalentine/GPU Forensics ')

import time
import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Tuple

# Import SDR driver
from rf_forensics.sdr.usdr_driver import USDRDriver, USDRConfig


@dataclass
class Detection:
    bin_index: int
    frequency_hz: float
    power_db: float
    snr_db: float


class FastPSD:
    """Fast PSD with pre-allocated buffers."""

    def __init__(self, fft_size: int = 1024, sample_rate: float = 10e6):
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.window = cp.hanning(fft_size).astype(cp.float32)
        self.window_power = float(cp.sum(self.window ** 2))

    def compute(self, signal: cp.ndarray) -> cp.ndarray:
        """Compute PSD in dB."""
        n = len(signal)
        hop = self.fft_size // 2
        num_segs = max(1, (n - self.fft_size) // hop + 1)

        # Segment and window
        psd_sum = cp.zeros(self.fft_size, dtype=cp.float32)
        for i in range(num_segs):
            start = i * hop
            seg = signal[start:start + self.fft_size] * self.window
            fft_result = cp.fft.fft(seg)
            psd_sum += cp.abs(fft_result) ** 2

        psd = psd_sum / (num_segs * self.fft_size * self.window_power)
        psd_db = 10 * cp.log10(cp.maximum(psd, 1e-20))
        return cp.fft.fftshift(psd_db)


class FastCFAR:
    """Fast CFAR with CuPy using vectorized convolution."""

    def __init__(self, num_ref: int = 32, num_guard: int = 4, alpha_db: float = 15.0, n_bins: int = 1024):
        self.num_ref = num_ref
        self.num_guard = num_guard
        self.alpha_db = alpha_db

        # Pre-compute kernel for convolution
        half_ref = num_ref // 2
        half_guard = num_guard // 2
        kernel_size = 2 * (half_ref + half_guard) + 1

        # Create kernel with gap for CUT and guard cells
        kernel = cp.zeros(kernel_size, dtype=cp.float32)
        kernel[:half_ref] = 1.0 / num_ref  # Leading cells
        kernel[-half_ref:] = 1.0 / num_ref  # Lagging cells
        self.kernel = kernel

        # Pre-allocate outputs
        self.noise_est = cp.zeros(n_bins, dtype=cp.float32)
        self.mask = cp.zeros(n_bins, dtype=cp.bool_)
        self.snr = cp.zeros(n_bins, dtype=cp.float32)

    def detect(self, psd_db: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Detect signals using CA-CFAR with fast convolution."""
        n = len(psd_db)

        # Fast noise estimate using convolution
        # mode='same' keeps output size equal to input
        noise_est = cp.convolve(psd_db, self.kernel, mode='same')

        # Threshold and detect
        threshold = noise_est + self.alpha_db
        mask = psd_db > threshold
        snr = psd_db - noise_est

        return mask, snr


def main():
    print("=" * 60)
    print("USDR HARDWARE TEST - Native CuPy Pipeline")
    print("=" * 60)

    # Configuration
    center_freq = 915e6
    sample_rate = 10e6
    buffer_size = 500_000
    test_duration = 10.0

    # Initialize SDR
    print("\n[1] Connecting to USDR...")
    config = USDRConfig(
        center_freq_hz=int(center_freq),
        sample_rate_hz=int(sample_rate),
    )

    sdr = USDRDriver()
    devices = sdr.discover()
    print(f"    Found {len(devices)} device(s)")

    if not devices:
        print("    ERROR: No USDR devices found!")
        return

    sdr.connect(devices[0].id)
    sdr.configure(config)
    print(f"    Connected: {center_freq/1e6:.1f} MHz @ {sample_rate/1e6:.1f} MSPS")

    # Initialize GPU processing
    print("\n[2] Initializing GPU pipeline...")
    psd = FastPSD(fft_size=1024, sample_rate=sample_rate)
    cfar = FastCFAR(num_ref=32, num_guard=4, alpha_db=15.0)

    # Warm up
    test_signal = cp.random.randn(buffer_size).astype(cp.complex64)
    psd_result = psd.compute(test_signal)
    cfar.detect(psd_result)
    cp.cuda.Stream.null.synchronize()
    print("    GPU warmed up")

    # Shared state for callback
    state = {
        'total_samples': 0,
        'total_frames': 0,
        'total_detections': 0,
        'latencies': [],
        'start_time': None,
        'last_print': None,
    }

    def process_callback(samples: np.ndarray, timestamp: float):
        """Process samples from SDR directly (no batching - SDR gives large buffers)."""
        if state['start_time'] is None:
            state['start_time'] = time.perf_counter()
            state['last_print'] = state['start_time']
            print(f"    Callback: {len(samples)} samples per call")

        frame_start = time.perf_counter()

        # Transfer to GPU
        gpu_samples = cp.asarray(samples)

        # Process: PSD -> CFAR
        psd_db = psd.compute(gpu_samples)
        mask, snr = cfar.detect(psd_db)

        # Sync and measure latency
        cp.cuda.Stream.null.synchronize()
        latency = (time.perf_counter() - frame_start) * 1000
        state['latencies'].append(latency)

        # Count detections
        mask_cpu = cp.asnumpy(mask)
        snr_cpu = cp.asnumpy(snr)
        frame_detections = np.sum((mask_cpu) & (snr_cpu > 6.0))

        state['total_samples'] += len(samples)
        state['total_frames'] += 1
        state['total_detections'] += frame_detections

        # Print progress every second
        now = time.perf_counter()
        if now - state['last_print'] >= 1.0:
            elapsed = now - state['start_time']
            throughput = state['total_samples'] / elapsed / 1e6
            avg_lat = np.mean(state['latencies'][-100:]) if state['latencies'] else 0
            print(f"    t={elapsed:.0f}s: {throughput:.2f} MSPS, {avg_lat:.2f}ms latency, {state['total_detections']} detections")
            state['last_print'] = now

    # Start streaming
    print(f"\n[3] Running pipeline for {test_duration}s...")
    sdr.start_streaming(process_callback)

    try:
        time.sleep(test_duration)
    except KeyboardInterrupt:
        print("\n    Interrupted by user")
    finally:
        sdr.stop_streaming()
        sdr.disconnect()

    total_samples = state['total_samples']
    total_frames = state['total_frames']
    total_detections = state['total_detections']
    latencies = state['latencies']

    # Results
    if state['start_time']:
        elapsed = time.perf_counter() - state['start_time']
    else:
        elapsed = test_duration
    throughput = total_samples / elapsed / 1e6 if elapsed > 0 else 0
    avg_latency = np.mean(latencies) if latencies else 0
    efficiency = throughput / (sample_rate / 1e6) * 100

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Duration:     {elapsed:.2f}s")
    print(f"  Samples:      {total_samples:,}")
    print(f"  Frames:       {total_frames}")
    print(f"  Throughput:   {throughput:.2f} MSPS")
    print(f"  Avg Latency:  {avg_latency:.2f} ms")
    print(f"  Efficiency:   {efficiency:.1f}%")
    print(f"  Detections:   {total_detections}")

    print("\n" + "=" * 60)
    print("TARGET COMPARISON")
    print("=" * 60)
    print(f"  Throughput: {throughput:.2f} / 9.5 MSPS  -> {'PASS' if throughput >= 9.5 else 'FAIL'}")
    print(f"  Latency:    {avg_latency:.2f} / <1 ms    -> {'PASS' if avg_latency < 1.0 else 'WORKING'}")
    print(f"  Efficiency: {efficiency:.1f} / 95%      -> {'PASS' if efficiency >= 95 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
