#!/usr/bin/env python3
"""
USDR Hardware Test with Ring Buffer Architecture

Decouples SDR acquisition from GPU processing:
- SDR callback: fast copy to ring buffer (no GPU work)
- GPU worker: async processing from ring buffer

This eliminates callback blocking and achieves full throughput.
"""

import sys
sys.path.insert(0, '/home/cvalentine/GPU Forensics ')

import time
import threading
import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

from rf_forensics.sdr.usdr_driver import USDRDriver, USDRConfig


# =============================================================================
# Ring Buffer for SDR -> GPU Pipeline
# =============================================================================
class RingBuffer:
    """Lock-free ring buffer for SDR samples."""

    def __init__(self, num_buffers: int = 8, buffer_size: int = 131072):
        self.num_buffers = num_buffers
        self.buffer_size = buffer_size

        # Pre-allocate buffers
        self.buffers = [np.zeros(buffer_size, dtype=np.complex64) for _ in range(num_buffers)]
        self.timestamps = [0.0] * num_buffers
        self.sizes = [0] * num_buffers

        # Indices (atomic via GIL)
        self.write_idx = 0
        self.read_idx = 0

        # Signaling
        self.data_ready = threading.Event()
        self.overflow_count = 0

    def write(self, samples: np.ndarray, timestamp: float) -> bool:
        """Write samples to buffer (called from SDR callback)."""
        next_idx = (self.write_idx + 1) % self.num_buffers

        # Check for overflow
        if next_idx == self.read_idx:
            self.overflow_count += 1
            return False

        # Copy to buffer
        n = min(len(samples), self.buffer_size)
        self.buffers[self.write_idx][:n] = samples[:n]
        self.sizes[self.write_idx] = n
        self.timestamps[self.write_idx] = timestamp

        # Advance write pointer
        self.write_idx = next_idx
        self.data_ready.set()
        return True

    def read(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Read samples from buffer (called from GPU worker)."""
        if self.read_idx == self.write_idx:
            # Buffer empty, wait for data
            self.data_ready.wait(timeout)
            self.data_ready.clear()
            if self.read_idx == self.write_idx:
                return None

        # Get data
        idx = self.read_idx
        samples = self.buffers[idx][:self.sizes[idx]]
        timestamp = self.timestamps[idx]

        # Advance read pointer
        self.read_idx = (self.read_idx + 1) % self.num_buffers

        return samples, timestamp

    def available(self) -> int:
        """Number of buffers available to read."""
        if self.write_idx >= self.read_idx:
            return self.write_idx - self.read_idx
        return self.num_buffers - self.read_idx + self.write_idx


# =============================================================================
# Fast GPU Processing (PSD + CFAR)
# =============================================================================
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
    """Fast CFAR with CuPy convolution."""

    def __init__(self, num_ref: int = 32, num_guard: int = 4, alpha_db: float = 15.0):
        self.alpha_db = alpha_db

        # Pre-compute kernel
        half_ref = num_ref // 2
        half_guard = num_guard // 2
        kernel_size = 2 * (half_ref + half_guard) + 1

        kernel = cp.zeros(kernel_size, dtype=cp.float32)
        kernel[:half_ref] = 1.0 / num_ref
        kernel[-half_ref:] = 1.0 / num_ref
        self.kernel = kernel

    def detect(self, psd_db: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        """Fast CA-CFAR detection."""
        noise_est = cp.convolve(psd_db, self.kernel, mode='same')
        threshold = noise_est + self.alpha_db
        mask = psd_db > threshold
        snr = psd_db - noise_est
        return mask, snr


# =============================================================================
# Main Test
# =============================================================================
def main():
    print("=" * 60)
    print("USDR HARDWARE TEST - Ring Buffer Architecture")
    print("=" * 60)

    # Configuration
    center_freq = 915e6
    sample_rate = 10e6
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

    # Initialize ring buffer
    ring_buffer = RingBuffer(num_buffers=16, buffer_size=131072)

    # Initialize GPU processing
    print("\n[2] Initializing GPU pipeline...")
    psd = FastPSD(fft_size=1024, sample_rate=sample_rate)
    cfar = FastCFAR(num_ref=32, num_guard=4, alpha_db=15.0)

    # Warm up GPU
    test_signal = cp.random.randn(131072).astype(cp.complex64)
    psd_result = psd.compute(test_signal)
    cfar.detect(psd_result)
    cp.cuda.Stream.null.synchronize()
    print("    GPU warmed up")

    # Statistics
    stats = {
        'total_samples': 0,
        'total_frames': 0,
        'total_detections': 0,
        'latencies': [],
        'start_time': None,
    }

    stop_event = threading.Event()

    # SDR callback - just queue data (fast!)
    def sdr_callback(samples: np.ndarray, timestamp: float):
        ring_buffer.write(samples, timestamp)

    # GPU worker thread
    def gpu_worker():
        last_print = time.perf_counter()

        while not stop_event.is_set():
            # Get data from ring buffer
            result = ring_buffer.read(timeout=0.05)
            if result is None:
                continue

            samples, timestamp = result

            if stats['start_time'] is None:
                stats['start_time'] = time.perf_counter()
                print(f"    Processing started: {len(samples)} samples/buffer")

            frame_start = time.perf_counter()

            # Transfer to GPU and process
            gpu_samples = cp.asarray(samples)
            psd_db = psd.compute(gpu_samples)
            mask, snr = cfar.detect(psd_db)
            cp.cuda.Stream.null.synchronize()

            latency = (time.perf_counter() - frame_start) * 1000
            stats['latencies'].append(latency)

            # Count detections
            mask_cpu = cp.asnumpy(mask)
            snr_cpu = cp.asnumpy(snr)
            frame_detections = np.sum((mask_cpu) & (snr_cpu > 6.0))

            stats['total_samples'] += len(samples)
            stats['total_frames'] += 1
            stats['total_detections'] += frame_detections

            # Print progress
            now = time.perf_counter()
            if now - last_print >= 1.0:
                elapsed = now - stats['start_time']
                throughput = stats['total_samples'] / elapsed / 1e6
                avg_lat = np.mean(stats['latencies'][-100:]) if stats['latencies'] else 0
                queued = ring_buffer.available()
                overflow = ring_buffer.overflow_count
                print(f"    t={elapsed:.0f}s: {throughput:.2f} MSPS, {avg_lat:.2f}ms lat, q={queued}, ovf={overflow}")
                last_print = now

    # Start GPU worker
    gpu_thread = threading.Thread(target=gpu_worker, daemon=True)
    gpu_thread.start()

    # Start SDR streaming
    print(f"\n[3] Running pipeline for {test_duration}s...")
    sdr.start_streaming(sdr_callback)

    try:
        time.sleep(test_duration)
    except KeyboardInterrupt:
        print("\n    Interrupted")
    finally:
        stop_event.set()
        sdr.stop_streaming()
        sdr.disconnect()
        gpu_thread.join(timeout=1.0)

    # Results
    if stats['start_time']:
        elapsed = time.perf_counter() - stats['start_time']
    else:
        elapsed = test_duration

    throughput = stats['total_samples'] / elapsed / 1e6 if elapsed > 0 else 0
    avg_latency = np.mean(stats['latencies']) if stats['latencies'] else 0
    efficiency = throughput / (sample_rate / 1e6) * 100

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Duration:     {elapsed:.2f}s")
    print(f"  Samples:      {stats['total_samples']:,}")
    print(f"  Frames:       {stats['total_frames']}")
    print(f"  Throughput:   {throughput:.2f} MSPS")
    print(f"  Avg Latency:  {avg_latency:.2f} ms")
    print(f"  Efficiency:   {efficiency:.1f}%")
    print(f"  Detections:   {stats['total_detections']}")
    print(f"  Overflows:    {ring_buffer.overflow_count}")

    print("\n" + "=" * 60)
    print("TARGET COMPARISON")
    print("=" * 60)
    tp_status = 'PASS' if throughput >= 9.5 else 'FAIL'
    lat_status = 'PASS' if avg_latency < 5.0 else 'WORKING'
    eff_status = 'PASS' if efficiency >= 95 else 'FAIL'
    print(f"  Throughput: {throughput:.2f} / 9.5 MSPS  -> {tp_status}")
    print(f"  Latency:    {avg_latency:.2f} / <5 ms    -> {lat_status}")
    print(f"  Efficiency: {efficiency:.1f} / 95%      -> {eff_status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
