#!/usr/bin/env python3
"""
GPU RF Forensics Engine - Socket.IO Server Runner

Starts REST API and Socket.IO server for Devolution(x) Spectrum Detect frontend.

Usage: python scripts/run_socketio_server.py [--simulate]

Socket.IO Namespaces:
  /spectrum    - Real-time spectrum measurements (spectrum:measurement)
  /detections  - Threat detection events (detection:new, detection:update)
  /analytics   - Performance metrics (analytics:metrics)
  /hardware    - SDR device telemetry (hardware:status)
"""

import argparse
import asyncio
import signal
import sys
import os
import time
import logging
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_rest_api(host: str = "0.0.0.0", port: int = 8000):
    """Run REST API server using uvicorn."""
    import uvicorn
    from api.rest_api import RFForensicsAPI, create_rest_api

    api_manager = RFForensicsAPI()
    app = create_rest_api(api_manager)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


async def run_socketio_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    simulate: bool = False
):
    """Run Socket.IO server with optional simulation."""
    from api.socketio_server import create_socketio_server

    server = create_socketio_server(host=host, port=port)

    if not simulate:
        # Real pipeline mode
        try:
            from pipeline.orchestrator import RFForensicsPipeline
            from api.socketio_pipeline import connect_pipeline_to_socketio

            logger.info("[PIPELINE] Initializing RF Forensics Pipeline...")
            pipeline = RFForensicsPipeline()

            # Connect pipeline to Socket.IO server
            bridge = connect_pipeline_to_socketio(pipeline, server)
            logger.info("[PIPELINE] Pipeline connected to Socket.IO server")

            async def start_pipeline():
                await asyncio.sleep(2)  # Wait for server to start
                logger.info("[PIPELINE] Starting pipeline processing...")
                await pipeline.start()

            asyncio.create_task(start_pipeline())

        except ImportError as e:
            logger.warning(f"[WARNING] Pipeline not available: {e}")
            logger.warning("[WARNING] Falling back to simulation mode")
            simulate = True

    if simulate:
        # Simulation mode
        async def simulation_loop():
            await asyncio.sleep(2)
            logger.info("[SIM] Starting simulated data @ 20 Hz")

            center_freq = 915_000_000
            sample_rate = 2_048_000
            fft_size = 2048

            while True:
                # Generate spectrum
                psd_db = np.random.normal(-95, 3, fft_size).astype(np.float32)

                # Add random signals
                for _ in range(np.random.randint(0, 5)):
                    sig_bin = np.random.randint(100, fft_size - 100)
                    sig_power = np.random.uniform(-70, -40)
                    sig_width = np.random.randint(5, 30)
                    for i in range(-sig_width, sig_width + 1):
                        if 0 <= sig_bin + i < fft_size:
                            psd_db[sig_bin + i] = max(
                                psd_db[sig_bin + i],
                                sig_power * np.exp(-0.5 * (i / (sig_width / 2)) ** 2)
                            )

                await server.emit_spectrum_measurement(
                    psd_db=psd_db,
                    center_freq_hz=center_freq,
                    sample_rate_hz=sample_rate,
                    fft_size=fft_size
                )

                # Occasional detection
                if np.random.random() < 0.1:
                    await server.emit_detection_new({
                        "detection_id": int(time.time() * 1000) % 100000,
                        "center_freq_hz": center_freq + np.random.randint(-500000, 500000),
                        "bandwidth_hz": np.random.choice([125000, 250000, 500000]),
                        "peak_power_db": np.random.uniform(-70, -40),
                        "snr_db": np.random.uniform(10, 25),
                        "modulation_type": np.random.choice(["LoRa", "FSK", "GFSK", "WiFi", "Bluetooth"]),
                        "confidence": np.random.uniform(0.7, 0.95),
                        "duration": np.random.uniform(0.05, 0.5)
                    })

                # Hardware status every 5 seconds
                if int(time.time()) % 5 == 0:
                    await server.emit_hardware_status({
                        "device_id": "usdr-0",
                        "connected": True,
                        "temperature_c": 42.5 + np.random.uniform(-2, 2),
                        "sample_rate_hz": sample_rate,
                        "center_freq_hz": center_freq,
                        "buffer_fill": np.random.uniform(0.1, 0.3),
                        "streaming": True
                    })

                # Metrics every second
                if int(time.time() * 10) % 10 == 0:
                    await server.emit_analytics_metrics({
                        "cpu_percent": np.random.uniform(15, 35),
                        "memory_percent": np.random.uniform(40, 60),
                        "detection_rate": np.random.uniform(0.5, 2.0),
                        "latency_ms": np.random.uniform(5, 20),
                        "gpu_utilization": np.random.uniform(30, 70)
                    })

                await asyncio.sleep(0.05)  # 20 Hz

        asyncio.create_task(simulation_loop())

    # Start Socket.IO server
    await server.start()

    # Keep running
    while True:
        await asyncio.sleep(1)


async def main(args):
    """Run both REST API and Socket.IO servers."""

    print("=" * 60)
    print("GPU RF Forensics Engine - Backend Server")
    print("=" * 60)
    print(f"REST API:        http://{args.host}:{args.rest_port}")
    print(f"Socket.IO:       http://{args.host}:{args.ws_port}")
    print(f"Health:          http://{args.host}:{args.rest_port}/health")
    print(f"Simulation:      {'Enabled' if args.simulate else 'Disabled'}")
    print("=" * 60)
    print()
    print("Socket.IO Namespaces:")
    print("  /spectrum     - Real-time spectrum data")
    print("  /detections   - Threat detection events")
    print("  /analytics    - Performance metrics")
    print("  /hardware     - SDR device telemetry")
    print("=" * 60)
    print()

    # Create tasks for both servers
    tasks = []

    if not args.ws_only:
        tasks.append(asyncio.create_task(
            run_rest_api(args.host, args.rest_port)
        ))
        logger.info(f"[REST] Started on port {args.rest_port}")

    if not args.rest_only:
        tasks.append(asyncio.create_task(
            run_socketio_server(args.host, args.ws_port, args.simulate)
        ))
        logger.info(f"[SocketIO] Started on port {args.ws_port}")

    print("Press Ctrl+C to stop servers...")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutting down servers...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU RF Forensics Engine - Socket.IO Server Runner"
    )
    parser.add_argument(
        "--rest-port", type=int, default=8000,
        help="REST API port (default: 8000)"
    )
    parser.add_argument(
        "--ws-port", type=int, default=8765,
        help="Socket.IO port (default: 8765)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Generate simulated spectrum data"
    )
    parser.add_argument(
        "--rest-only", action="store_true",
        help="Run only REST API server"
    )
    parser.add_argument(
        "--ws-only", action="store_true",
        help="Run only Socket.IO server"
    )

    args = parser.parse_args()

    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nShutdown complete.")
