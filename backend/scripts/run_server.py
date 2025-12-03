#!/usr/bin/env python3
"""
GPU RF Forensics Engine - Unified Server Runner

Single-process architecture where REST API and WebSocket servers share
the same event loop and pipeline instance.

Usage: python scripts/run_server.py [--simulate]
"""

import argparse
import asyncio
import os
import signal
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


async def run_unified_server(
    host: str = "0.0.0.0",
    rest_port: int = 8000,
    ws_port: int = 8765,
    simulate: bool = False
):
    """
    Run unified server with REST API and WebSocket sharing the same pipeline.

    This single-process architecture ensures:
    - REST API /api/start, /api/stop actually control the pipeline
    - Both servers share state (detections, clusters, status)
    - No IPC overhead between processes
    """
    from rf_forensics.api.rest_api import RFForensicsAPI, create_rest_api
    from rf_forensics.api.websocket_server import SpectrumWebSocketServer, create_websocket_app

    # Create shared components
    pipeline = None
    bridge = None

    # Create WebSocket server
    ws_server = SpectrumWebSocketServer(host=host, port=ws_port)

    # Create API manager (will get pipeline attached below)
    recordings_dir = os.environ.get("RF_RECORDINGS_DIR", "/data/recordings")
    api_manager = RFForensicsAPI(recordings_dir=recordings_dir)

    if not simulate:
        try:
            from rf_forensics.pipeline.orchestrator import RFForensicsPipeline
            from rf_forensics.api.pipeline_integration import connect_pipeline_to_websocket

            print("[PIPELINE] Initializing RF Forensics Pipeline...")
            pipeline = RFForensicsPipeline()

            # Connect pipeline to WebSocket server AND REST API manager
            bridge = connect_pipeline_to_websocket(pipeline, ws_server, api_manager)
            print("[PIPELINE] Pipeline connected to WebSocket server")

            # Connect pipeline to REST API manager
            api_manager.set_pipeline(pipeline)
            print("[PIPELINE] Pipeline connected to REST API")

        except ImportError as e:
            print(f"[WARNING] Pipeline not available: {e}")
            print("[WARNING] Falling back to simulation mode")
            simulate = True

    # Create FastAPI apps
    rest_app = create_rest_api(api_manager)
    ws_app = create_websocket_app(ws_server, api_manager)  # Share api_manager for REST on 8765

    # Create uvicorn server configs
    rest_config = uvicorn.Config(
        rest_app,
        host=host,
        port=rest_port,
        log_level="info",
        loop="asyncio"
    )
    ws_config = uvicorn.Config(
        ws_app,
        host=host,
        port=ws_port,
        log_level="info",
        loop="asyncio"
    )

    rest_server = uvicorn.Server(rest_config)
    ws_server_uv = uvicorn.Server(ws_config)

    # Shutdown event
    shutdown_event = asyncio.Event()

    def signal_handler():
        print("\n[SHUTDOWN] Received shutdown signal...")
        shutdown_event.set()

    # Register signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    async def start_pipeline_delayed():
        """
        Check if pipeline can start after servers are ready.

        Pipeline only starts if SDR is connected (via frontend /api/sdr/connect).
        Otherwise waits for frontend to initiate connection.
        """
        if pipeline and not simulate:
            await asyncio.sleep(2)
            # Check if SDR is connected (frontend must connect first)
            if pipeline._sdr_manager and pipeline._sdr_manager.is_connected:
                print("[PIPELINE] SDR already connected - starting pipeline processing...")
                try:
                    await pipeline.start()
                except Exception as e:
                    print(f"[PIPELINE] Error starting pipeline: {e}")
            else:
                print("[PIPELINE] Waiting for SDR connection from frontend...")
                print("[PIPELINE] Frontend workflow: /api/sdr/devices -> /api/sdr/connect -> /api/start")

    async def run_simulation():
        """Generate simulated spectrum data."""
        await asyncio.sleep(2)
        print("[SIM] Starting simulated spectrum generation @ 20 Hz")

        center_freq = 915_000_000
        sample_rate = 2_048_000
        fft_size = 2048

        while not shutdown_event.is_set():
            # Generate noise floor with occasional signals
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

            # Broadcast as JSON
            await ws_server.broadcast_spectrum_json(
                psd_db=psd_db,
                center_freq_hz=center_freq,
                sample_rate_hz=sample_rate,
                fft_size=fft_size
            )

            # Occasionally send fake detections
            if np.random.random() < 0.1:
                detection = {
                    "detection_id": int(time.time() * 1000) % 100000,
                    "center_freq_hz": center_freq + np.random.randint(-500000, 500000),
                    "bandwidth_hz": np.random.choice([125000, 250000, 500000]),
                    "peak_power_db": np.random.uniform(-70, -40),
                    "snr_db": np.random.uniform(10, 25),
                    "modulation_type": np.random.choice(["LoRa", "FSK", "GFSK", "WiFi", "Bluetooth"]),
                    "confidence": np.random.uniform(0.7, 0.95),
                    "duration": np.random.uniform(0.05, 0.5)
                }
                await ws_server.send_detection_json(detection)
                api_manager.add_detection(detection)

            await asyncio.sleep(0.05)

    async def shutdown_servers():
        """Wait for shutdown signal and cleanup."""
        await shutdown_event.wait()
        print("[SHUTDOWN] Stopping servers...")

        # Stop pipeline if running
        if pipeline:
            try:
                await pipeline.stop()
                print("[PIPELINE] Pipeline stopped")
            except Exception as e:
                print(f"[PIPELINE] Error stopping pipeline: {e}")

        # Signal servers to stop
        rest_server.should_exit = True
        ws_server_uv.should_exit = True

    # Build task list
    tasks = [
        asyncio.create_task(rest_server.serve(), name="rest_server"),
        asyncio.create_task(ws_server_uv.serve(), name="ws_server"),
        asyncio.create_task(shutdown_servers(), name="shutdown_handler"),
    ]

    if simulate:
        tasks.append(asyncio.create_task(run_simulation(), name="simulation"))
    else:
        tasks.append(asyncio.create_task(start_pipeline_delayed(), name="pipeline_start"))

    # Run all tasks
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
    finally:
        print("[SHUTDOWN] Server shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="GPU RF Forensics Engine - Unified Server Runner"
    )
    parser.add_argument(
        "--rest-port", type=int, default=8000,
        help="REST API port (default: 8000)"
    )
    parser.add_argument(
        "--ws-port", type=int, default=8765,
        help="WebSocket port (default: 8765)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Generate simulated spectrum data"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GPU RF Forensics Engine - Unified Backend Server")
    print("=" * 60)
    print(f"REST API:    http://{args.host}:{args.rest_port}")
    print(f"WebSocket:   ws://{args.host}:{args.ws_port}")
    print(f"Health:      http://{args.host}:{args.rest_port}/health")
    print(f"Simulation:  {'Enabled' if args.simulate else 'Disabled (real SDR)'}")
    print("-" * 60)
    print("Architecture: Single-process (shared pipeline)")
    print("=" * 60)
    print("")

    # Run the unified server
    asyncio.run(
        run_unified_server(
            host=args.host,
            rest_port=args.rest_port,
            ws_port=args.ws_port,
            simulate=args.simulate
        )
    )


if __name__ == "__main__":
    main()
