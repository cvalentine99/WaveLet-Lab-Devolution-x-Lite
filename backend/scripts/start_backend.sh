#!/bin/bash
# GPU RF Forensics Engine - Backend Startup Script
# Starts REST API (port 8000) and WebSocket server (port 8765)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "GPU RF Forensics Engine - Backend Startup"
echo "=============================================="
echo ""
echo "REST API:    http://localhost:8000"
echo "WebSocket:   ws://localhost:8765"
echo "Health:      http://localhost:8000/health"
echo ""

# Check for required Python packages
python -c "import fastapi, uvicorn, websockets" 2>/dev/null || {
    echo "[ERROR] Required packages not found. Run: pip install fastapi uvicorn websockets"
    exit 1
}

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $REST_PID $WS_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start REST API server (port 8000)
echo "Starting REST API server on port 8000..."
python -m uvicorn api.rest_api:app --host 0.0.0.0 --port 8000 &
REST_PID=$!

sleep 1

# Start WebSocket server (port 8765)
echo "Starting WebSocket server on port 8765..."
python -c "
import asyncio
import uvicorn
from api.websocket_server import SpectrumWebSocketServer, create_websocket_app

server = SpectrumWebSocketServer(port=8765)
app = create_websocket_app(server)
uvicorn.run(app, host='0.0.0.0', port=8765)
" &
WS_PID=$!

echo ""
echo "=============================================="
echo "Both servers started. Press Ctrl+C to stop."
echo "=============================================="
echo ""

# Wait for both processes
wait $REST_PID $WS_PID
