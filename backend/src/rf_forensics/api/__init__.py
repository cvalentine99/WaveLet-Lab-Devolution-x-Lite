"""
GPU RF Forensics Engine - API Module

REST API and WebSocket server for frontend integration.
"""

from rf_forensics.api.frontend_adapter import (
    FrontendMessageFormatter,
    RESTRequestAdapter,
    RESTResponseAdapter,
    to_camel_case,
    to_snake_case,
)
from rf_forensics.api.pipeline_integration import (
    PipelineWebSocketBridge,
    connect_pipeline_to_websocket,
)
from rf_forensics.api.rest_api import RFForensicsAPI, create_rest_api
from rf_forensics.api.websocket_server import SpectrumWebSocketServer, create_websocket_app

__all__ = [
    "RFForensicsAPI",
    "create_rest_api",
    "SpectrumWebSocketServer",
    "create_websocket_app",
    "FrontendMessageFormatter",
    "RESTResponseAdapter",
    "RESTRequestAdapter",
    "PipelineWebSocketBridge",
    "connect_pipeline_to_websocket",
    "to_camel_case",
    "to_snake_case",
]
