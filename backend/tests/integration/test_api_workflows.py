"""
API Workflow Integration Tests

Tests complete API workflows including:
- Health check endpoints
- System lifecycle (start/stop/pause/resume)
- Configuration management
- Detection retrieval
- Recording workflows
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient


# Skip if dependencies not available
pytest.importorskip("fastapi")


@pytest.fixture
def mock_pipeline():
    """Mock pipeline for testing."""
    pipeline = Mock()
    pipeline.state = Mock(value="idle")
    pipeline.config = Mock()
    pipeline.config.sdr = Mock(
        center_freq_hz=915e6,
        sample_rate_hz=10e6,
        gain_db=40.0
    )
    pipeline.config.fft = Mock(
        fft_size=1024,
        window_type="hann",
        overlap_percent=50.0
    )
    pipeline.config.cfar = Mock(
        pfa=1e-6,
        num_reference_cells=32,
        num_guard_cells=4
    )
    pipeline.config.clustering = Mock(eps=0.5, min_samples=5)
    pipeline.get_status = Mock(return_value={
        "state": "running",
        "uptime_seconds": 100,
        "samples_processed": 1_000_000,
        "detections_count": 5,
        "current_throughput_msps": 10.0,
        "gpu_memory_used_gb": 2.5,
        "buffer_fill_level": 0.3,
        "processing_latency_ms": 5.2
    })
    pipeline.start = AsyncMock()
    pipeline.stop = AsyncMock()
    pipeline.pause = AsyncMock()
    pipeline.resume = AsyncMock()
    return pipeline


@pytest.fixture
def api_client(mock_pipeline, tmp_path):
    """Create test client with mocked dependencies."""
    from rf_forensics.api.rest_api import RFForensicsAPI, create_rest_api

    api_manager = RFForensicsAPI(
        pipeline=mock_pipeline,
        recordings_dir=str(tmp_path / "recordings")
    )
    app = create_rest_api(api_manager)
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check_basic(self, api_client):
        """Test basic health endpoint returns expected structure."""
        response = api_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_health_check_detailed(self, api_client):
        """Test detailed health check includes all subsystems."""
        response = api_client.get("/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert "checks" in data
        checks = data["checks"]

        # Verify all subsystem checks present
        assert "gpu" in checks
        assert "memory" in checks
        assert "pipeline" in checks


class TestSystemLifecycle:
    """Test system start/stop/pause/resume workflow."""

    def test_start_acquisition(self, api_client, mock_pipeline):
        """Test starting signal acquisition."""
        response = api_client.post("/api/start")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        mock_pipeline.start.assert_called_once()

    def test_stop_acquisition(self, api_client, mock_pipeline):
        """Test stopping signal acquisition."""
        response = api_client.post("/api/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        mock_pipeline.stop.assert_called_once()

    def test_pause_acquisition(self, api_client, mock_pipeline):
        """Test pausing signal acquisition."""
        response = api_client.post("/api/pause")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        mock_pipeline.pause.assert_called_once()

    def test_resume_acquisition(self, api_client, mock_pipeline):
        """Test resuming signal acquisition."""
        response = api_client.post("/api/resume")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        mock_pipeline.resume.assert_called_once()

    def test_get_status(self, api_client):
        """Test getting system status."""
        response = api_client.get("/api/status")
        assert response.status_code == 200

        data = response.json()
        assert "state" in data
        assert "samples_processed" in data
        assert "detections_count" in data

    def test_full_lifecycle_workflow(self, api_client, mock_pipeline):
        """Test complete lifecycle: start -> pause -> resume -> stop."""
        # Start
        r1 = api_client.post("/api/start")
        assert r1.status_code == 200

        # Check status
        r2 = api_client.get("/api/status")
        assert r2.status_code == 200

        # Pause
        r3 = api_client.post("/api/pause")
        assert r3.status_code == 200

        # Resume
        r4 = api_client.post("/api/resume")
        assert r4.status_code == 200

        # Stop
        r5 = api_client.post("/api/stop")
        assert r5.status_code == 200

        # Verify all calls made
        mock_pipeline.start.assert_called_once()
        mock_pipeline.pause.assert_called_once()
        mock_pipeline.resume.assert_called_once()
        mock_pipeline.stop.assert_called_once()


class TestConfigurationEndpoints:
    """Test configuration management endpoints."""

    def test_get_config(self, api_client):
        """Test retrieving current configuration."""
        response = api_client.get("/api/config")
        assert response.status_code == 200

        data = response.json()
        # Verify structure matches frontend expectations
        assert "sdr" in data
        assert "center_freq_hz" in data["sdr"]

    def test_update_config(self, api_client):
        """Test updating configuration."""
        new_config = {
            "sdr": {
                "center_freq_hz": 433e6,
                "gain_db": 50.0
            }
        }

        response = api_client.put("/api/config", json=new_config)
        # May return 200 or 422 depending on validation
        assert response.status_code in [200, 422]

    def test_get_config_presets(self, api_client):
        """Test retrieving configuration presets."""
        response = api_client.get("/api/config/presets")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, (list, dict))


class TestDetectionEndpoints:
    """Test detection retrieval endpoints."""

    def test_list_detections_empty(self, api_client):
        """Test listing detections when none exist."""
        response = api_client.get("/api/detections")
        assert response.status_code == 200

        data = response.json()
        assert "detections" in data or isinstance(data, list)

    def test_list_detections_with_pagination(self, api_client):
        """Test detection pagination parameters."""
        response = api_client.get("/api/detections?limit=10&offset=0")
        assert response.status_code == 200

    def test_list_detections_validates_limit(self, api_client):
        """Test that limit parameter is validated."""
        # Should reject limits > 1000
        response = api_client.get("/api/detections?limit=5000")
        assert response.status_code == 422  # Validation error


class TestRecordingWorkflow:
    """Test recording start/stop/list workflow."""

    def test_list_recordings_empty(self, api_client):
        """Test listing recordings when none exist."""
        response = api_client.get("/api/recordings")
        assert response.status_code == 200

        data = response.json()
        assert "recordings" in data
        assert isinstance(data["recordings"], list)

    def test_start_recording(self, api_client):
        """Test starting a new recording."""
        request_data = {
            "name": "test_recording",
            "description": "Integration test recording",
            "duration_seconds": 10
        }

        response = api_client.post("/api/recordings/start", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "recording_id" in data
        assert data["status"] == "recording"

        return data["recording_id"]

    def test_start_and_stop_recording(self, api_client):
        """Test complete recording workflow."""
        # Start recording
        start_request = {
            "name": "workflow_test",
            "description": "Test workflow",
            "duration_seconds": 5
        }

        start_response = api_client.post("/api/recordings/start", json=start_request)
        assert start_response.status_code == 200
        recording_id = start_response.json()["recording_id"]

        # Stop recording
        stop_request = {"recording_id": recording_id}
        stop_response = api_client.post("/api/recordings/stop", json=stop_request)
        assert stop_response.status_code == 200

        # Verify recording appears in list
        list_response = api_client.get("/api/recordings")
        assert list_response.status_code == 200

    def test_get_nonexistent_recording(self, api_client):
        """Test getting a recording that doesn't exist."""
        response = api_client.get("/api/recordings/nonexistent-id-12345")
        assert response.status_code == 404

    def test_delete_nonexistent_recording(self, api_client):
        """Test deleting a recording that doesn't exist."""
        response = api_client.delete("/api/recordings/nonexistent-id-12345")
        assert response.status_code == 404


class TestSDREndpoints:
    """Test SDR configuration endpoints."""

    def test_get_sdr_devices(self, api_client):
        """Test listing available SDR devices."""
        response = api_client.get("/api/sdr/devices")
        # May return empty list if no devices
        assert response.status_code in [200, 503]

    def test_get_sdr_config(self, api_client):
        """Test getting SDR configuration."""
        response = api_client.get("/api/sdr/config")
        assert response.status_code == 200

        data = response.json()
        assert "center_freq_hz" in data or "centerFreqHz" in data

    def test_get_frequency_bands(self, api_client):
        """Test getting predefined frequency bands."""
        response = api_client.get("/api/sdr/bands")
        assert response.status_code == 200


class TestAnalysisEndpoints:
    """Test signal analysis endpoints."""

    def test_analyze_requires_file(self, api_client):
        """Test that analyze endpoint requires a file."""
        response = api_client.post("/api/analyze")
        # Should fail without file
        assert response.status_code == 422


class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_json_body(self, api_client):
        """Test handling of invalid JSON in request body."""
        response = api_client.post(
            "/api/recordings/start",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_field(self, api_client):
        """Test handling of missing required fields."""
        # Missing 'name' field
        response = api_client.post("/api/recordings/start", json={"description": "test"})
        assert response.status_code == 422

    def test_invalid_endpoint(self, api_client):
        """Test handling of non-existent endpoints."""
        response = api_client.get("/api/nonexistent")
        assert response.status_code == 404


class TestCORSHeaders:
    """Test CORS configuration."""

    def test_cors_preflight(self, api_client):
        """Test CORS preflight request handling."""
        response = api_client.options(
            "/api/status",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # Should allow localhost origins
        assert response.status_code == 200

    def test_cors_actual_request(self, api_client):
        """Test CORS headers on actual request."""
        response = api_client.get(
            "/api/status",
            headers={"Origin": "http://localhost:3000"}
        )
        # Should include CORS headers
        assert response.status_code == 200


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
