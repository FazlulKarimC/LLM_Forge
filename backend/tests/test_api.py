"""
API Tests

Updated to focus on critical features:
- Health checks
- Available models (HF Integration)
- Global Error Handling (422 Validations, 404 AppExceptions, request_ids)
"""

import pytest
from uuid import uuid4
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c

class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

class TestExperimentCoreEndpoints:
    """Tests for critical experiment features and error handling."""
    
    def test_list_available_models(self, client):
        """Test the curated models endpoint (HF Integration)."""
        response = client.get(f"{settings.API_V1_PREFIX}/experiments/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0
        assert "value" in data["models"][0]
        assert "meta-llama" in data["models"][0]["value"]
    
    def test_custom_422_validation_error(self, client):
        """
        Verify our global `validation_exception_handler` intercepts
        Pydantic errors and formats them securely with `request_id`.
        """
        # POST an incomplete payload to trigger 422
        bad_payload = {"name": "Test Env"}
        response = client.post(f"{settings.API_V1_PREFIX}/experiments/", json=bad_payload)
        
        assert response.status_code == 422
        assert "X-Request-ID" in response.headers
        
        data = response.json()
        assert data["error"] is True
        assert data["status_code"] == 422
        assert "request_id" in data
        assert isinstance(data["details"], list)
        assert len(data["details"]) > 0
        assert "config" in data["details"][0]["field"]
    
    def test_custom_app_exception_404(self, client):
        """
        Verify our base `AppException` (ResourceNotFound) handler formats
        errors into standardized JSON with context and `request_id`.
        """
        fake_uuid = str(uuid4())
        response = client.get(f"{settings.API_V1_PREFIX}/experiments/{fake_uuid}")
        
        assert response.status_code == 404
        assert "X-Request-ID" in response.headers
        
        data = response.json()
        assert data["error"] is True
        assert data["status_code"] == 404
        assert "request_id" in data
        assert "not found" in data["message"].lower()

    def test_missing_endpoint_404(self, client):
        """Test random missing endpoint behavior."""
        response = client.get("/api/v1/does-not-exist")
        assert response.status_code == 404

from unittest.mock import patch, MagicMock

class TestExperimentRunCustomHeaders:
    """Tests for starting an experiment with custom headers."""

    @patch('app.api.experiments.ExperimentService.get')
    @patch('app.api.experiments.ExperimentService.update_status')
    @patch('app.api.experiments._enqueue_or_fallback')
    def test_run_experiment_with_custom_headers(self, mock_enqueue, mock_update, mock_get, client):
        """Test /run endpoint parses the custom headers correctly."""
        class MockHyperParams:
            temperature = 0.1
            max_tokens = 10
            top_p = 0.9
            top_k = None
            seed = 42
        
        class MockConfig:
            model_name = "custom_hosted"
            reasoning_method = "naive"
            dataset_name = "sample"
            num_samples = 2
            rag = None
            agent = None
            optimization = None
            hyperparameters = MockHyperParams()
            
        class MockExperiment:
            id = "00000000-0000-0000-0000-000000000000"
            name = "Test"
            description = ""
            status = "pending"
            error_message = None
            started_at = None
            completed_at = None
            created_at = "2025-01-01T00:00:00Z"
            updated_at = "2025-01-01T00:00:00Z"
            config = MockConfig()

        mock_exp = MockExperiment()
        
        # When run_experiment calls service.get() it returns our mock
        mock_get.return_value = mock_exp

        # 2. Run the experiment with headers
        headers = {
            "X-Custom-LLM-Base": "http://mock-base.local/v1",
            "X-Custom-LLM-Key": "mock-api-key"
        }
        
        run_resp = client.post(
            f"{settings.API_V1_PREFIX}/experiments/{mock_exp.id}/run",
            headers=headers
        )
        assert run_resp.status_code == 200, run_resp.text
        
        # Verify enqueue was called with custom headers
        mock_enqueue.assert_called_once()
        kwargs = mock_enqueue.call_args.kwargs
        assert kwargs.get("custom_base_url") == "http://mock-base.local/v1"
        assert kwargs.get("custom_api_key") == "mock-api-key"
