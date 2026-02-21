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
