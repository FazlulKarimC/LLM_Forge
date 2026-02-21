"""
API Tests

Tests for experiment and results API endpoints.

TODO (Iteration 1): Add experiment CRUD tests
TODO (Iteration 2): Add results and metrics tests
TODO (Iteration 3): Add integration tests with database
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


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
    
    def test_readiness_check(self, client):
        """Test readiness check returns status."""
        response = client.get("/ready")
        assert response.status_code == 200
        assert "checks" in response.json()


class TestExperimentEndpoints:
    """Tests for experiment API endpoints."""
    
    def test_create_experiment(self, client):
        """Test creating an experiment."""
        # Basic skeleton for future expansion
        pass
    
    def test_list_experiments(self, client):
        """Test listing experiments."""
        response = client.get("/api/v1/experiments")
        assert response.status_code in [200, 404], "Ensure endpoint exists or is simply empty"
    
    def test_get_experiment(self, client):
        """Test getting experiment by ID."""
        pass


class TestResultEndpoints:
    """Tests for results API endpoints."""
    
    def test_get_results(self, client):
        """Test getting results for an experiment."""
        pass
    
    def test_get_metrics(self, client):
        """Test getting metrics for an experiment."""
        pass
