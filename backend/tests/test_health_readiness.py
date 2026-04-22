"""
Tests for the /ready endpoint with dispatch readiness checks.

Covers:
- Readiness reports task_dispatch status
- Readiness handles fallback_inline
- Readiness handles circuit_open
- Readiness handles worker_missing
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestReadinessDispatch:
    """Tests for /ready endpoint dispatch status reporting."""

    def test_readiness_includes_dispatch_keys(self, client):
        """The /ready endpoint should include task_dispatch, upstash, rq_worker keys."""
        with patch("app.api.health._check_database", return_value="healthy"), \
             patch("app.api.health._check_vector_db", return_value="not_configured"), \
             patch("app.api.health._check_models", return_value="not_configured"), \
             patch("app.api.health._check_dispatch", return_value={
                 "task_dispatch": "healthy",
                 "upstash": "healthy",
                 "rq_worker": "healthy",
             }):
            response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        checks = data["checks"]
        assert "task_dispatch" in checks
        assert "upstash" in checks
        assert "rq_worker" in checks

    def test_readiness_fallback_inline(self, client):
        """Inline fallback should still be ready because core execution still works."""
        with patch("app.api.health._check_database", return_value="healthy"), \
             patch("app.api.health._check_vector_db", return_value="not_configured"), \
             patch("app.api.health._check_models", return_value="healthy"), \
             patch("app.api.health._check_dispatch", return_value={
                  "task_dispatch": "fallback_inline",
                  "upstash": "not_configured",
                  "rq_worker": "not_configured",
              }):
            response = client.get("/ready")

        data = response.json()
        assert data["status"] == "ready"
        assert data["mode"] == "degraded"
        assert data["checks"]["task_dispatch"] == "fallback_inline"

    def test_readiness_circuit_open(self, client):
        """When the circuit is open, upstash status should show circuit_open."""
        with patch("app.api.health._check_database", return_value="healthy"), \
             patch("app.api.health._check_vector_db", return_value="not_configured"), \
             patch("app.api.health._check_models", return_value="not_configured"), \
             patch("app.api.health._check_dispatch", return_value={
                 "task_dispatch": "fallback_inline",
                 "upstash": "circuit_open",
                 "rq_worker": "healthy",
             }):
            response = client.get("/ready")

        data = response.json()
        assert data["checks"]["upstash"] == "circuit_open"
        assert data["checks"]["task_dispatch"] == "fallback_inline"

    def test_readiness_worker_missing(self, client):
        """Missing worker should be degraded but ready when inline fallback is available."""
        with patch("app.api.health._check_database", return_value="healthy"), \
             patch("app.api.health._check_vector_db", return_value="not_configured"), \
             patch("app.api.health._check_models", return_value="healthy"), \
             patch("app.api.health._check_dispatch", return_value={
                  "task_dispatch": "fallback_inline",
                  "upstash": "healthy",
                  "rq_worker": "worker_missing",
              }):
            response = client.get("/ready")

        data = response.json()
        assert data["status"] == "ready"
        assert data["mode"] == "degraded"
        assert data["checks"]["rq_worker"] == "worker_missing"

    def test_readiness_all_healthy_returns_ready(self, client):
        """When all checks are healthy, status should be 'ready'."""
        with patch("app.api.health._check_database", return_value="healthy"), \
             patch("app.api.health._check_vector_db", return_value="healthy"), \
             patch("app.api.health._check_models", return_value="healthy"), \
             patch("app.api.health._check_dispatch", return_value={
                 "task_dispatch": "healthy",
                 "upstash": "healthy",
                 "rq_worker": "healthy",
             }):
            response = client.get("/ready")

        data = response.json()
        assert data["status"] == "ready"
        assert data["mode"] == "healthy"
