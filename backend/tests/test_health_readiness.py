"""Core readiness never probes optional external services."""
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.mark.parametrize("database,expected", [("healthy", 200), ("unhealthy: schema missing", 503)])
def test_core_readiness_is_independent_of_optional_dependencies(database, expected):
    with patch("app.services.evaluation_runner.runner.start", new_callable=AsyncMock), \
         patch("app.api.health._check_database", new_callable=AsyncMock, return_value=database), \
         patch("app.api.health._check_vector_db", new_callable=AsyncMock) as vector, \
         patch("app.api.health._check_dispatch", new_callable=AsyncMock) as dispatch, \
         patch("app.services.evaluation_runner.runner.health", return_value="unhealthy"), \
         TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == expected
        assert response.json()["checks"]["database"] == database
        vector.assert_not_awaited()
        dispatch.assert_not_awaited()
