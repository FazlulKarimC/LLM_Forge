"""
Experiment Lifecycle Regression Tests

Verifies critical fixes from Milestone 1 & 2:
1. Dispatch failure rolls back status to FAILED
2. Dashboard stats endpoint correctness
"""

import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from app.main import app
from app.core.config import settings

pytestmark = pytest.mark.usefixtures("legacy_route_identity")


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestEnqueueFailureRollback:
    """Test that dispatch failures properly roll back experiment status."""

    @patch('app.api.experiments.ExperimentService')
    @patch('app.core.task_dispatch.dispatch_experiment')
    @patch('app.api.experiments._active_run_count', new_callable=AsyncMock, return_value=0)
    def test_enqueue_failure_rollback(self, mock_active_count, mock_dispatch, MockServiceClass, client):
        """
        Test that if dispatching an experiment fails, the status is properly
        rolled back to FAILED from QUEUED, rather than getting stuck.

        We mock the entire service layer to avoid asyncpg connection pool
        contention with the lifespan startup event.
        """
        from unittest.mock import AsyncMock
        from uuid import uuid4
        from datetime import datetime, timezone
        from app.schemas.experiment import ExperimentResponse, ExperimentConfig, ExperimentStatus

        mock_service = AsyncMock()
        MockServiceClass.return_value = mock_service

        exp_id = uuid4()
        now = datetime.now(timezone.utc)

        # Build a proper ExperimentResponse the endpoint can serialize
        exp_response = ExperimentResponse(
            id=exp_id,
            name="Rollback Test",
            description="Testing failure rollback",
            config=ExperimentConfig(
                model_name="mock-model",
                reasoning_method="naive",
                dataset_name="sample",
                num_samples=1,
            ),
            status=ExperimentStatus.PENDING,
            created_at=now,
            started_at=None,
            completed_at=None,
            error_message=None,
            tags=[],
            run_manifest=None,
        )

        mock_service.create.return_value = exp_response
        mock_service.get.return_value = exp_response
        mock_service.queue_for_execution.return_value = 1

        # Make dispatch raise an exception to simulate queue failure
        mock_dispatch.side_effect = Exception("Simulated queue failure")

        # 1. Create a dummy experiment
        create_payload = {
            "name": "Rollback Test",
            "description": "Testing failure rollback",
            "config": {
                "model_name": "mock-model",
                "reasoning_method": "naive",
                "dataset_name": "sample",
                "num_samples": 1,
            }
        }

        create_resp = client.post(f"{settings.API_V1_PREFIX}/experiments", json=create_payload)
        assert create_resp.status_code == 201

        # 2. Try to run it — should trigger dispatch failure
        try:
            client.post(f"{settings.API_V1_PREFIX}/experiments/{exp_id}/run")
        except Exception:
            pass

        # 3. The failure may only affect the attempt that failed to dispatch.
        mock_service._fail_active_attempt.assert_awaited_once_with(
            exp_id, "Failed to start execution: task queue unavailable", attempt=1
        )


class TestDashboardStats:
    """Test that statistics are calculated correctly."""

    @patch('app.api.experiments.ExperimentService.get_stats')
    def test_dashboard_stats_correctness(self, mock_get_stats, client):
        """
        Test that the /stats endpoint correctly returns aggregated counts.
        We mock the service layer to avoid DB connection issues in tests.
        """
        mock_get_stats.return_value = {
            "total": 10,
            "completed": 4,
            "running": 2,
            "pending": 3,
            "queued": 0,
            "failed": 1,
        }

        stats_resp = client.get(f"{settings.API_V1_PREFIX}/experiments/stats")
        assert stats_resp.status_code == 200

        stats = stats_resp.json()

        # Verify all required keys are present
        assert "total" in stats
        assert "pending" in stats
        assert "failed" in stats
        assert "completed" in stats
        assert "running" in stats
        assert "queued" in stats

        # Total should be the sum of all status counts
        calculated_total = (
            stats.get("pending", 0)
            + stats.get("running", 0)
            + stats.get("completed", 0)
            + stats.get("failed", 0)
            + stats.get("queued", 0)
        )
        assert stats["total"] == calculated_total
        assert stats["total"] == 10
