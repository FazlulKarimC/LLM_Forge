"""
Targeted safety tests for execution lifecycle fixes.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.api.experiments import _active_run_count
from app.schemas.experiment import ExperimentConfig, ExperimentResponse, ExperimentStatus
from app.services.experiment_service import ExperimentService


def test_active_run_count_fails_closed_on_stats_error():
    service = AsyncMock()
    service.get_stats.side_effect = RuntimeError("db down")

    count = asyncio.run(_active_run_count(service))

    from app.core.rate_limit import MAX_CONCURRENT_RUNS

    assert count == MAX_CONCURRENT_RUNS


def test_execute_does_not_clear_results_before_rerun():
    import pytest

    experiment_id = uuid4()
    now = datetime.now(timezone.utc)
    experiment = ExperimentResponse(
        id=experiment_id,
        name="Rerun Safety",
        description=None,
        config=ExperimentConfig(
            model_name="mock-model",
            reasoning_method="naive",
            dataset_name="sample",
            num_samples=1,
        ),
        status=ExperimentStatus.COMPLETED,
        created_at=now,
        started_at=None,
        completed_at=now,
        error_message=None,
        tags=[],
        run_manifest=None,
    )

    db = AsyncMock()
    max_attempt_result = MagicMock()
    max_attempt_result.scalar.return_value = 0
    experiment_row_result = MagicMock()
    experiment_row_result.scalar_one_or_none.return_value = SimpleNamespace(
        current_attempt=1,
        regression_status="not_checked",
        regression_passed=None,
    )
    db.execute.side_effect = [max_attempt_result, experiment_row_result]
    db.flush = AsyncMock()
    db.commit = AsyncMock()

    service = ExperimentService(db)
    service.get = AsyncMock(return_value=experiment)
    service.update_status = AsyncMock(return_value=experiment)

    with patch("app.services.metrics_service.MetricsService.clear_results", new_callable=AsyncMock) as clear_results:
        with patch("app.services.dataset_service.DatasetService.load", side_effect=RuntimeError("dataset boom")):
            with pytest.raises(RuntimeError, match="dataset boom"):
                asyncio.run(service.execute(experiment_id))

    clear_results.assert_not_awaited()
