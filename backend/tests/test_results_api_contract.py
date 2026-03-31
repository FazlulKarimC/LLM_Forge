"""
Contract tests for result payloads consumed by the frontend.
"""

import uuid
from datetime import datetime, timezone

from app.api.results import _result_to_metrics_response
from app.models.result import Result
from app.models.run import Run
from app.schemas.result import RunSummary


class TestMetricsResponseContract:
    def test_metrics_response_includes_summary_failure_modes_and_cost_metadata(self):
        experiment_id = uuid.uuid4()
        result = Result(
            experiment_id=experiment_id,
            accuracy_exact=0.5,
            accuracy_f1=0.6,
            accuracy_substring=0.7,
            latency_p50=100.0,
            latency_p95=150.0,
            latency_p99=200.0,
            throughput=2.5,
            total_tokens_input=10,
            total_tokens_output=20,
            total_runs=2,
            gpu_time_seconds=1.2,
            raw_metrics={
                "summary_text": "Summary",
                "failure_modes": {"counts": {"api_error": 1}, "total_failures": 1, "sample_errors": []},
                "cost": {
                    "total_cost_usd": 0.0123,
                    "cost_per_correct_answer": 0.00615,
                    "provider": "openrouter",
                },
            },
            computed_at=datetime.now(timezone.utc),
        )

        response = _result_to_metrics_response(result)

        assert response.summary_text == "Summary"
        assert response.failure_modes["total_failures"] == 1
        assert response.cost.total_cost_usd == 0.0123
        assert response.cost.cost_per_correct_answer == 0.00615
        assert response.cost.provider == "openrouter"


class TestRunSummaryContract:
    def test_run_summary_includes_rag_chunks_grader_results_and_served_provider(self):
        run = Run(
            experiment_id=uuid.uuid4(),
            example_id="ex-1",
            prompt="Prompt",
            raw_output="Output",
            expected_output="Expected",
            is_correct=True,
            score=1.0,
            latency_ms=123.0,
            attempt=2,
            retrieved_chunks={"chunks": [{"text": "chunk text", "score": 0.9}]},
            grader_results={"latency_budget": {"status": "pass", "reason": "ok"}},
            served_provider="OpenRouterEngine",
        )

        summary = RunSummary.model_validate(run)

        assert summary.retrieved_chunks["chunks"][0]["text"] == "chunk text"
        assert summary.grader_results["latency_budget"]["status"] == "pass"
        assert summary.served_provider == "OpenRouterEngine"
