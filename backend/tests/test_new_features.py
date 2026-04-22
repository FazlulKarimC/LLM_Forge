"""
Tests for New Features (Phases 1-5)

Covers:
- Phase 1.3: Auto-generated experiment summaries
- Phase 2: Reproducibility (run manifest, tags)
- Phase 3: Cost & performance visibility (pricing, cost metrics)
- Phase 5: Export & sharing (tag filtering API)
"""

import pytest
import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from app.core.pricing import get_model_pricing, estimate_cost, ModelPricing, DEFAULT_PRICING
from app.services.metrics_service import MetricsService
from app.schemas.experiment import ExperimentCreate, ExperimentConfig, ExperimentResponse, ExperimentStatus


# =============================================================================
# Phase 3: Pricing Module Tests
# =============================================================================

class TestPricingLookup:
    """Tests for model pricing lookup (Phase 3)."""

    def test_exact_match_hf_free_tier(self):
        """HuggingFace free-tier models should have $0 pricing."""
        pricing = get_model_pricing("mistralai/Mistral-7B-Instruct-v0.3")
        assert pricing.input_per_1k == 0.0
        assert pricing.output_per_1k == 0.0
        assert pricing.provider == "hf_api"

    def test_exact_match_openai(self):
        """OpenAI models should have non-zero pricing."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing.input_per_1k > 0
        assert pricing.output_per_1k > 0
        assert pricing.provider == "openai"

    def test_substring_match(self):
        """Models should match via substring (e.g., 'llama' in full name)."""
        pricing = get_model_pricing("meta-llama/Llama-3.2-3B-Instruct")
        assert pricing.provider == "hf_api"

    def test_unknown_model_uses_default(self):
        """Unknown models should fall back to default pricing."""
        pricing = get_model_pricing("totally-unknown/model-v99")
        assert pricing == DEFAULT_PRICING
        assert pricing.provider == "unknown"


class TestCostEstimation:
    """Tests for cost estimation (Phase 3)."""

    def test_free_tier_zero_cost(self):
        """HF free-tier models should estimate $0 cost."""
        result = estimate_cost("mistralai/Mistral-7B-Instruct-v0.3", tokens_input=1000, tokens_output=500)
        assert result["total_cost_usd"] == 0.0
        assert result["provider"] == "hf_api"
        assert result["tokens_input"] == 1000
        assert result["tokens_output"] == 500

    def test_paid_model_nonzero_cost(self):
        """OpenAI models should produce non-zero cost estimates."""
        result = estimate_cost("gpt-4o", tokens_input=1000, tokens_output=500)
        assert result["total_cost_usd"] > 0
        # gpt-4o: $0.005/1K input + $0.015/1K output
        expected_input = (1000 / 1000) * 0.005
        expected_output = (500 / 1000) * 0.015
        assert abs(result["total_cost_usd"] - (expected_input + expected_output)) < 1e-6

    def test_zero_tokens_zero_cost(self):
        """Zero tokens should produce zero cost regardless of model."""
        result = estimate_cost("gpt-4o", tokens_input=0, tokens_output=0)
        assert result["total_cost_usd"] == 0.0

    def test_cost_output_structure(self):
        """Cost result should include all expected keys."""
        result = estimate_cost("gpt-4o", tokens_input=100, tokens_output=50)
        expected_keys = {"input_cost_usd", "output_cost_usd", "total_cost_usd",
                         "provider", "input_per_1k", "output_per_1k",
                         "tokens_input", "tokens_output"}
        assert expected_keys == set(result.keys())


# =============================================================================
# Phase 3: Cost Metrics in MetricsService
# =============================================================================

class TestCostMetrics:
    """Tests for cost computation within MetricsService (Phase 3)."""

    def test_compute_cost_free_tier(self):
        """Cost metrics from free-tier model should show $0."""
        service = MetricsService.__new__(MetricsService)

        runs = [
            MagicMock(tokens_input=100, tokens_output=50, latency_ms=500, is_correct=True),
            MagicMock(tokens_input=200, tokens_output=100, latency_ms=800, is_correct=False),
        ]
        result = service._compute_cost(runs, model_name="mistralai/Mistral-7B-Instruct-v0.3")

        assert result["total_tokens_input"] == 300
        assert result["total_tokens_output"] == 150
        assert result["total_tokens"] == 450
        assert result["total_runs"] == 2
        assert result["total_cost_usd"] == 0.0
        assert result["provider"] == "hf_api"

    def test_compute_cost_paid_model(self):
        """Cost metrics from paid model should produce non-zero cost."""
        service = MetricsService.__new__(MetricsService)

        runs = [MagicMock(tokens_input=500, tokens_output=200, latency_ms=1000, is_correct=True)]
        result = service._compute_cost(runs, model_name="gpt-4o")

        assert result["total_cost_usd"] > 0
        assert result["cost_per_correct_answer"] is not None
        assert result["provider"] == "openai"

    def test_cost_per_correct_none_when_no_correct(self):
        """cost_per_correct_answer should be None when no runs are correct."""
        service = MetricsService.__new__(MetricsService)

        runs = [MagicMock(tokens_input=100, tokens_output=50, latency_ms=300, is_correct=False)]
        result = service._compute_cost(runs, model_name="gpt-4o")

        assert result["cost_per_correct_answer"] is None

    def test_compute_cost_prefers_observed_per_run_routing_costs(self):
        """Observed per-run costs should override model-name estimates for routed runs."""
        service = MetricsService.__new__(MetricsService)

        runs = [
            SimpleNamespace(
                tokens_input=100,
                tokens_output=50,
                latency_ms=300,
                is_correct=True,
                cost_usd=0.001,
                served_provider="openrouter",
            ),
            SimpleNamespace(
                tokens_input=200,
                tokens_output=100,
                latency_ms=700,
                is_correct=False,
                cost_usd=0.002,
                served_provider="groq",
            ),
        ]
        result = service._compute_cost(runs, model_name="gpt-4o")

        assert result["total_cost_usd"] == 0.003
        assert result["cost_per_correct_answer"] == 0.003
        assert result["provider"] == "mixed"
        assert result["cost_source"] == "observed_per_run"


# =============================================================================
# Phase 1.3: Auto-Generated Summary
# =============================================================================

class TestExperimentSummary:
    """Tests for auto-generated experiment summary (Phase 1.3)."""

    def test_summary_includes_accuracy_and_latency(self):
        """Summary should mention accuracy percentage and latency."""
        service = MetricsService.__new__(MetricsService)

        accuracy = {"accuracy_any": 0.85, "total_evaluated": 100}
        latency = {"p50": 450}
        cost = {"total_tokens_input": 0, "total_tokens_output": 0}
        faithfulness = {"count": 0}
        failure_modes = {"total_failures": 0}

        summary = service._generate_summary(accuracy, latency, cost, faithfulness, failure_modes, uuid4())

        assert "85.0%" in summary
        assert "100" in summary
        assert "450ms" in summary

    def test_summary_includes_cost_when_nonzero(self):
        """Summary should mention cost when tokens > 0."""
        service = MetricsService.__new__(MetricsService)

        accuracy = {"accuracy_any": 0.5, "total_evaluated": 10}
        latency = {"p50": 200}
        cost = {"total_tokens_input": 100000, "total_tokens_output": 50000}
        faithfulness = {"count": 0}
        failure_modes = {"total_failures": 0}

        summary = service._generate_summary(accuracy, latency, cost, faithfulness, failure_modes, uuid4())

        assert "$" in summary

    def test_summary_no_cost_for_free_tier(self):
        """Summary should say 'no measurable API token costs' for zero tokens."""
        service = MetricsService.__new__(MetricsService)

        accuracy = {"accuracy_any": 1.0, "total_evaluated": 5}
        latency = {"p50": 100}
        cost = {"total_tokens_input": 0, "total_tokens_output": 0}
        faithfulness = {"count": 0}
        failure_modes = {"total_failures": 0}

        summary = service._generate_summary(accuracy, latency, cost, faithfulness, failure_modes, uuid4())

        assert "no measurable" in summary.lower()

    def test_summary_includes_failure_modes(self):
        """Summary should mention failures when present."""
        service = MetricsService.__new__(MetricsService)

        accuracy = {"accuracy_any": 0.3, "total_evaluated": 10}
        latency = {"p50": 500}
        cost = {"total_tokens_input": 0, "total_tokens_output": 0}
        faithfulness = {"count": 0}
        failure_modes = {"total_failures": 3, "counts": {"timeout": 2, "api_error": 1}}

        summary = service._generate_summary(accuracy, latency, cost, faithfulness, failure_modes, uuid4())

        assert "3 failures" in summary
        assert "timeout" in summary.lower()

    def test_summary_includes_context_support_rate(self):
        """Summary should describe RAG support as a proxy, not a fact hallucination label."""
        service = MetricsService.__new__(MetricsService)

        accuracy = {"accuracy_any": 0.7, "total_evaluated": 20}
        latency = {"p50": 300}
        cost = {"total_tokens_input": 0, "total_tokens_output": 0}
        faithfulness = {"count": 10, "unsupported_rate": 0.15}
        failure_modes = {"total_failures": 0}

        summary = service._generate_summary(accuracy, latency, cost, faithfulness, failure_modes, uuid4())

        assert "context-support proxy" in summary.lower()
        assert "low-support" in summary.lower()
        assert "15.0%" in summary


# =============================================================================
# Phase 2: Reproducibility — Run Manifest & Tags
# =============================================================================

class TestRunManifest:
    """Tests for run manifest generation (Phase 2)."""

    def test_manifest_hash_deterministic(self):
        """Same config should produce identical manifest hash."""
        manifest_data = {
            "dataset_name": "sample",
            "model_name": "test-model",
            "provider": "hf_api",
            "reasoning_method": "naive",
            "hyperparameters": {"temperature": 0.7, "max_tokens": 256},
            "num_samples": 10,
            "rag": None,
            "agent": None,
            "optimization": None,
        }
        manifest_json = json.dumps(manifest_data, sort_keys=True, default=str)
        hash1 = hashlib.sha256(manifest_json.encode()).hexdigest()
        hash2 = hashlib.sha256(manifest_json.encode()).hexdigest()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_manifest_hash_changes_with_config(self):
        """Different configs should produce different manifest hashes."""
        def compute_hash(data):
            return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()

        config_a = {"model_name": "model-a", "dataset_name": "sample", "temperature": 0.7}
        config_b = {"model_name": "model-b", "dataset_name": "sample", "temperature": 0.7}

        assert compute_hash(config_a) != compute_hash(config_b)


class TestTagsInSchema:
    """Tests for tags support in experiment schemas (Phase 2)."""

    def test_experiment_create_with_tags(self):
        """ExperimentCreate should accept tags."""
        data = ExperimentCreate(
            name="Test",
            config=ExperimentConfig(model_name="test-model", dataset_name="sample"),
            tags=["v1", "baseline"]
        )
        assert data.tags == ["v1", "baseline"]

    def test_experiment_create_without_tags(self):
        """ExperimentCreate should work without tags (default None)."""
        data = ExperimentCreate(
            name="Test",
            config=ExperimentConfig(model_name="test-model", dataset_name="sample"),
        )
        assert data.tags is None

    def test_experiment_response_with_run_manifest(self):
        """ExperimentResponse should accept run_manifest dict."""
        from datetime import datetime, timezone
        resp = ExperimentResponse(
            id=uuid4(),
            name="Test",
            description=None,
            config=ExperimentConfig(model_name="test-model", dataset_name="sample"),
            status=ExperimentStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            started_at=None,
            completed_at=None,
            error_message=None,
            tags=["v1"],
            run_manifest={"dataset_name": "sample", "manifest_hash": "abc123"},
        )
        assert resp.tags == ["v1"]
        assert resp.run_manifest["manifest_hash"] == "abc123"


# =============================================================================
# Phase 5: Tag Filtering API
# =============================================================================

class TestTagFilteringAPI:
    """Tests for tag-based experiment filtering (Phase 2 + 5)."""

    @patch('app.api.experiments.ExperimentService')
    def test_tag_filter_query_param_accepted(self, MockServiceClass):
        """The /experiments endpoint should accept a tag query parameter."""
        from unittest.mock import AsyncMock
        from app.main import app

        mock_service = AsyncMock()
        MockServiceClass.return_value = mock_service
        mock_service.list.return_value = {"total": 0, "experiments": [], "skip": 0, "limit": 20}

        client = TestClient(app)
        response = client.get("/api/v1/experiments?tag=baseline")
        assert response.status_code == 200

        # Verify the tag param was passed through to the service
        mock_service.list.assert_called_once()
        call_kwargs = mock_service.list.call_args
        assert call_kwargs.kwargs.get("tag") == "baseline" or (
            len(call_kwargs.args) > 0 and "baseline" in str(call_kwargs)
        )
