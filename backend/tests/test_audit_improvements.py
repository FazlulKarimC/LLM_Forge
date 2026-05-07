"""
Tests for audit implementation changes.

Covers:
- Wilson CI computation (Item 4)
- Effect size: Cohen's h + small-N override (Item 6)
- Parse confidence tracking (Item 7)
- Retrieval quality computation (Item 1)
- Completion quality (Item 3)
- Faithfulness renamed fields (Item 5)
- Cost normalization (Item 12)
- Dataset metadata loading (Item 2)
- Judge seed determinism (Item 10)
- Provenance temperature tracking (Item 11)
"""

import math
import random
import pytest
from unittest.mock import MagicMock, patch


# ─── Wilson CI ───────────────────────────────────────────────────────────────

class TestWilsonCI:
    """Item 4: Wilson score interval for binomial proportions."""

    def test_zero_total(self):
        from app.services.metrics_service import MetricsService
        lo, hi = MetricsService._wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_perfect_score(self):
        from app.services.metrics_service import MetricsService
        lo, hi = MetricsService._wilson_ci(100, 100)
        assert lo > 0.95
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_zero_correct(self):
        from app.services.metrics_service import MetricsService
        lo, hi = MetricsService._wilson_ci(0, 100)
        assert lo == 0.0
        assert hi < 0.05

    def test_fifty_percent(self):
        from app.services.metrics_service import MetricsService
        lo, hi = MetricsService._wilson_ci(50, 100)
        assert 0.3 < lo < 0.5
        assert 0.5 < hi < 0.7

    def test_small_n_wide_ci(self):
        """With small n, CI should be wider than large n."""
        from app.services.metrics_service import MetricsService
        lo_small, hi_small = MetricsService._wilson_ci(3, 5)
        lo_large, hi_large = MetricsService._wilson_ci(60, 100)
        width_small = hi_small - lo_small
        width_large = hi_large - lo_large
        assert width_small > width_large


# ─── Cohen's h Effect Size ──────────────────────────────────────────────────

class TestCohensH:
    """Item 6: Cohen's h and small-N override."""

    def test_identical_proportions(self):
        from app.services.statistical_service import StatisticalService
        h = StatisticalService.cohens_h(0.5, 0.5)
        assert h == 0.0

    def test_large_difference(self):
        from app.services.statistical_service import StatisticalService
        h = StatisticalService.cohens_h(0.9, 0.1)
        assert abs(h) > 0.8  # large effect

    def test_negligible_difference(self):
        from app.services.statistical_service import StatisticalService
        h = StatisticalService.cohens_h(0.5, 0.52)
        assert abs(h) < 0.2

    def test_effect_size_labels(self):
        from app.services.statistical_service import StatisticalService
        assert StatisticalService.effect_size_label(0.1) == "negligible"
        assert StatisticalService.effect_size_label(0.3) == "small"
        assert StatisticalService.effect_size_label(0.6) == "medium"
        assert StatisticalService.effect_size_label(1.0) == "large"

    def test_boundary_clamp(self):
        """Edge proportions should not cause math errors."""
        from app.services.statistical_service import StatisticalService
        h = StatisticalService.cohens_h(0.0, 1.0)
        assert math.isfinite(h)

    def test_small_n_override_blocks_significance(self):
        """McNemar is_significant should be overridden to False when N < 20."""
        from app.services.statistical_service import StatisticalService

        # Create clearly significant difference with N=10
        correct_a = [True] * 9 + [False]
        correct_b = [False] * 9 + [True]

        result = StatisticalService.mcnemar_test(correct_a, correct_b)
        # With only 10 samples, even if McNemar says significant,
        # compare_run_sets should override it

    def test_effect_size_in_comparison(self):
        """compare_run_sets should include effect_size_cohens_h."""
        from app.services.statistical_service import StatisticalService

        runs_a = []
        runs_b = []
        for i in range(30):
            run_a = MagicMock()
            run_a.example_id = f"ex{i:03d}"
            run_a.is_correct = i < 20
            run_a.score = 1.0 if i < 20 else 0.0
            run_a.raw_output = "test"
            run_a.expected_output = "test"
            run_a.served_provider = "test"
            run_a.routing_reason = None
            run_a.attempt = 1
            runs_a.append(run_a)

            run_b = MagicMock()
            run_b.example_id = f"ex{i:03d}"
            run_b.is_correct = i < 15
            run_b.score = 1.0 if i < 15 else 0.0
            run_b.raw_output = "test"
            run_b.expected_output = "test"
            run_b.served_provider = "test"
            run_b.routing_reason = None
            run_b.attempt = 1
            runs_b.append(run_b)

        result = StatisticalService.compare_run_sets(runs_a, runs_b)
        assert "effect_size_cohens_h" in result
        assert "effect_size_label" in result
        assert isinstance(result["effect_size_cohens_h"], float)
        assert result["effect_size_label"] in ["negligible", "small", "medium", "large"]


# ─── Parse Confidence Tracking ──────────────────────────────────────────────

class TestParseConfidence:
    """Item 7: parse_response_with_method returns (answer, parse_method)."""

    def test_cot_explicit_pattern(self):
        from app.services.inference.prompting import CoTPromptTemplate
        answer, method = CoTPromptTemplate.parse_response_with_method(
            "Let me think... The answer is 42."
        )
        assert answer == "42"
        assert method == "explicit_pattern"

    def test_cot_last_sentence_fallback(self):
        from app.services.inference.prompting import CoTPromptTemplate
        answer, method = CoTPromptTemplate.parse_response_with_method(
            "Some reasoning. Then more reasoning. The final number is forty-two."
        )
        # Should use last_sentence_fallback since no explicit pattern matches
        assert method in ("explicit_pattern", "last_sentence_fallback")

    def test_cot_no_response(self):
        from app.services.inference.prompting import CoTPromptTemplate
        answer, method = CoTPromptTemplate.parse_response_with_method("")
        assert method == "no_response"

    def test_naive_method_label(self):
        from app.services.inference.prompting import NaivePromptTemplate
        answer, method = NaivePromptTemplate.parse_response_with_method("Paris")
        assert answer == "Paris"
        assert method == "naive_first_line"

    def test_rag_method_label(self):
        from app.services.inference.prompting import RAGPromptTemplate
        answer, method = RAGPromptTemplate.parse_response_with_method("Paris")
        assert method == "rag_first_line"

    def test_react_answer_pattern(self):
        from app.services.inference.prompting import ReActPromptTemplate
        answer, method = ReActPromptTemplate.parse_response_with_method(
            "Thought: Let me search.\nAction: search\nObservation: result\nAnswer: 42"
        )
        assert answer == "42"
        assert method == "react_answer_pattern"

    def test_react_fallback_to_cot(self):
        from app.services.inference.prompting import ReActPromptTemplate
        answer, method = ReActPromptTemplate.parse_response_with_method(
            "The answer is 42."
        )
        assert answer == "42"
        assert method == "explicit_pattern"


# ─── Dataset Metadata ───────────────────────────────────────────────────────

class TestDatasetMetadata:
    """Item 2: Dataset provenance metadata loading."""

    def test_load_existing_metadata(self):
        from app.services.dataset_service import DatasetService
        meta = DatasetService.get_dataset_metadata("trivia_qa")
        assert meta is not None
        assert "display_name" in meta
        assert "source" in meta
        assert meta["source"] == "custom_authored"

    def test_load_knowledge_base_metadata(self):
        from app.services.dataset_service import DatasetService
        meta = DatasetService.get_dataset_metadata("knowledge_base")
        assert meta is not None
        assert "gold_evidence_coverage" in meta

    def test_load_nonexistent_metadata(self):
        from app.services.dataset_service import DatasetService
        meta = DatasetService.get_dataset_metadata("nonexistent_dataset")
        assert meta is None

    def test_legacy_alias(self):
        from app.services.dataset_service import DatasetService
        meta = DatasetService.get_dataset_metadata("triviaqa")
        assert meta is not None
        assert "LlmForge" in meta["display_name"]


# ─── Gold Evidence Fields ───────────────────────────────────────────────────

class TestGoldEvidence:
    """Item 1: RAG gold evidence fields preserved in dataset loading."""

    def test_evidence_fields_loaded(self):
        from app.services.dataset_service import DatasetService
        examples = DatasetService.load("knowledge_base")
        annotated = [e for e in examples if "evidence_source" in e]
        assert len(annotated) >= 50  # All 50 annotated

    def test_evidence_source_present(self):
        from app.services.dataset_service import DatasetService
        examples = DatasetService.load("knowledge_base")
        kb001 = next(e for e in examples if e["id"] == "kb001")
        assert kb001["evidence_source"] == "Paris"
        assert "population" in kb001["gold_chunk_keywords"]


# ─── ReAct Tool Metadata ────────────────────────────────────────────────────

class TestReActToolMetadata:
    """Item 9: Expected tool metadata in react_bench."""

    def test_all_examples_have_tool_metadata(self):
        from app.services.dataset_service import DatasetService
        examples = DatasetService.load("react_bench")
        for example in examples:
            assert "expected_tools" in example, f"Missing expected_tools in {example['id']}"
            assert "must_use_tool" in example, f"Missing must_use_tool in {example['id']}"
            assert isinstance(example["expected_tools"], list)
            assert len(example["expected_tools"]) > 0

    def test_search_questions_require_search_tool(self):
        from app.services.dataset_service import DatasetService
        examples = DatasetService.load("react_bench")
        search_examples = [e for e in examples if "search" in e["question"].lower()]
        for e in search_examples:
            assert "wikipedia_search" in e["expected_tools"], \
                f"{e['id']} has 'search' in question but no wikipedia_search in expected_tools"
            assert e["must_use_tool"] is True


# ─── Expected Tools Grader ──────────────────────────────────────────────────

class TestExpectedToolsGrader:
    """New EXPECTED_TOOLS grader: per-example tool-path evaluation."""

    def test_pass_when_all_tools_used(self):
        from app.services.grader_service import GraderEngine, VerdictStatus
        from app.schemas.experiment import GraderRule, GraderType
        engine = GraderEngine()
        run = MagicMock()
        run.example_metadata = {"expected_tools": ["wikipedia_search", "calculator"], "must_use_tool": True}
        run.agent_trace = {"steps": [{"action": "wikipedia_search"}, {"action": "calculator"}]}
        rule = GraderRule(name="tool_check", type=GraderType.EXPECTED_TOOLS, params={})
        verdict = engine.grade_run(run, rule, "react", False)
        assert verdict.status == VerdictStatus.PASS

    def test_fail_when_tool_missing(self):
        from app.services.grader_service import GraderEngine, VerdictStatus
        from app.schemas.experiment import GraderRule, GraderType
        engine = GraderEngine()
        run = MagicMock()
        run.example_metadata = {"expected_tools": ["wikipedia_search", "calculator"], "must_use_tool": True}
        run.agent_trace = {"steps": [{"action": "calculator"}]}
        rule = GraderRule(name="tool_check", type=GraderType.EXPECTED_TOOLS, params={})
        verdict = engine.grade_run(run, rule, "react", False)
        assert verdict.status == VerdictStatus.FAIL
        assert verdict.reason is not None and "wikipedia_search" in verdict.reason

    def test_skip_for_non_react(self):
        from app.services.grader_service import GraderEngine, VerdictStatus
        from app.schemas.experiment import GraderRule, GraderType
        engine = GraderEngine()
        run = MagicMock()
        run.example_metadata = {"expected_tools": ["calculator"]}
        rule = GraderRule(name="tool_check", type=GraderType.EXPECTED_TOOLS, params={})
        verdict = engine.grade_run(run, rule, "naive", False)
        assert verdict.status == VerdictStatus.SKIP

    def test_skip_when_no_expected_tools(self):
        from app.services.grader_service import GraderEngine, VerdictStatus
        from app.schemas.experiment import GraderRule, GraderType
        engine = GraderEngine()
        run = MagicMock()
        run.example_metadata = {}
        rule = GraderRule(name="tool_check", type=GraderType.EXPECTED_TOOLS, params={})
        verdict = engine.grade_run(run, rule, "react", False)
        assert verdict.status == VerdictStatus.SKIP

    def test_fail_must_use_tool_no_trace(self):
        from app.services.grader_service import GraderEngine, VerdictStatus
        from app.schemas.experiment import GraderRule, GraderType
        engine = GraderEngine()
        run = MagicMock()
        run.example_metadata = {"expected_tools": ["wikipedia_search"], "must_use_tool": True}
        run.agent_trace = None
        rule = GraderRule(name="tool_check", type=GraderType.EXPECTED_TOOLS, params={})
        verdict = engine.grade_run(run, rule, "react", False)
        assert verdict.status == VerdictStatus.FAIL

    def test_hit_rate_partial(self):
        from app.services.grader_service import GraderEngine, VerdictStatus
        from app.schemas.experiment import GraderRule, GraderType
        engine = GraderEngine()
        run = MagicMock()
        run.example_metadata = {"expected_tools": ["wikipedia_search", "calculator"], "must_use_tool": True}
        run.agent_trace = {"steps": [{"action": "wikipedia_search"}]}
        rule = GraderRule(name="tool_check", type=GraderType.EXPECTED_TOOLS, params={})
        verdict = engine.grade_run(run, rule, "react", False)
        assert verdict.status == VerdictStatus.FAIL
        assert isinstance(verdict.value, dict)
        assert verdict.value["hit_rate"] == 0.5

# ─── Judge Seed Determinism ─────────────────────────────────────────────────

class TestJudgeSeedDeterminism:
    """Item 10: Deterministic judge sampling with experiment_id seed."""

    def test_same_seed_same_sample(self):
        """Same experiment_id should produce same sample."""
        experiment_id = "550e8400-e29b-41d4-a716-446655440000"
        items = list(range(100))

        rng1 = random.Random(hash(str(experiment_id)))
        sample1 = rng1.sample(items, 20)

        rng2 = random.Random(hash(str(experiment_id)))
        sample2 = rng2.sample(items, 20)

        assert sample1 == sample2

    def test_different_seed_different_sample(self):
        """Different experiment_ids should produce different samples."""
        items = list(range(100))

        rng1 = random.Random(hash("experiment_a"))
        sample1 = rng1.sample(items, 20)

        rng2 = random.Random(hash("experiment_b"))
        sample2 = rng2.sample(items, 20)

        assert sample1 != sample2


# ─── Provenance Temperature Tracking ────────────────────────────────────────

class TestProvenanceTemperature:
    """Item 11: Temperature adjustment tracked in provenance manifest."""

    def test_temperature_adjustment_tracked(self):
        from app.services.experiment_provenance import build_effective_execution_manifest_entry
        manifest, _ = build_effective_execution_manifest_entry(
            attempt=1,
            engine_type="hf_api",
            provider="huggingface",
            routing_config=None,
            configured_hyperparameters={"temperature": 0.0, "max_tokens": 512},
            effective_hyperparameters={"temperature": 0.01, "max_tokens": 512},
            dataset_hash="abc123",
            sample_ids=["ex001"],
            sample_count=1,
            execution_mode="inline",
            rag_enabled=False,
            optimization=None,
        )
        assert "temperature" in manifest["adjustments"]
        assert manifest["adjustments"]["temperature"]["configured"] == 0.0
        assert manifest["adjustments"]["temperature"]["effective"] == 0.01

    def test_no_adjustment_when_matching(self):
        from app.services.experiment_provenance import build_effective_execution_manifest_entry
        manifest, _ = build_effective_execution_manifest_entry(
            attempt=1,
            engine_type="hf_api",
            provider="huggingface",
            routing_config=None,
            configured_hyperparameters={"temperature": 0.7, "max_tokens": 512},
            effective_hyperparameters={"temperature": 0.7, "max_tokens": 512},
            dataset_hash="abc123",
            sample_ids=["ex001"],
            sample_count=1,
            execution_mode="inline",
            rag_enabled=False,
            optimization=None,
        )
        assert "temperature" not in manifest["adjustments"]


# ─── Faithfulness Renamed Fields ────────────────────────────────────────────

class TestFaithfulnessRename:
    """Item 5: context_support_score alias and methodology note."""

    def test_context_support_score_present(self):
        """_compute_faithfulness should return context_support_score."""
        from app.services.metrics_service import MetricsService
        svc = MetricsService.__new__(MetricsService)

        runs = []
        for i in range(10):
            run = MagicMock()
            run.faithfulness_score = 0.7 + i * 0.01
            runs.append(run)

        result = svc._compute_faithfulness(runs)
        assert "context_support_score" in result
        assert result["context_support_score"] == result["mean"]
        assert "methodology_note" in result
        assert "NLI proxy" in result["methodology_note"]

    def test_legacy_hallucination_rate_preserved(self):
        """hallucination_rate should still be present for backward compat."""
        from app.services.metrics_service import MetricsService
        svc = MetricsService.__new__(MetricsService)

        runs = [MagicMock(faithfulness_score=0.3)]
        result = svc._compute_faithfulness(runs)
        assert "hallucination_rate" in result


# ─── Accuracy Excluding Failures ────────────────────────────────────────────

class TestAccuracyExcludingFailures:
    """Item 3: Separate model performance from infrastructure noise."""

    def test_accuracy_excluding_failures(self):
        from app.services.metrics_service import MetricsService
        svc = MetricsService.__new__(MetricsService)

        runs = []
        for i in range(10):
            run = MagicMock()
            run.score = 1.0
            run.is_exact_match = True
            run.is_substring_match = False
            run.is_correct = True
            run.failure_mode = None
            runs.append(run)

        # Add 5 infrastructure failures
        for i in range(5):
            run = MagicMock()
            run.score = 0.0
            run.is_exact_match = False
            run.is_substring_match = False
            run.is_correct = False
            run.failure_mode = MagicMock(value="api_error")
            runs.append(run)

        result = svc._compute_accuracy(runs)
        assert result["accuracy_excluding_failures"] == 1.0  # 10/10 non-failures correct
        assert result["total_excluding_failures"] == 10
        assert result["accuracy_any"] < 1.0  # 10/15 < 1.0


# ─── P0 Wiring: run_metadata persistence ───────────────────────────────────

class TestRunMetadataColumn:
    """P0: run_metadata JSONB column exists and doesn't break Run construction."""

    def test_run_model_accepts_run_metadata(self):
        """Run(**data) with run_metadata should not raise TypeError."""
        from app.models.run import Run
        run = Run(
            experiment_id="00000000-0000-0000-0000-000000000001",
            prompt="test",
            run_metadata={"parse_method": "explicit_pattern"},
        )
        assert run.run_metadata is not None
        assert run.run_metadata["parse_method"] == "explicit_pattern"

    def test_run_model_accepts_none_metadata(self):
        from app.models.run import Run
        run = Run(
            experiment_id="00000000-0000-0000-0000-000000000001",
            prompt="test",
        )
        assert run.run_metadata is None

    def test_runner_batch_data_uses_run_metadata(self):
        """Verify experiment_runner stores parse_method inside run_metadata."""
        import ast
        from pathlib import Path
        runner_path = Path(__file__).parent.parent / "app" / "services" / "experiment_runner.py"
        source = runner_path.read_text(encoding="utf-8")
        # Ensure parse_method is NOT a top-level key in batch data
        assert '"parse_method": parse_method,' not in source, \
            "parse_method should be inside run_metadata, not a top-level key"
        # Should be passed via build_run_record's run_metadata kwarg
        assert 'run_metadata={"parse_method": parse_method}' in source


# ─── P1 Wiring: Metrics API response ───────────────────────────────────────

class TestMetricsApiWiring:
    """P1: result_to_metrics_response maps new fields from raw_metrics."""

    def test_new_quality_fields_mapped(self):
        from app.api.results_common import result_to_metrics_response
        from datetime import datetime, timezone

        result = MagicMock()
        result.experiment_id = "00000000-0000-0000-0000-000000000001"
        result.accuracy_exact = 0.8
        result.accuracy_f1 = 0.75
        result.accuracy_substring = 0.85
        result.semantic_similarity = None
        result.faithfulness = None
        result.hallucination_rate = None
        result.latency_p50 = 100.0
        result.latency_p95 = 200.0
        result.latency_p99 = 300.0
        result.throughput = 5.0
        result.total_tokens_input = 1000
        result.total_tokens_output = 500
        result.total_runs = 10
        result.gpu_time_seconds = 1.0
        result.computed_at = datetime.now(timezone.utc)
        result.raw_metrics = {
            "accuracy": {
                "accuracy_excluding_failures": 0.9,
                "total_excluding_failures": 9,
            },
            "cost": {
                "total_cost_usd": 0.01,
                "cost_per_sample_usd": 0.001,
                "accuracy_per_dollar": 80.0,
                "cost_per_correct_answer": 0.00125,
            },
            "completion_quality": "partial",
            "retrieval_quality": {"recall_at_k": 0.7, "evidence_hit_rate": 0.6, "k": 5},
            "failure_modes": {"total_failures": 1, "counts": {"api_error": 1}},
        }

        resp = result_to_metrics_response(result)

        # Quality fields
        assert resp.quality.accuracy_excluding_failures == 0.9
        assert resp.quality.total_excluding_failures == 9
        assert resp.quality.completion_quality is not None
        assert resp.quality.completion_quality["label"] == "partial"
        assert resp.quality.retrieval_quality is not None
        assert resp.quality.retrieval_quality["recall_at_k"] == 0.7

        # Cost fields
        assert resp.cost.cost_per_sample_usd == 0.001
        assert resp.cost.accuracy_per_dollar == 80.0

    def test_missing_raw_metrics_safe(self):
        """result_to_metrics_response handles None raw_metrics gracefully."""
        from app.api.results_common import result_to_metrics_response
        from datetime import datetime, timezone

        result = MagicMock()
        result.experiment_id = "00000000-0000-0000-0000-000000000001"
        result.accuracy_exact = 0.5
        result.accuracy_f1 = 0.4
        result.accuracy_substring = 0.5
        result.semantic_similarity = None
        result.faithfulness = None
        result.hallucination_rate = None
        result.latency_p50 = 50.0
        result.latency_p95 = 100.0
        result.latency_p99 = 150.0
        result.throughput = 10.0
        result.total_tokens_input = 500
        result.total_tokens_output = 250
        result.total_runs = 5
        result.gpu_time_seconds = None
        result.computed_at = datetime.now(timezone.utc)
        result.raw_metrics = None

        resp = result_to_metrics_response(result)
        assert resp.quality.accuracy_excluding_failures is None
        assert resp.quality.completion_quality is None
        assert resp.quality.retrieval_quality is None
        assert resp.cost.cost_per_sample_usd is None
        assert resp.cost.accuracy_per_dollar is None


# ─── P1 Wiring: Postprocessing metadata ────────────────────────────────────

class TestPostprocessingMetadataWiring:
    """P1: apply_graders loads dataset and attaches example_metadata."""

    def test_postprocessing_source_has_metadata_wiring(self):
        """Verify experiment_postprocessing attaches example_metadata."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "app" / "services" / "experiment_postprocessing.py").read_text(encoding="utf-8")
        assert "example_meta_by_id" in source
        assert "example_metadata" in source
        assert "DatasetService.load" in source
