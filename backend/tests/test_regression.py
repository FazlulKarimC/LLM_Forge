"""
Tests for RegressionService — Phase 2I.

Covers:
- Aggregate threshold checks (accuracy delta, F1 delta)
- Per-sample regression detection via example_id matching
- Baseline pinning rules (completed only, one per lineage, no self-baseline)
- Overlap ratio below threshold → inconclusive verdict
- Config diff from run_manifest
- compare_run_sets() produces consistent results with compare_experiments()
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from app.schemas.experiment import RegressionConfig
from app.services.regression_service import RegressionService
from app.services.statistical_service import StatisticalService


# ─── Mock objects ────────────────────────────────────────────────────────────

@dataclass
class MockRun:
    """Minimal Run-like object for regression tests."""
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    experiment_id: uuid.UUID = field(default_factory=uuid.uuid4)
    example_id: str = ""
    is_correct: bool = False
    score: float = 0.0
    attempt: int = 1
    raw_output: Optional[str] = None
    expected_output: Optional[str] = None
    failure_mode: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    agent_trace: Optional[Dict] = None
    retrieved_chunks: Optional[Any] = None
    grader_results: Optional[Dict] = None
    served_provider: Optional[str] = None
    routing_reason: Optional[str] = None


@dataclass
class MockExperiment:
    """Minimal Experiment-like object for lineage tests."""

    dataset_name: str = "sample"
    model_name: str = "mock-model"
    dataset_hash: Optional[str] = "dataset-hash"
    config: Dict[str, Any] = field(default_factory=dict)


# ─── compare_run_sets tests ──────────────────────────────────────────────────

class TestCompareRunSets:
    """Test the low-level statistical comparison on pre-filtered run lists."""
    
    def _make_runs(self, correctness_list, prefix="ex"):
        """Build mock runs with given correctness pattern."""
        runs = []
        for i, correct in enumerate(correctness_list):
            runs.append(MockRun(
                example_id=f"{prefix}_{i}",
                is_correct=correct,
                score=1.0 if correct else 0.0,
            ))
        return runs
    
    def test_identical_runs_no_diff(self):
        """Two identical sets → no differences."""
        runs_a = self._make_runs([True, True, False, True])
        runs_b = self._make_runs([True, True, False, True])
        
        result = StatisticalService.compare_run_sets(runs_a, runs_b)
        assert result["accuracy_diff"] == 0.0
        assert result["num_common_examples"] == 4
        assert len(result["per_example_differences"]) == 0
    
    def test_regression_detected(self):
        """Candidate worse than baseline → negative diff."""
        runs_a = self._make_runs([True, True, True, True])  # baseline: 4/4
        runs_b = self._make_runs([True, True, False, False]) # candidate: 2/4
        
        result = StatisticalService.compare_run_sets(runs_a, runs_b)
        assert result["accuracy_diff"] == -0.5
        assert result["summary"]["a_only_correct"] > 0
    
    def test_improvement_detected(self):
        """Candidate better than baseline → positive diff."""
        runs_a = self._make_runs([False, False, True, True])  # baseline: 2/4
        runs_b = self._make_runs([True, True, True, True])    # candidate: 4/4
        
        result = StatisticalService.compare_run_sets(runs_a, runs_b)
        assert result["accuracy_diff"] == 0.5
        assert result["summary"]["b_only_correct"] > 0
    
    def test_overlap_ratio_full(self):
        """Full overlap → ratio = 1.0."""
        runs_a = self._make_runs([True, True])
        runs_b = self._make_runs([True, True])
        
        result = StatisticalService.compare_run_sets(runs_a, runs_b)
        assert result["overlap_ratio"] == 1.0
    
    def test_overlap_ratio_partial(self):
        """Partial overlap → ratio < 1.0."""
        runs_a = [
            MockRun(example_id="ex_0", is_correct=True, score=1.0),
            MockRun(example_id="ex_1", is_correct=True, score=1.0),
            MockRun(example_id="ex_2", is_correct=False, score=0.0),
        ]
        runs_b = [
            MockRun(example_id="ex_0", is_correct=True, score=1.0),
            MockRun(example_id="ex_1", is_correct=True, score=1.0),
            MockRun(example_id="ex_3", is_correct=True, score=1.0),  # only in B
        ]
        
        result = StatisticalService.compare_run_sets(runs_a, runs_b)
        assert result["overlap_ratio"] == pytest.approx(2.0/3.0, abs=0.01)
        assert any("not a clean paired comparison" in warning for warning in result["warnings"])

    def test_small_n_warning(self):
        """Small common-example sets should be labeled exploratory."""
        runs_a = self._make_runs([True, False])
        runs_b = self._make_runs([False, False])

        result = StatisticalService.compare_run_sets(runs_a, runs_b)

        assert any("Only 2 common examples" in warning for warning in result["warnings"])

    def test_provider_routing_warning(self):
        """Mixed providers should produce a comparison caveat."""
        runs_a = [
            MockRun(example_id="ex_0", is_correct=True, score=1.0, served_provider="openrouter"),
            MockRun(example_id="ex_1", is_correct=False, score=0.0, served_provider="openrouter"),
        ]
        runs_b = [
            MockRun(example_id="ex_0", is_correct=True, score=1.0, served_provider="groq", routing_reason="fallback_1"),
            MockRun(example_id="ex_1", is_correct=True, score=1.0, served_provider="groq", routing_reason="fallback_1"),
        ]

        result = StatisticalService.compare_run_sets(runs_a, runs_b)

        assert result["routing"]["providers_a"] == ["openrouter"]
        assert result["routing"]["providers_b"] == ["groq"]
        assert any("confounded by routing" in warning for warning in result["warnings"])
    
    def test_no_common_examples_raises(self):
        """Disjoint sets → ValueError."""
        runs_a = [MockRun(example_id="a_0", is_correct=True, score=1.0)]
        runs_b = [MockRun(example_id="b_0", is_correct=True, score=1.0)]
        
        with pytest.raises(ValueError, match="No common examples"):
            StatisticalService.compare_run_sets(runs_a, runs_b)


class TestBaselineLineage:
    """Auto baselines should only match methodologically comparable runs."""

    def _experiment(self, **overrides):
        config = {
            "model_name": "mock-model",
            "dataset_name": "sample",
            "reasoning_method": "naive",
            "provider": "auto",
            "hyperparameters": {"temperature": 0.1, "max_tokens": 150, "seed": 42},
            "num_samples": 10,
            "rag": None,
            "agent": None,
            "optimization": None,
            "graders": None,
            "routing": {
                "policy": "fallback_chain",
                "epsilon": 0.15,
                "exploration_window": 10,
                "strict_comparison": True,
            },
            "prompt_version_id": None,
        }
        config.update(overrides.pop("config_updates", {}))
        return MockExperiment(config=config, **overrides)

    def test_same_execution_defining_config_matches(self):
        baseline = self._experiment()
        candidate = self._experiment()

        assert RegressionService._same_baseline_lineage(baseline, candidate)

    def test_reasoning_method_mismatch_does_not_match(self):
        baseline = self._experiment()
        candidate = self._experiment(config_updates={"reasoning_method": "cot"})

        assert not RegressionService._same_baseline_lineage(baseline, candidate)

    def test_routing_strictness_mismatch_does_not_match(self):
        baseline = self._experiment()
        candidate = self._experiment(
            config_updates={
                "routing": {
                    "policy": "fallback_chain",
                    "epsilon": 0.15,
                    "exploration_window": 10,
                    "strict_comparison": False,
                }
            }
        )

        assert not RegressionService._same_baseline_lineage(baseline, candidate)

    def test_dataset_hash_mismatch_does_not_match(self):
        baseline = self._experiment(dataset_hash="hash-a")
        candidate = self._experiment(dataset_hash="hash-b")

        assert not RegressionService._same_baseline_lineage(baseline, candidate)


# ─── Aggregate threshold checks ─────────────────────────────────────────────

class TestAggregateThresholds:
    """Test RegressionService._check_aggregate_thresholds()."""
    
    def test_no_violations_when_within_bounds(self):
        stats = {
            "accuracy_diff": -0.02,  # Within default -0.05
            "f1_ci_a": {"mean": 0.8},
            "f1_ci_b": {"mean": 0.78},
        }
        config = RegressionConfig()
        violations = RegressionService._check_aggregate_thresholds(stats, config)
        assert len(violations) == 0
    
    def test_accuracy_violation(self):
        stats = {
            "accuracy_diff": -0.10,  # Exceeds -0.05 threshold
            "f1_ci_a": {"mean": 0.8},
            "f1_ci_b": {"mean": 0.8},
        }
        config = RegressionConfig()
        violations = RegressionService._check_aggregate_thresholds(stats, config)
        assert len(violations) == 1
        assert violations[0]["rule"] == "accuracy_min_delta"
    
    def test_f1_violation(self):
        stats = {
            "accuracy_diff": 0.0,
            "f1_ci_a": {"mean": 0.9},
            "f1_ci_b": {"mean": 0.8},  # Drop of 0.1
        }
        config = RegressionConfig(f1_min_delta=-0.05)
        violations = RegressionService._check_aggregate_thresholds(stats, config)
        assert len(violations) == 1
        assert violations[0]["rule"] == "f1_min_delta"
    
    def test_custom_thresholds(self):
        stats = {
            "accuracy_diff": -0.08,
            "f1_ci_a": {"mean": 0.8},
            "f1_ci_b": {"mean": 0.8},
        }
        config = RegressionConfig(accuracy_min_delta=-0.10)  # Lenient
        violations = RegressionService._check_aggregate_thresholds(stats, config)
        assert len(violations) == 0

    def test_both_violations(self):
        stats = {
            "accuracy_diff": -0.15,
            "f1_ci_a": {"mean": 0.9},
            "f1_ci_b": {"mean": 0.7},
        }
        config = RegressionConfig()
        violations = RegressionService._check_aggregate_thresholds(stats, config)
        assert len(violations) == 2

    def test_latency_p95_violation(self):
        stats = {
            "accuracy_diff": 0.0,
            "f1_ci_a": {"mean": 0.9},
            "f1_ci_b": {"mean": 0.9},
            "candidate_latency_p95_ms": 950.0,
        }
        config = RegressionConfig(latency_p95_max_ms=500.0)
        violations = RegressionService._check_aggregate_thresholds(stats, config)

        assert len(violations) == 1
        assert violations[0]["rule"] == "latency_p95_max_ms"


# ─── Sample regression detection ────────────────────────────────────────────

class TestSampleRegressions:
    """Test RegressionService._detect_sample_regressions()."""
    
    def test_detects_regressions(self):
        stats = {
            "per_example_differences": [
                {"example_id": "ex_0", "a_correct": True, "b_correct": False,
                 "a_score": 1.0, "b_score": 0.0},
            ]
        }
        regressions, improvements = RegressionService._detect_sample_regressions(stats)
        assert len(regressions) == 1
        assert regressions[0]["example_id"] == "ex_0"
        assert len(improvements) == 0
    
    def test_detects_improvements(self):
        stats = {
            "per_example_differences": [
                {"example_id": "ex_0", "a_correct": False, "b_correct": True,
                 "a_score": 0.0, "b_score": 1.0},
            ]
        }
        regressions, improvements = RegressionService._detect_sample_regressions(stats)
        assert len(regressions) == 0
        assert len(improvements) == 1
    
    def test_mixed_changes(self):
        stats = {
            "per_example_differences": [
                {"example_id": "ex_0", "a_correct": True, "b_correct": False,
                 "a_score": 1.0, "b_score": 0.0},
                {"example_id": "ex_1", "a_correct": False, "b_correct": True,
                 "a_score": 0.0, "b_score": 1.0},
                {"example_id": "ex_2", "a_correct": True, "b_correct": False,
                 "a_score": 1.0, "b_score": 0.0},
            ]
        }
        regressions, improvements = RegressionService._detect_sample_regressions(stats)
        assert len(regressions) == 2
        assert len(improvements) == 1


# ─── Config diff ─────────────────────────────────────────────────────────────

class TestConfigDiff:
    """Test RegressionService._compute_config_diff()."""
    
    def test_identical_manifests(self):
        m = {"model_name": "phi-2", "dataset_name": "trivia_qa", "manifest_hash": "abc"}
        diff = RegressionService._compute_config_diff(m, m)
        assert len(diff) == 0
    
    def test_detects_changes(self):
        m_a = {"model_name": "phi-2", "temperature": 0.7, "manifest_hash": "abc"}
        m_b = {"model_name": "phi-2", "temperature": 0.9, "manifest_hash": "def"}
        diff = RegressionService._compute_config_diff(m_a, m_b)
        assert "temperature" in diff
        assert diff["temperature"]["baseline"] == 0.7
        assert diff["temperature"]["candidate"] == 0.9
        # manifest_hash should be excluded
        assert "manifest_hash" not in diff
    
    def test_missing_manifests(self):
        diff = RegressionService._compute_config_diff(None, None)
        assert diff == {"error": "missing_manifest"}
    
    def test_new_keys(self):
        m_a = {"model_name": "phi-2", "manifest_hash": "abc"}
        m_b = {"model_name": "phi-2", "graders": {"rules": []}, "manifest_hash": "def"}
        diff = RegressionService._compute_config_diff(m_a, m_b)
        assert "graders" in diff
        assert diff["graders"]["baseline"] is None
        assert diff["graders"]["candidate"] == {"rules": []}


# ─── RegressionConfig validation ──────────────────────────────────────────────

class TestRegressionConfigValidation:
    def test_defaults(self):
        config = RegressionConfig()
        assert config.accuracy_min_delta == -0.05
        assert config.f1_min_delta == -0.05
        assert config.min_overlap_ratio == 0.8
    
    def test_overlap_ratio_bounds(self):
        config = RegressionConfig(min_overlap_ratio=0.0)
        assert config.min_overlap_ratio == 0.0
        
        config = RegressionConfig(min_overlap_ratio=1.0)
        assert config.min_overlap_ratio == 1.0
        
        with pytest.raises(ValueError):
            RegressionConfig(min_overlap_ratio=1.5)
        
        with pytest.raises(ValueError):
            RegressionConfig(min_overlap_ratio=-0.1)


# ─── RegressionVerdict serialization ─────────────────────────────────────────

class TestRegressionVerdictSerialization:
    def test_to_dict(self):
        from app.services.regression_service import RegressionVerdict
        v = RegressionVerdict(
            passed=True,
            baseline_experiment_id=uuid.uuid4(),
            baseline_attempt=1,
            candidate_experiment_id=uuid.uuid4(),
            candidate_attempt=2,
            overlap_ratio=0.95,
            violations=[],
            sample_regressions=[{"example_id": "ex_1"}],
            sample_improvements=[],
        )
        d = v.to_dict()
        assert d["passed"] is True
        assert d["baseline_attempt"] == 1
        assert d["candidate_attempt"] == 2
        assert d["overlap_ratio"] == 0.95
        assert d["sample_regressions_count"] == 1
    
    def test_inconclusive_none_passed(self):
        from app.services.regression_service import RegressionVerdict
        v = RegressionVerdict(
            passed=None,
            baseline_experiment_id=uuid.uuid4(),
            baseline_attempt=1,
            candidate_experiment_id=uuid.uuid4(),
            candidate_attempt=1,
            overlap_ratio=0.3,
        )
        d = v.to_dict()
        assert d["passed"] is None
