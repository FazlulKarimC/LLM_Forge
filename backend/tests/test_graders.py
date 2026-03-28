"""
Tests for GraderEngine — Phase 1G.

Covers:
- All 7 grader types with pass and fail cases
- Skip behavior for wrong reasoning method / no RAG
- Unique name validation on GradersConfig
- grade_all_runs batch operation
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from app.services.grader_service import GraderEngine, GraderVerdict, VerdictStatus
from app.schemas.experiment import GraderRule, GraderType, GradersConfig


# ─── Mock Run object ────────────────────────────────────────────────────────

@dataclass
class MockRun:
    """Minimal Run-like object for grader tests."""
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    agent_trace: Optional[Dict[str, Any]] = None
    failure_mode: Optional[Any] = None
    retrieved_chunks: Optional[Any] = None
    latency_ms: Optional[float] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    score: Optional[float] = None


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return GraderEngine()


# ─── max_turns ───────────────────────────────────────────────────────────────

class TestMaxTurnsGrader:
    def test_pass_within_budget(self, engine):
        run = MockRun(agent_trace={"steps": [{"action": "search"}, {"action": "calc"}]})
        rule = GraderRule(name="t", type=GraderType.MAX_TURNS, params={"max": 5})
        v = engine.grade_run(run, rule, "react", False)
        assert v.status == VerdictStatus.PASS
        assert v.value == 2
        assert v.threshold == 5

    def test_fail_over_budget(self, engine):
        steps = [{"action": f"step_{i}"} for i in range(6)]
        run = MockRun(agent_trace={"steps": steps})
        rule = GraderRule(name="t", type=GraderType.MAX_TURNS, params={"max": 3})
        v = engine.grade_run(run, rule, "react", False)
        assert v.status == VerdictStatus.FAIL
        assert v.value == 6

    def test_skip_on_naive(self, engine):
        run = MockRun()
        rule = GraderRule(name="t", type=GraderType.MAX_TURNS, params={"max": 5})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.SKIP

    def test_skip_on_cot(self, engine):
        run = MockRun()
        rule = GraderRule(name="t", type=GraderType.MAX_TURNS, params={"max": 5})
        v = engine.grade_run(run, rule, "cot", False)
        assert v.status == VerdictStatus.SKIP

    def test_skip_no_trace(self, engine):
        run = MockRun(agent_trace=None)
        rule = GraderRule(name="t", type=GraderType.MAX_TURNS, params={"max": 5})
        v = engine.grade_run(run, rule, "react", False)
        assert v.status == VerdictStatus.SKIP

    def test_exact_boundary_pass(self, engine):
        run = MockRun(agent_trace={"steps": [{"action": "a"}] * 5})
        rule = GraderRule(name="t", type=GraderType.MAX_TURNS, params={"max": 5})
        v = engine.grade_run(run, rule, "react", False)
        assert v.status == VerdictStatus.PASS

    def test_default_max(self, engine):
        """Default max is 5 when params omit 'max'."""
        run = MockRun(agent_trace={"steps": [{"action": "a"}] * 4})
        rule = GraderRule(name="t", type=GraderType.MAX_TURNS, params={})
        v = engine.grade_run(run, rule, "react", False)
        assert v.status == VerdictStatus.PASS
        assert v.threshold == 5


# ─── required_tools ──────────────────────────────────────────────────────────

class TestRequiredToolsGrader:
    def test_pass_all_present(self, engine):
        run = MockRun(agent_trace={"steps": [
            {"action": "wikipedia_search"},
            {"action": "calculator"},
        ]})
        rule = GraderRule(name="t", type=GraderType.REQUIRED_TOOLS,
                          params={"tools": ["wikipedia_search", "calculator"]})
        v = engine.grade_run(run, rule, "react", False)
        assert v.status == VerdictStatus.PASS

    def test_fail_missing_tool(self, engine):
        run = MockRun(agent_trace={"steps": [{"action": "calculator"}]})
        rule = GraderRule(name="t", type=GraderType.REQUIRED_TOOLS,
                          params={"tools": ["wikipedia_search", "calculator"]})
        v = engine.grade_run(run, rule, "react", False)
        assert v.status == VerdictStatus.FAIL
        assert "wikipedia_search" in str(v.reason)

    def test_skip_on_naive(self, engine):
        run = MockRun()
        rule = GraderRule(name="t", type=GraderType.REQUIRED_TOOLS,
                          params={"tools": ["calculator"]})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.SKIP


# ─── forbidden_failure_modes ─────────────────────────────────────────────────

class TestForbiddenFailureModes:
    def test_pass_no_failure(self, engine):
        run = MockRun(failure_mode=None)
        rule = GraderRule(name="t", type=GraderType.FORBIDDEN_FAILURE_MODES,
                          params={"modes": ["api_error", "timeout"]})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.PASS

    def test_pass_allowed_failure(self, engine):
        """A failure mode that isn't in the forbidden list passes."""
        run = MockRun(failure_mode="truncated")
        rule = GraderRule(name="t", type=GraderType.FORBIDDEN_FAILURE_MODES,
                          params={"modes": ["api_error", "timeout"]})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.PASS

    def test_fail_forbidden(self, engine):
        run = MockRun(failure_mode="api_error")
        rule = GraderRule(name="t", type=GraderType.FORBIDDEN_FAILURE_MODES,
                          params={"modes": ["api_error", "timeout"]})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.FAIL

    def test_fail_with_enum_failure_mode(self, engine):
        """FailureMode enum values should be extracted via .value."""
        from app.models.run import FailureMode
        run = MockRun(failure_mode=FailureMode.API_ERROR)
        rule = GraderRule(name="t", type=GraderType.FORBIDDEN_FAILURE_MODES,
                          params={"modes": ["api_error"]})
        v = engine.grade_run(run, rule, "react", False)
        assert v.status == VerdictStatus.FAIL

    def test_never_skipped(self, engine):
        """This grader never skips — applies to all reasoning methods."""
        run = MockRun(failure_mode=None)
        rule = GraderRule(name="t", type=GraderType.FORBIDDEN_FAILURE_MODES,
                          params={"modes": ["timeout"]})
        for method in ["naive", "cot", "react"]:
            v = engine.grade_run(run, rule, method, False)
            assert v.status != VerdictStatus.SKIP


# ─── must_use_retrieval_when_rag ─────────────────────────────────────────────

class TestMustUseRetrievalWhenRag:
    def test_pass_chunks_present(self, engine):
        run = MockRun(retrieved_chunks={"chunks": [{"text": "foo"}]})
        rule = GraderRule(name="t", type=GraderType.MUST_USE_RETRIEVAL_WHEN_RAG)
        v = engine.grade_run(run, rule, "naive", True)
        assert v.status == VerdictStatus.PASS

    def test_fail_no_chunks(self, engine):
        run = MockRun(retrieved_chunks=None)
        rule = GraderRule(name="t", type=GraderType.MUST_USE_RETRIEVAL_WHEN_RAG)
        v = engine.grade_run(run, rule, "naive", True)
        assert v.status == VerdictStatus.FAIL

    def test_fail_empty_chunks(self, engine):
        run = MockRun(retrieved_chunks={"chunks": []})
        rule = GraderRule(name="t", type=GraderType.MUST_USE_RETRIEVAL_WHEN_RAG)
        v = engine.grade_run(run, rule, "naive", True)
        assert v.status == VerdictStatus.FAIL

    def test_skip_no_rag(self, engine):
        run = MockRun()
        rule = GraderRule(name="t", type=GraderType.MUST_USE_RETRIEVAL_WHEN_RAG)
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.SKIP


# ─── latency_budget_ms ───────────────────────────────────────────────────────

class TestLatencyBudget:
    def test_pass_under_budget(self, engine):
        run = MockRun(latency_ms=1200.0)
        rule = GraderRule(name="t", type=GraderType.LATENCY_BUDGET_MS, params={"max": 2000})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.PASS

    def test_fail_over_budget(self, engine):
        run = MockRun(latency_ms=3500.0)
        rule = GraderRule(name="t", type=GraderType.LATENCY_BUDGET_MS, params={"max": 2000})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.FAIL

    def test_skip_no_data(self, engine):
        run = MockRun(latency_ms=None)
        rule = GraderRule(name="t", type=GraderType.LATENCY_BUDGET_MS, params={"max": 2000})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.SKIP


# ─── token_budget ────────────────────────────────────────────────────────────

class TestTokenBudget:
    def test_pass_under_budget(self, engine):
        run = MockRun(tokens_input=500, tokens_output=200)
        rule = GraderRule(name="t", type=GraderType.TOKEN_BUDGET, params={"max": 1000})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.PASS
        assert v.value == 700

    def test_fail_over_budget(self, engine):
        run = MockRun(tokens_input=800, tokens_output=500)
        rule = GraderRule(name="t", type=GraderType.TOKEN_BUDGET, params={"max": 1000})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.FAIL
        assert v.value == 1300

    def test_handles_none_tokens(self, engine):
        """None tokens treated as 0."""
        run = MockRun(tokens_input=None, tokens_output=None)
        rule = GraderRule(name="t", type=GraderType.TOKEN_BUDGET, params={"max": 1000})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.PASS
        assert v.value == 0


# ─── min_f1_score ────────────────────────────────────────────────────────────

class TestMinF1Score:
    def test_pass_above_min(self, engine):
        run = MockRun(score=0.85)
        rule = GraderRule(name="t", type=GraderType.MIN_F1_SCORE, params={"min": 0.5})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.PASS

    def test_fail_below_min(self, engine):
        run = MockRun(score=0.3)
        rule = GraderRule(name="t", type=GraderType.MIN_F1_SCORE, params={"min": 0.5})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.FAIL

    def test_skip_no_score(self, engine):
        run = MockRun(score=None)
        rule = GraderRule(name="t", type=GraderType.MIN_F1_SCORE, params={"min": 0.5})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.SKIP

    def test_exact_boundary_pass(self, engine):
        run = MockRun(score=0.5)
        rule = GraderRule(name="t", type=GraderType.MIN_F1_SCORE, params={"min": 0.5})
        v = engine.grade_run(run, rule, "naive", False)
        assert v.status == VerdictStatus.PASS


# ─── GradersConfig validation ────────────────────────────────────────────────

class TestGradersConfigValidation:
    def test_unique_names_pass(self):
        config = GradersConfig(rules=[
            GraderRule(name="a", type=GraderType.MAX_TURNS),
            GraderRule(name="b", type=GraderType.TOKEN_BUDGET),
        ])
        assert len(config.rules) == 2

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            GradersConfig(rules=[
                GraderRule(name="dup", type=GraderType.MAX_TURNS),
                GraderRule(name="dup", type=GraderType.TOKEN_BUDGET),
            ])


# ─── grade_all_runs ──────────────────────────────────────────────────────────

class TestGradeAllRuns:
    def test_batch_grading(self, engine):
        runs = [
            MockRun(latency_ms=1000.0, score=0.9),
            MockRun(latency_ms=3000.0, score=0.2),
        ]
        config = GradersConfig(rules=[
            GraderRule(name="lat", type=GraderType.LATENCY_BUDGET_MS, params={"max": 2000}),
            GraderRule(name="f1", type=GraderType.MIN_F1_SCORE, params={"min": 0.5}),
        ])
        results = engine.grade_all_runs(runs, config, "naive", False)

        assert len(results) == 2
        # First run: both pass
        r0 = results[str(runs[0].id)]
        assert all(v.status == VerdictStatus.PASS for v in r0)
        # Second run: both fail
        r1 = results[str(runs[1].id)]
        assert all(v.status == VerdictStatus.FAIL for v in r1)


# ─── GraderVerdict serialization ─────────────────────────────────────────────

class TestVerdictSerialization:
    def test_to_dict_pass(self):
        v = GraderVerdict("lat", VerdictStatus.PASS, value=1200, threshold=2000)
        d = v.to_dict()
        assert d == {"status": "pass", "value": 1200, "threshold": 2000}

    def test_to_dict_skip(self):
        v = GraderVerdict("mt", VerdictStatus.SKIP, reason="not_applicable_for_naive")
        d = v.to_dict()
        assert d == {"status": "skip", "reason": "not_applicable_for_naive"}

    def test_to_dict_fail(self):
        v = GraderVerdict("tok", VerdictStatus.FAIL, value=1500, threshold=1000)
        d = v.to_dict()
        assert d == {"status": "fail", "value": 1500, "threshold": 1000}
