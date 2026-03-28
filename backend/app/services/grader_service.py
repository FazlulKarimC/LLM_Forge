"""
Grader Service

Deterministic, rule-based graders applied per-run after execution.
Each grader inspects a Run and produces a tri-state verdict:
  - PASS: the grading criterion was met
  - FAIL: the grading criterion was violated
  - SKIP: the grader is not applicable for this run's config (e.g. max_turns on a naive run)

Grader taxonomy:
  - max_turns: Agent didn't exceed N iterations
  - required_tools: Agent used specific tools
  - forbidden_failure_modes: No specific failure modes occurred
  - must_use_retrieval_when_rag: RAG experiments retrieved context
  - latency_budget_ms: Per-sample latency under threshold
  - token_budget: Tokens within budget
  - min_f1_score: F1 score above minimum
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.schemas.experiment import GraderRule, GraderType, GradersConfig

logger = logging.getLogger(__name__)


class VerdictStatus(str, Enum):
    """Tri-state grader verdict."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class GraderVerdict:
    """Result of a single grader applied to a single run."""
    grader_name: str
    status: VerdictStatus
    value: Any = None
    threshold: Any = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSONB storage on Run.grader_results."""
        d: Dict[str, Any] = {"status": self.status.value}
        if self.value is not None:
            d["value"] = self.value
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.reason is not None:
            d["reason"] = self.reason
        return d


class GraderEngine:
    """
    Applies GradersConfig rules to Run objects.
    
    Stateless — instantiate and call. No DB access needed;
    operates purely on in-memory Run model instances.
    """

    # Dispatch table: GraderType → method name
    _GRADER_METHODS = {
        GraderType.MAX_TURNS: "_grade_max_turns",
        GraderType.REQUIRED_TOOLS: "_grade_required_tools",
        GraderType.FORBIDDEN_FAILURE_MODES: "_grade_forbidden_failure_modes",
        GraderType.MUST_USE_RETRIEVAL_WHEN_RAG: "_grade_must_use_retrieval_when_rag",
        GraderType.LATENCY_BUDGET_MS: "_grade_latency_budget_ms",
        GraderType.TOKEN_BUDGET: "_grade_token_budget",
        GraderType.MIN_F1_SCORE: "_grade_min_f1_score",
    }

    def grade_run(
        self,
        run: Any,
        rule: GraderRule,
        reasoning_method: str,
        has_rag: bool,
    ) -> GraderVerdict:
        """
        Apply a single grader rule to a single run.
        
        Args:
            run: Run model instance (or any object with matching attributes)
            rule: The grader rule to apply
            reasoning_method: e.g. "naive", "cot", "react"
            has_rag: Whether the experiment uses RAG retrieval
            
        Returns:
            GraderVerdict with pass/fail/skip status
        """
        method_name = self._GRADER_METHODS.get(rule.type)
        if method_name is None:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.SKIP,
                reason=f"Unknown grader type: {rule.type}",
            )
        
        method = getattr(self, method_name)
        return method(run, rule, reasoning_method, has_rag)

    def grade_all_runs(
        self,
        runs: List[Any],
        config: GradersConfig,
        reasoning_method: str,
        has_rag: bool,
    ) -> Dict[str, List[GraderVerdict]]:
        """
        Apply all grader rules to all runs.
        
        Returns:
            Dict mapping run.id (as str) to list of GraderVerdicts
        """
        results: Dict[str, List[GraderVerdict]] = {}
        for run in runs:
            run_id = str(run.id)
            verdicts = [
                self.grade_run(run, rule, reasoning_method, has_rag)
                for rule in config.rules
            ]
            results[run_id] = verdicts
        return results

    # =========================================================================
    # Individual grader implementations
    # =========================================================================

    def _grade_max_turns(
        self, run: Any, rule: GraderRule, reasoning_method: str, has_rag: bool,
    ) -> GraderVerdict:
        """Agent didn't exceed N iterations. SKIP for non-ReAct."""
        if reasoning_method != "react":
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.SKIP,
                reason="not_applicable_for_non_react",
            )
        
        max_turns = rule.params.get("max", 5)
        trace = getattr(run, "agent_trace", None)
        
        if trace is None:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.SKIP,
                reason="no_agent_trace_available",
            )
        
        steps = trace.get("steps", []) if isinstance(trace, dict) else []
        actual_turns = len(steps)
        passed = actual_turns <= max_turns
        
        return GraderVerdict(
            grader_name=rule.name,
            status=VerdictStatus.PASS if passed else VerdictStatus.FAIL,
            value=actual_turns,
            threshold=max_turns,
        )

    def _grade_required_tools(
        self, run: Any, rule: GraderRule, reasoning_method: str, has_rag: bool,
    ) -> GraderVerdict:
        """Agent used specific tools. SKIP for non-ReAct."""
        if reasoning_method != "react":
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.SKIP,
                reason="not_applicable_for_non_react",
            )
        
        required = set(rule.params.get("tools", []))
        if not required:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.PASS,
                value=[],
                threshold=list(required),
                reason="no_tools_required",
            )
        
        trace = getattr(run, "agent_trace", None)
        if trace is None:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.SKIP,
                reason="no_agent_trace_available",
            )
        
        steps = trace.get("steps", []) if isinstance(trace, dict) else []
        used_tools = set()
        for step in steps:
            action = step.get("action", "") if isinstance(step, dict) else ""
            if action:
                used_tools.add(action)
        
        missing = required - used_tools
        passed = len(missing) == 0
        
        return GraderVerdict(
            grader_name=rule.name,
            status=VerdictStatus.PASS if passed else VerdictStatus.FAIL,
            value=sorted(used_tools),
            threshold=sorted(required),
            reason=f"missing: {sorted(missing)}" if missing else None,
        )

    def _grade_forbidden_failure_modes(
        self, run: Any, rule: GraderRule, reasoning_method: str, has_rag: bool,
    ) -> GraderVerdict:
        """No specific failure modes occurred. Never skipped."""
        forbidden = set(rule.params.get("modes", []))
        if not forbidden:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.PASS,
                reason="no_modes_forbidden",
            )
        
        failure_mode = getattr(run, "failure_mode", None)
        actual_mode = failure_mode.value if hasattr(failure_mode, "value") else failure_mode
        
        if actual_mode is None:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.PASS,
                value=None,
                threshold=sorted(forbidden),
            )
        
        passed = actual_mode not in forbidden
        return GraderVerdict(
            grader_name=rule.name,
            status=VerdictStatus.PASS if passed else VerdictStatus.FAIL,
            value=actual_mode,
            threshold=sorted(forbidden),
        )

    def _grade_must_use_retrieval_when_rag(
        self, run: Any, rule: GraderRule, reasoning_method: str, has_rag: bool,
    ) -> GraderVerdict:
        """RAG experiments retrieved context. SKIP on non-RAG."""
        if not has_rag:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.SKIP,
                reason="not_applicable_for_non_rag",
            )
        
        chunks = getattr(run, "retrieved_chunks", None)
        
        if chunks is None:
            passed = False
            chunk_count = 0
        elif isinstance(chunks, dict):
            chunk_list = chunks.get("chunks", [])
            chunk_count = len(chunk_list)
            passed = chunk_count > 0
        elif isinstance(chunks, list):
            chunk_count = len(chunks)
            passed = chunk_count > 0
        else:
            passed = False
            chunk_count = 0
        
        return GraderVerdict(
            grader_name=rule.name,
            status=VerdictStatus.PASS if passed else VerdictStatus.FAIL,
            value=chunk_count,
            threshold=1,
            reason="no_chunks_retrieved" if not passed else None,
        )

    def _grade_latency_budget_ms(
        self, run: Any, rule: GraderRule, reasoning_method: str, has_rag: bool,
    ) -> GraderVerdict:
        """Per-sample latency under threshold. Never skipped."""
        max_latency = rule.params.get("max", 5000)
        actual_latency = getattr(run, "latency_ms", None)
        
        if actual_latency is None:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.SKIP,
                reason="no_latency_data",
            )
        
        passed = actual_latency <= max_latency
        return GraderVerdict(
            grader_name=rule.name,
            status=VerdictStatus.PASS if passed else VerdictStatus.FAIL,
            value=round(actual_latency, 1),
            threshold=max_latency,
        )

    def _grade_token_budget(
        self, run: Any, rule: GraderRule, reasoning_method: str, has_rag: bool,
    ) -> GraderVerdict:
        """Tokens within budget. Never skipped."""
        max_tokens = rule.params.get("max", 1500)
        tokens_in = getattr(run, "tokens_input", 0) or 0
        tokens_out = getattr(run, "tokens_output", 0) or 0
        total = tokens_in + tokens_out
        
        passed = total <= max_tokens
        return GraderVerdict(
            grader_name=rule.name,
            status=VerdictStatus.PASS if passed else VerdictStatus.FAIL,
            value=total,
            threshold=max_tokens,
        )

    def _grade_min_f1_score(
        self, run: Any, rule: GraderRule, reasoning_method: str, has_rag: bool,
    ) -> GraderVerdict:
        """F1 score above minimum. Never skipped."""
        min_score = rule.params.get("min", 0.5)
        actual_score = getattr(run, "score", None)
        
        if actual_score is None:
            return GraderVerdict(
                grader_name=rule.name,
                status=VerdictStatus.SKIP,
                reason="no_score_available",
            )
        
        passed = actual_score >= min_score
        return GraderVerdict(
            grader_name=rule.name,
            status=VerdictStatus.PASS if passed else VerdictStatus.FAIL,
            value=round(actual_score, 4),
            threshold=min_score,
        )
