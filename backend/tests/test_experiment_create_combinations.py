"""
Experiment Create — Pairwise Combination Tests

Pure Pydantic schema tests covering valid and invalid combinations of
experiment config options that the frontend form exposes.
No HTTP, no DB — tests ExperimentCreate + ExperimentConfig directly.

Uses pairwise coverage (~35 cases) instead of full Cartesian product.
"""

import pytest
from pydantic import ValidationError

from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentConfig,
    AgentConfig,
    GradersConfig,
    GraderRule,
    GraderType,
    OptimizationConfig,
    RAGConfig,
    RegressionConfig,
    RoutingConfig,
)


def _create(
    name: str = "combo-test",
    reasoning_method: str = "naive",
    model_name: str = "test-model",
    dataset_name: str = "sample",
    provider: str = "auto",
    num_samples: int = 10,
    rag: dict | None = None,
    agent: dict | None = None,
    optimization: dict | None = None,
    regression: dict | None = None,
    graders: dict | None = None,
    routing: dict | None = None,
) -> ExperimentCreate:
    """Build and validate an ExperimentCreate from kwargs."""
    config_kw: dict = {
        "reasoning_method": reasoning_method,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "provider": provider,
        "num_samples": num_samples,
    }
    if rag is not None:
        config_kw["rag"] = RAGConfig(**rag)
    if agent is not None:
        config_kw["agent"] = AgentConfig(**agent)
    if optimization is not None:
        config_kw["optimization"] = OptimizationConfig(**optimization)
    if regression is not None:
        config_kw["regression"] = RegressionConfig(**regression)
    if graders is not None:
        config_kw["graders"] = GradersConfig(**graders) if isinstance(graders, dict) and "rules" not in graders else graders
    if routing is not None:
        config_kw["routing"] = RoutingConfig(**routing)

    return ExperimentCreate(name=name, config=ExperimentConfig(**config_kw))


# ── Pairwise valid combos ───────────────────────────────────────────────

VALID_COMBOS = [
    # (id, method, dataset, provider, rag, opt, regression, routing)
    ("naive-sample-auto", "naive", "sample", "auto", None, None, None, None),
    ("naive-trivia-hf", "naive", "trivia_qa", "hf_api", None, None, None, None),
    ("naive-sample-openrouter", "naive", "sample", "openrouter", None, None, None, None),
    ("naive-sample-groq", "naive", "sample", "groq", None, None, None, None),
    ("cot-sample-auto", "cot", "sample", "auto", None, None, None, None),
    ("cot-trivia-hf", "cot", "trivia_qa", "hf_api", None, None, None, None),
    ("cot-sample-groq", "cot", "sample", "groq", None, None, None, None),
    ("react-sample-auto", "react", "sample", "auto", None, None, None, None),
    ("react-trivia-openrouter", "react", "trivia_qa", "openrouter", None, None, None, None),
    # RAG
    ("naive-rag-naive", "naive", "sample", "auto",
     {"retrieval_method": "naive", "top_k": 5}, None, None, None),
    ("naive-rag-hybrid", "naive", "sample", "auto",
     {"retrieval_method": "hybrid", "top_k": 10}, None, None, None),
    ("cot-rag-reranked", "cot", "trivia_qa", "hf_api",
     {"retrieval_method": "reranked", "top_k": 3}, None, None, None),
    # Optimization
    ("naive-batch-only", "naive", "sample", "auto", None,
     {"enable_batching": True, "batch_size": 8, "enable_caching": False}, None, None),
    ("naive-cache-only", "naive", "sample", "auto", None,
     {"enable_batching": False, "enable_caching": True, "cache_max_size": 128}, None, None),
    ("cot-batch-and-cache", "cot", "trivia_qa", "groq", None,
     {"enable_batching": True, "batch_size": 16, "enable_caching": True}, None, None),
    # Regression
    ("naive-regression-defaults", "naive", "sample", "auto", None, None, {}, None),
    ("cot-regression-custom", "cot", "sample", "hf_api", None, None,
     {"accuracy_min_delta": -0.10, "f1_min_delta": -0.08, "min_overlap_ratio": 0.9}, None),
    # Routing
    ("naive-route-fallback", "naive", "sample", "auto", None, None, None, {"policy": "fallback_chain"}),
    ("naive-route-cheapest", "naive", "sample", "auto", None, None, None, {"policy": "cheapest_first"}),
    ("naive-route-fastest", "naive", "sample", "auto", None, None, None, {"policy": "fastest_first"}),
    ("naive-route-adaptive", "naive", "sample", "auto", None, None, None, {"policy": "adaptive", "epsilon": 0.2}),
    # Combined
    ("full-naive", "naive", "sample", "auto",
     {"retrieval_method": "hybrid", "top_k": 5},
     {"enable_batching": True, "batch_size": 4, "enable_caching": True},
     {"accuracy_min_delta": -0.05},
     {"policy": "adaptive", "epsilon": 0.1}),
    ("full-cot", "cot", "trivia_qa", "openrouter",
     {"retrieval_method": "reranked", "top_k": 3},
     {"enable_batching": True, "batch_size": 8},
     {"f1_min_delta": -0.03, "no_sample_regressions": True},
     {"policy": "cheapest_first"}),
    ("react-with-routing", "react", "sample", "groq",
     {"retrieval_method": "naive", "top_k": 5}, None, None, {"policy": "fallback_chain"}),
]


class TestValidCombinations:
    @pytest.mark.parametrize(
        "combo_id, method, dataset, provider, rag, opt, regression, routing",
        VALID_COMBOS,
        ids=[c[0] for c in VALID_COMBOS],
    )
    def test_valid_combo_accepted(self, combo_id, method, dataset, provider, rag, opt, regression, routing):
        exp = _create(
            name=f"test-{combo_id}",
            reasoning_method=method,
            dataset_name=dataset,
            provider=provider,
            rag=rag,
            optimization=opt,
            regression=regression,
            routing=routing,
        )
        assert exp.name == f"test-{combo_id}"
        assert exp.config.reasoning_method.value == method
        assert exp.config.dataset_name == dataset
        assert exp.config.provider.value == provider

        if rag:
            assert exp.config.rag is not None
            assert exp.config.rag.retrieval_method.value == rag["retrieval_method"]
        if opt:
            assert exp.config.optimization is not None
            assert exp.config.optimization.enable_batching == opt["enable_batching"]
        if regression is not None:
            assert exp.config.regression is not None
        if routing:
            assert exp.config.routing is not None
            assert exp.config.routing.policy == routing["policy"]


class TestInvalidCombinations:
    def test_agent_on_naive_rejected(self):
        with pytest.raises(ValidationError, match="Agent config only valid for ReAct"):
            _create(reasoning_method="naive", agent={"max_iterations": 3, "tools": ["calculator"]})

    def test_agent_on_cot_rejected(self):
        with pytest.raises(ValidationError, match="Agent config only valid for ReAct"):
            _create(reasoning_method="cot", agent={"max_iterations": 5, "tools": ["wikipedia_search"]})

    def test_zero_samples_rejected(self):
        with pytest.raises(ValidationError):
            _create(num_samples=0)

    def test_501_samples_rejected(self):
        with pytest.raises(ValidationError):
            _create(num_samples=501)


class TestReactAgentCombinations:
    def test_react_calculator_only(self):
        exp = _create(reasoning_method="react", agent={"max_iterations": 3, "tools": ["calculator"]})
        assert exp.config.agent.tools == ["calculator"]

    def test_react_retrieval_and_search(self):
        exp = _create(reasoning_method="react", agent={"max_iterations": 5, "tools": ["retrieval", "wikipedia_search"]})
        assert set(exp.config.agent.tools) == {"retrieval", "wikipedia_search"}

    def test_react_default_tools(self):
        exp = _create(reasoning_method="react", agent={"max_iterations": 5})
        assert len(exp.config.agent.tools) > 0

    def test_react_empty_tools(self):
        exp = _create(reasoning_method="react", agent={"max_iterations": 3, "tools": []})
        assert exp.config.agent.tools == []


class TestGraderCombinations:
    def test_single_grader(self):
        exp = _create(graders={"rules": [
            {"name": "latency_check", "type": "latency_budget_ms", "params": {"max_ms": 5000}},
        ]})
        assert len(exp.config.graders.rules) == 1

    def test_multiple_graders(self):
        exp = _create(graders={"rules": [
            {"name": "latency_check", "type": "latency_budget_ms", "params": {"max_ms": 5000}},
            {"name": "f1_gate", "type": "min_f1_score", "params": {"min": 0.5}},
            {"name": "max_turns_gate", "type": "max_turns", "params": {"max": 5}},
        ]})
        assert len(exp.config.graders.rules) == 3

    def test_duplicate_grader_names_rejected(self):
        with pytest.raises(ValidationError, match="Grader names must be unique"):
            _create(graders={"rules": [
                {"name": "check", "type": "latency_budget_ms", "params": {"max_ms": 5000}},
                {"name": "check", "type": "min_f1_score", "params": {"min": 0.5}},
            ]})
