"""
Experiment Config Schema Validation Tests

Unit-level schema validation tests (no HTTP, no DB) for edge cases
in ExperimentConfig, AgentConfig, GradersConfig, RegressionConfig,
OptimizationConfig, and RoutingConfig.
"""

import pytest
from pydantic import ValidationError

from app.schemas.experiment import (
    ExperimentConfig,
    AgentConfig,
    GradersConfig,
    GraderRule,
    GraderType,
    HyperParameters,
    OptimizationConfig,
    RAGConfig,
    RegressionConfig,
    ReasoningMethod,
    RetrievalMethod,
    RoutingConfig,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _base_config(**overrides) -> dict:
    """Minimal valid ExperimentConfig kwargs."""
    defaults = {
        "model_name": "test-model",
        "dataset_name": "sample",
        "reasoning_method": "naive",
    }
    defaults.update(overrides)
    return defaults


# ── AgentConfig ──────────────────────────────────────────────────────────

class TestAgentConfig:
    """Agent config is only valid with ReAct reasoning."""

    def test_agent_requires_react(self):
        with pytest.raises(ValidationError, match="Agent config only valid for ReAct method"):
            ExperimentConfig(
                **_base_config(
                    reasoning_method="naive",
                    agent=AgentConfig(max_iterations=3, tools=["calculator"]),
                )
            )

    def test_agent_requires_react_cot(self):
        with pytest.raises(ValidationError, match="Agent config only valid for ReAct method"):
            ExperimentConfig(
                **_base_config(
                    reasoning_method="cot",
                    agent=AgentConfig(max_iterations=3, tools=["calculator"]),
                )
            )

    def test_agent_accepted_with_react(self):
        cfg = ExperimentConfig(
            **_base_config(
                reasoning_method="react",
                agent=AgentConfig(max_iterations=3, tools=["calculator"]),
            )
        )
        assert cfg.agent is not None
        assert cfg.agent.tools == ["calculator"]

    def test_empty_tools_accepted(self):
        """An empty tools list is fine — the agent can still run."""
        cfg = ExperimentConfig(
            **_base_config(
                reasoning_method="react",
                agent=AgentConfig(max_iterations=3, tools=[]),
            )
        )
        assert cfg.agent.tools == []

    def test_too_many_tools_rejected(self):
        with pytest.raises(ValidationError, match="At most 10 tools"):
            AgentConfig(max_iterations=3, tools=[f"tool_{i}" for i in range(11)])


# ── HyperParameters ─────────────────────────────────────────────────────

class TestHyperParameters:
    """Temperature and token boundary validation."""

    def test_temperature_zero_accepted(self):
        hp = HyperParameters(temperature=0.0)
        assert hp.temperature == 0.0

    def test_temperature_two_accepted(self):
        hp = HyperParameters(temperature=2.0)
        assert hp.temperature == 2.0

    def test_temperature_over_two_rejected(self):
        with pytest.raises(ValidationError):
            HyperParameters(temperature=2.1)

    def test_temperature_negative_rejected(self):
        with pytest.raises(ValidationError):
            HyperParameters(temperature=-0.1)

    def test_max_tokens_one_accepted(self):
        hp = HyperParameters(max_tokens=1)
        assert hp.max_tokens == 1

    def test_max_tokens_zero_rejected(self):
        with pytest.raises(ValidationError):
            HyperParameters(max_tokens=0)


# ── num_samples ──────────────────────────────────────────────────────────

class TestNumSamples:
    """Dataset sampling boundary validation."""

    def test_num_samples_one_accepted(self):
        cfg = ExperimentConfig(**_base_config(num_samples=1))
        assert cfg.num_samples == 1

    def test_num_samples_500_accepted(self):
        cfg = ExperimentConfig(**_base_config(num_samples=500))
        assert cfg.num_samples == 500

    def test_num_samples_zero_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(**_base_config(num_samples=0))

    def test_num_samples_501_rejected(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(**_base_config(num_samples=501))


# ── OptimizationConfig ──────────────────────────────────────────────────

class TestOptimizationConfig:
    """Batch size boundary validation."""

    def test_batch_size_one_accepted(self):
        opt = OptimizationConfig(enable_batching=True, batch_size=1)
        assert opt.batch_size == 1

    def test_batch_size_32_accepted(self):
        opt = OptimizationConfig(enable_batching=True, batch_size=32)
        assert opt.batch_size == 32

    def test_batch_size_zero_rejected(self):
        with pytest.raises(ValidationError):
            OptimizationConfig(enable_batching=True, batch_size=0)

    def test_batch_size_33_rejected(self):
        with pytest.raises(ValidationError):
            OptimizationConfig(enable_batching=True, batch_size=33)


# ── GradersConfig ────────────────────────────────────────────────────────

class TestGradersConfig:
    """Grader rules validation."""

    def test_duplicate_grader_names_rejected(self):
        with pytest.raises(ValidationError, match="Grader names must be unique"):
            GradersConfig(rules=[
                GraderRule(name="latency", type=GraderType.LATENCY_BUDGET_MS, params={"max_ms": 5000}),
                GraderRule(name="latency", type=GraderType.TOKEN_BUDGET, params={"max_tokens": 200}),
            ])

    def test_too_many_grader_rules_rejected(self):
        with pytest.raises(ValidationError, match="At most 10 grader rules"):
            GradersConfig(rules=[
                GraderRule(name=f"grader_{i}", type=GraderType.LATENCY_BUDGET_MS, params={"max_ms": 5000})
                for i in range(11)
            ])

    def test_valid_graders_accepted(self):
        gc = GradersConfig(rules=[
            GraderRule(name="latency_check", type=GraderType.LATENCY_BUDGET_MS, params={"max_ms": 5000}),
            GraderRule(name="f1_check", type=GraderType.MIN_F1_SCORE, params={"min": 0.5}),
        ])
        assert len(gc.rules) == 2


# ── RegressionConfig ────────────────────────────────────────────────────

class TestRegressionConfig:
    """Regression gate validation."""

    def test_min_overlap_ratio_zero_accepted(self):
        rc = RegressionConfig(min_overlap_ratio=0.0)
        assert rc.min_overlap_ratio == 0.0

    def test_min_overlap_ratio_one_accepted(self):
        rc = RegressionConfig(min_overlap_ratio=1.0)
        assert rc.min_overlap_ratio == 1.0

    def test_min_overlap_ratio_over_one_rejected(self):
        with pytest.raises(ValidationError):
            RegressionConfig(min_overlap_ratio=1.1)

    def test_min_overlap_ratio_negative_rejected(self):
        with pytest.raises(ValidationError):
            RegressionConfig(min_overlap_ratio=-0.1)

    def test_negative_accuracy_delta_accepted(self):
        """Negative deltas are intentional — represent tolerated degradation."""
        rc = RegressionConfig(accuracy_min_delta=-0.10)
        assert rc.accuracy_min_delta == -0.10


# ── RoutingConfig ────────────────────────────────────────────────────────

class TestRoutingConfig:
    """Routing policy validation."""

    @pytest.mark.parametrize("policy", [
        "fallback_chain", "cheapest_first", "fastest_first", "adaptive",
    ])
    def test_valid_policies_accepted(self, policy):
        rc = RoutingConfig(policy=policy)
        assert rc.policy == policy

    def test_epsilon_boundaries(self):
        RoutingConfig(epsilon=0.0)
        RoutingConfig(epsilon=1.0)
        with pytest.raises(ValidationError):
            RoutingConfig(epsilon=1.1)
        with pytest.raises(ValidationError):
            RoutingConfig(epsilon=-0.01)
