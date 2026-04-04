"""
Tests for ProviderStatsTracker and ProviderRouter.
"""

import threading
from dataclasses import dataclass
from typing import List, Optional

from app.models.run import FailureMode
from app.services.inference.base import GenerationConfig, GenerationResult
from app.services.inference.provider_router import ProviderRouter
from app.services.inference.provider_stats import ProviderStatsTracker, RoutingPolicy


class TestProviderStatsTracker:
    def test_record_and_summary(self):
        tracker = ProviderStatsTracker()
        tracker.record("engine_a", 100.0, 50, 30, 0.001, False)
        tracker.record("engine_a", 200.0, 60, 40, 0.002, False)
        tracker.record("engine_b", 300.0, 70, 50, 0.003, True)

        summary = tracker.summary()
        assert summary["engine_a"]["total_requests"] == 2
        assert summary["engine_a"]["total_cost_usd"] == 0.003
        assert summary["engine_b"]["total_errors"] == 1
        assert summary["engine_b"]["error_rate"] == 1.0

    def test_recommend_cheapest(self):
        tracker = ProviderStatsTracker()
        tracker.record("cheap", 200.0, 50, 30, 0.001, False)
        tracker.record("expensive", 100.0, 50, 30, 0.010, False)

        assert tracker.recommend(RoutingPolicy.CHEAPEST_FIRST, ["cheap", "expensive"]) == "cheap"

    def test_recommend_fastest(self):
        tracker = ProviderStatsTracker()
        tracker.record("slow", 500.0, 50, 30, 0.001, False)
        tracker.record("fast", 100.0, 50, 30, 0.010, False)

        assert tracker.recommend(RoutingPolicy.FASTEST_FIRST, ["slow", "fast"]) == "fast"

    def test_recommend_adaptive_uses_composite_score(self):
        tracker = ProviderStatsTracker()
        tracker.record("balanced", 120.0, 50, 30, 0.001, False)
        tracker.record("slow_but_cheaper", 400.0, 50, 30, 0.0, False)
        tracker.record("flaky_fast", 80.0, 50, 30, 0.002, True)

        assert tracker.recommend(
            RoutingPolicy.ADAPTIVE,
            ["balanced", "slow_but_cheaper", "flaky_fast"],
        ) == "balanced"

    def test_recommend_no_stats_returns_first_available(self):
        tracker = ProviderStatsTracker()
        assert tracker.recommend(RoutingPolicy.CHEAPEST_FIRST, ["engine_a", "engine_b"]) == "engine_a"

    def test_recommend_empty_available(self):
        tracker = ProviderStatsTracker()
        assert tracker.recommend(RoutingPolicy.CHEAPEST_FIRST, []) is None

    def test_thread_safety(self):
        tracker = ProviderStatsTracker()
        errors = []

        def worker(name: str, count: int):
            try:
                for i in range(count):
                    tracker.record(name, float(i), 10, 5, 0.001, i % 10 == 0)
            except Exception as exc:  # pragma: no cover - defensive test harness
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=("engine_a", 100)),
            threading.Thread(target=worker, args=("engine_b", 100)),
            threading.Thread(target=worker, args=("engine_c", 100)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        summary = tracker.summary()
        assert summary["engine_a"]["total_requests"] == 100
        assert summary["engine_b"]["total_requests"] == 100

    def test_from_historical(self):
        historical = {
            "openrouter": {
                "total_requests": 50,
                "total_errors": 2,
                "mean_latency_ms": 150.0,
                "total_tokens": 5000,
                "total_cost_usd": 0.05,
            },
            "groq": {
                "total_requests": 30,
                "total_errors": 0,
                "mean_latency_ms": 100.0,
                "total_tokens": 3000,
                "total_cost_usd": 0.01,
            },
        }
        tracker = ProviderStatsTracker.from_historical(historical)
        summary = tracker.summary()

        assert summary["openrouter"]["total_requests"] == 50
        assert summary["groq"]["total_requests"] == 30
        assert summary["groq"]["total_errors"] == 0


@dataclass
class MockEngine:
    """Minimal inference-engine double for router tests."""

    provider_id: str
    name: str
    latency: float = 100.0
    cost_usd: float = 0.0
    fail_with_rate_limit: bool = False
    fail_with_exception: bool = False
    _is_loaded: bool = True

    def load_model(self, model_name: str) -> None:
        self._is_loaded = True

    def unload_model(self) -> None:
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_name(self) -> Optional[str]:
        return "test-model"

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        if self.fail_with_exception:
            raise ConnectionError("Connection refused")

        if self.fail_with_rate_limit:
            return GenerationResult(
                text="",
                tokens_input=10,
                tokens_output=0,
                latency_ms=self.latency,
                finish_reason="error",
                cost_usd=0.0,
                failure_mode=FailureMode.API_ERROR,
                error_message="429 Rate limit exceeded",
            )

        return GenerationResult(
            text=f"response from {self.name}",
            tokens_input=10,
            tokens_output=20,
            latency_ms=self.latency,
            finish_reason="stop",
            cost_usd=self.cost_usd,
        )

    def generate_batch(
        self,
        prompts: List[str],
        config: GenerationConfig,
        max_workers: int = 8,
    ) -> List[GenerationResult]:
        return [self.generate(prompt, config) for prompt in prompts]


class TestProviderRouter:
    def test_fallback_chain_uses_first_available(self):
        router = ProviderRouter(
            [
                MockEngine(provider_id="hf_api", name="Primary"),
                MockEngine(provider_id="openrouter", name="Secondary"),
            ],
            RoutingPolicy.FALLBACK_CHAIN,
        )
        router.load_model("test-model")

        result = router.generate("hello", GenerationConfig())

        assert "Primary" in result.text
        assert result.served_provider == "hf_api"
        assert result.routing_reason == "selected"

    def test_fallback_on_rate_limit(self):
        router = ProviderRouter(
            [
                MockEngine(provider_id="hf_api", name="Primary", fail_with_rate_limit=True),
                MockEngine(provider_id="groq", name="Secondary"),
            ],
            RoutingPolicy.FALLBACK_CHAIN,
        )
        router.load_model("test-model")

        result = router.generate("hello", GenerationConfig())

        assert "Secondary" in result.text
        assert result.served_provider == "groq"
        assert result.routing_reason == "fallback_1"

    def test_cheapest_first_selects_lowest_cost(self):
        tracker = ProviderStatsTracker()
        tracker.record("openrouter", 150.0, 10, 20, 0.002, False)
        tracker.record("groq", 90.0, 10, 20, 0.0005, False)

        router = ProviderRouter(
            [
                MockEngine(provider_id="openrouter", name="Expensive", cost_usd=0.002),
                MockEngine(provider_id="groq", name="Cheap", cost_usd=0.0005),
            ],
            RoutingPolicy.CHEAPEST_FIRST,
            tracker,
        )
        router.load_model("test-model")

        result = router.generate("hello", GenerationConfig())

        assert "Cheap" in result.text
        assert result.served_provider == "groq"

    def test_fastest_first_selects_lowest_latency(self):
        tracker = ProviderStatsTracker()
        tracker.record("hf_api", 350.0, 10, 20, 0.0, False)
        tracker.record("groq", 50.0, 10, 20, 0.0, False)

        router = ProviderRouter(
            [
                MockEngine(provider_id="hf_api", name="Slow", latency=350.0),
                MockEngine(provider_id="groq", name="Fast", latency=50.0),
            ],
            RoutingPolicy.FASTEST_FIRST,
            tracker,
        )
        router.load_model("test-model")

        result = router.generate("hello", GenerationConfig())

        assert "Fast" in result.text
        assert result.served_provider == "groq"

    def test_adaptive_explores_then_exploits(self):
        router = ProviderRouter(
            [
                MockEngine(provider_id="hf_api", name="Slow", latency=300.0, cost_usd=0.001),
                MockEngine(provider_id="groq", name="Fast", latency=50.0, cost_usd=0.0),
            ],
            RoutingPolicy.ADAPTIVE,
            epsilon=0.0,
            exploration_window=2,
        )
        router.load_model("test-model")

        first = router.generate("one", GenerationConfig())
        second = router.generate("two", GenerationConfig())
        third = router.generate("three", GenerationConfig())

        assert first.served_provider == "hf_api"
        assert second.served_provider == "groq"
        assert third.served_provider == "groq"
        assert third.routing_reason == "adaptive_selected"

    def test_cost_recorded_in_stats(self):
        router = ProviderRouter(
            [MockEngine(provider_id="openrouter", name="Costed", cost_usd=0.123456)],
            RoutingPolicy.FALLBACK_CHAIN,
        )
        router.load_model("test-model")

        router.generate("hello", GenerationConfig())

        summary = router.stats_tracker.summary()
        assert summary["openrouter"]["total_cost_usd"] == 0.123456
        assert summary["openrouter"]["cost_per_request"] == 0.123456

    def test_batch_fallback_chain_uses_per_prompt_fallback_with_multiple_providers(self):
        router = ProviderRouter(
            [
                MockEngine(provider_id="hf_api", name="Primary", fail_with_rate_limit=True),
                MockEngine(provider_id="groq", name="Secondary"),
            ],
            RoutingPolicy.FALLBACK_CHAIN,
        )
        router.load_model("test-model")

        results = router.generate_batch(["a", "b", "c"], GenerationConfig())

        assert len(results) == 3
        assert all(result.served_provider == "groq" for result in results)
        summary = router.stats_tracker.summary()
        assert summary["hf_api"]["total_requests"] == 3
        assert summary["hf_api"]["total_errors"] == 3
        assert summary["groq"]["total_requests"] == 3

    def test_all_providers_exhausted(self):
        router = ProviderRouter(
            [
                MockEngine(provider_id="hf_api", name="Primary", fail_with_rate_limit=True),
                MockEngine(provider_id="groq", name="Secondary", fail_with_rate_limit=True),
            ],
            RoutingPolicy.FALLBACK_CHAIN,
        )
        router.load_model("test-model")

        result = router.generate("hello", GenerationConfig())

        assert result.failure_mode == FailureMode.API_ERROR
        assert "All providers exhausted" in (result.error_message or "")
