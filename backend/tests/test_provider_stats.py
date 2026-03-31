"""
Tests for ProviderStatsTracker and ProviderRouter — Phase 4.

Covers:
- Stats recording and summary
- Policy-based recommendations (cheapest, fastest)
- Adaptive epsilon-greedy selection
- Warm-start from historical data
- Thread safety
- Router fallback behavior
- Batch routing for non-fallback policies
- served_provider tagging
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from app.services.inference.provider_stats import ProviderStatsTracker, RoutingPolicy


# ─── ProviderStatsTracker Tests ──────────────────────────────────────────────

class TestProviderStatsTracker:
    def test_record_and_summary(self):
        tracker = ProviderStatsTracker()
        tracker.record("engine_a", 100.0, 50, 30, 0.001, False)
        tracker.record("engine_a", 200.0, 60, 40, 0.002, False)
        tracker.record("engine_b", 300.0, 70, 50, 0.003, True)
        
        summary = tracker.summary()
        assert "engine_a" in summary
        assert "engine_b" in summary
        assert summary["engine_a"]["total_requests"] == 2
        assert summary["engine_a"]["total_errors"] == 0
        assert summary["engine_b"]["total_errors"] == 1
        assert summary["engine_b"]["error_rate"] == 1.0

    def test_recommend_cheapest(self):
        tracker = ProviderStatsTracker()
        tracker.record("cheap", 200.0, 50, 30, 0.001, False)
        tracker.record("expensive", 100.0, 50, 30, 0.010, False)
        
        result = tracker.recommend(RoutingPolicy.CHEAPEST_FIRST, ["cheap", "expensive"])
        assert result == "cheap"

    def test_recommend_cheapest_tiebreak_latency(self):
        """When cost is equal (free tier), tie-break by latency."""
        tracker = ProviderStatsTracker()
        tracker.record("slow", 500.0, 50, 30, 0.0, False)
        tracker.record("fast", 100.0, 50, 30, 0.0, False)
        
        result = tracker.recommend(RoutingPolicy.CHEAPEST_FIRST, ["slow", "fast"])
        assert result == "fast"

    def test_recommend_fastest(self):
        tracker = ProviderStatsTracker()
        tracker.record("slow", 500.0, 50, 30, 0.001, False)
        tracker.record("fast", 100.0, 50, 30, 0.010, False)
        
        result = tracker.recommend(RoutingPolicy.FASTEST_FIRST, ["slow", "fast"])
        assert result == "fast"

    def test_recommend_no_stats(self):
        """No stats → return first available."""
        tracker = ProviderStatsTracker()
        result = tracker.recommend(RoutingPolicy.CHEAPEST_FIRST, ["engine_a", "engine_b"])
        assert result == "engine_a"

    def test_recommend_empty_available(self):
        tracker = ProviderStatsTracker()
        result = tracker.recommend(RoutingPolicy.CHEAPEST_FIRST, [])
        assert result is None

    def test_thread_safety(self):
        """Multiple threads recording concurrently should not crash."""
        tracker = ProviderStatsTracker()
        errors = []
        
        def worker(name, n):
            try:
                for i in range(n):
                    tracker.record(name, float(i), 10, 5, 0.001, i % 10 == 0)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=worker, args=("engine_a", 100)),
            threading.Thread(target=worker, args=("engine_b", 100)),
            threading.Thread(target=worker, args=("engine_c", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        summary = tracker.summary()
        assert summary["engine_a"]["total_requests"] == 100
        assert summary["engine_b"]["total_requests"] == 100

    def test_from_historical(self):
        """Warm-start recreates stats from stored data."""
        historical = {
            "EngineA": {
                "total_requests": 50,
                "total_errors": 2,
                "mean_latency_ms": 150.0,
                "total_tokens": 5000,
                "total_cost_usd": 0.05,
            },
            "EngineB": {
                "total_requests": 30,
                "total_errors": 0,
                "mean_latency_ms": 100.0,
                "total_tokens": 3000,
                "total_cost_usd": 0.01,
            },
        }
        tracker = ProviderStatsTracker.from_historical(historical)
        summary = tracker.summary()
        
        assert summary["EngineA"]["total_requests"] == 50
        assert summary["EngineB"]["total_requests"] == 30
        assert summary["EngineB"]["total_errors"] == 0

    def test_p95_latency(self):
        tracker = ProviderStatsTracker()
        for i in range(100):
            tracker.record("test", float(i), 10, 5, 0.0, False)
        
        summary = tracker.summary()
        # p95 should be around 95
        assert 90 <= summary["test"]["p95_latency_ms"] <= 99


# ─── RoutingPolicy Enum Tests ───────────────────────────────────────────────

class TestRoutingPolicy:
    def test_all_values(self):
        assert RoutingPolicy.FALLBACK_CHAIN == "fallback_chain"
        assert RoutingPolicy.CHEAPEST_FIRST == "cheapest_first"
        assert RoutingPolicy.FASTEST_FIRST == "fastest_first"
        assert RoutingPolicy.ADAPTIVE == "adaptive"

    def test_from_string(self):
        assert RoutingPolicy("cheapest_first") == RoutingPolicy.CHEAPEST_FIRST


# ─── ProviderRouter with Mock Engines ───────────────────────────────────────

@dataclass
class MockEngine:
    """Minimal InferenceEngine mock for router tests."""
    name: str = "MockEngine"
    _is_loaded: bool = True
    _fail: bool = False
    _fail_with_rate_limit: bool = False
    latency: float = 100.0
    
    def load_model(self, model_name):
        self._is_loaded = True
    
    def unload_model(self):
        self._is_loaded = False
    
    @property
    def is_loaded(self):
        return self._is_loaded
    
    @property
    def model_name(self):
        return "test-model"
    
    def generate(self, prompt, config):
        from app.services.inference.base import GenerationResult
        from app.models.run import FailureMode as _FM
        
        if self._fail:
            raise ConnectionError("Connection refused")
        
        if self._fail_with_rate_limit:
            return GenerationResult(
                text="",
                tokens_input=10,
                tokens_output=0,
                latency_ms=self.latency,
                finish_reason="error",
                failure_mode=_FM.API_ERROR,
                error_message="429 Rate limit exceeded",
            )
        
        return GenerationResult(
            text=f"response from {self.name}",
            tokens_input=10,
            tokens_output=20,
            latency_ms=self.latency,
            finish_reason="stop",
        )
    
    def generate_batch(self, prompts, config, max_workers=8):
        return [self.generate(p, config) for p in prompts]


class TestProviderRouter:
    def test_fallback_chain_first_engine(self):
        from app.services.inference.provider_router import ProviderRouter
        
        e1 = MockEngine(name="Primary")
        e2 = MockEngine(name="Secondary")
        router = ProviderRouter([e1, e2], RoutingPolicy.FALLBACK_CHAIN)
        router.load_model("test")
        
        from app.services.inference.base import GenerationConfig
        result = router.generate("hello", GenerationConfig())
        assert "Primary" in result.text
        assert result.served_provider is not None

    def test_fallback_on_rate_limit(self):
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig
        
        e1 = MockEngine(name="Primary", _fail_with_rate_limit=True)
        e2 = MockEngine(name="Secondary")
        router = ProviderRouter([e1, e2], RoutingPolicy.FALLBACK_CHAIN)
        router.load_model("test")
        
        result = router.generate("hello", GenerationConfig())
        assert "Secondary" in result.text

    def test_cheapest_first_policy(self):
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig
        
        e1 = MockEngine(name="Expensive", latency=100.0)
        e2 = MockEngine(name="Cheap", latency=50.0)
        
        # Pre-populate stats so cheapest is known
        tracker = ProviderStatsTracker()
        tracker.record("MockEngine", 100.0, 10, 20, 0.01, False)
        # Second engine is same class name — they'll both be "MockEngine"
        # In real usage, engines are different classes
        
        router = ProviderRouter([e1, e2], RoutingPolicy.CHEAPEST_FIRST, tracker)
        router.load_model("test")
        
        result = router.generate("hello", GenerationConfig())
        assert result.served_provider is not None

    def test_served_provider_tagged(self):
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig
        
        e1 = MockEngine(name="Primary")
        router = ProviderRouter([e1], RoutingPolicy.FALLBACK_CHAIN)
        router.load_model("test")
        
        result = router.generate("hello", GenerationConfig())
        assert result.served_provider == "MockEngine"
        assert result.routing_reason == "selected"

    def test_batch_fallback_chain_delegates(self):
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig
        
        e1 = MockEngine(name="Primary")
        router = ProviderRouter([e1], RoutingPolicy.FALLBACK_CHAIN)
        router.load_model("test")
        
        results = router.generate_batch(["a", "b", "c"], GenerationConfig())
        assert len(results) == 3
        assert all(r.served_provider == "MockEngine" for r in results)

    def test_batch_fallback_chain_uses_per_prompt_fallback_with_multiple_providers(self):
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig

        primary = MockEngine(name="Primary", _fail_with_rate_limit=True)
        secondary = MockEngine(name="Secondary")
        router = ProviderRouter([primary, secondary], RoutingPolicy.FALLBACK_CHAIN)
        router.load_model("test")

        results = router.generate_batch(["a", "b", "c"], GenerationConfig())

        assert len(results) == 3
        assert all("Secondary" in result.text for result in results)
        summary = router.stats_tracker.summary()
        assert summary["MockEngine"]["total_requests"] == 6
        assert summary["MockEngine"]["total_errors"] == 3

    def test_stats_summary_after_requests(self):
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig
        
        e1 = MockEngine(name="Primary")
        router = ProviderRouter([e1], RoutingPolicy.FALLBACK_CHAIN)
        router.load_model("test")
        
        for _ in range(5):
            router.generate("hello", GenerationConfig())
        
        summary = router.stats_tracker.summary()
        assert "MockEngine" in summary
        assert summary["MockEngine"]["total_requests"] == 5

    def test_all_providers_exhausted(self):
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig
        
        e1 = MockEngine(name="Primary", _fail_with_rate_limit=True)
        e2 = MockEngine(name="Secondary", _fail_with_rate_limit=True)
        router = ProviderRouter([e1, e2], RoutingPolicy.FALLBACK_CHAIN)
        router.load_model("test")
        
        result = router.generate("hello", GenerationConfig())
        assert result.failure_mode is not None
        assert "All providers exhausted" in result.error_message
