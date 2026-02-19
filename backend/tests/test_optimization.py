"""
Unit tests for Phase 8: Optimization components.

Tests for PromptCache, ProfilerContext, and OptimizationReport.
"""

import time
import pytest

from app.services.optimization import PromptCache, ProfilerContext, OptimizationReport
from app.services.inference.base import GenerationResult


# =============================================================================
# Helpers
# =============================================================================

# Common params for cache lookups
PROMPT = "What is the capital of France?"
MODEL = "test-model"
MAX_TOKENS = 100
TEMP = 0.5
SEED = 42


def _make_result(text: str = "answer", latency: float = 100.0) -> GenerationResult:
    """Create a mock GenerationResult for testing."""
    return GenerationResult(
        text=text,
        tokens_input=10,
        tokens_output=5,
        latency_ms=latency,
        finish_reason="stop",
    )


# =============================================================================
# PromptCache Tests
# =============================================================================

class TestPromptCache:
    def test_basic_put_get(self):
        cache = PromptCache(max_size=10)
        result = _make_result("hello")
        cache.put(PROMPT, MODEL, MAX_TOKENS, TEMP, SEED, result)

        cached = cache.get(PROMPT, MODEL, MAX_TOKENS, TEMP, SEED)
        assert cached is not None
        assert cached.text == "hello"

    def test_cache_miss_returns_none(self):
        cache = PromptCache(max_size=10)
        assert cache.get("nonexistent", MODEL, MAX_TOKENS, TEMP, SEED) is None

    def test_cache_hit_miss_stats(self):
        cache = PromptCache(max_size=10)
        result = _make_result()

        cache.put(PROMPT, MODEL, MAX_TOKENS, TEMP, SEED, result)
        cache.get(PROMPT, MODEL, MAX_TOKENS, TEMP, SEED)       # hit
        cache.get(PROMPT, MODEL, MAX_TOKENS, TEMP, SEED)       # hit
        cache.get("missing", MODEL, MAX_TOKENS, TEMP, SEED)    # miss

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_lru_eviction(self):
        cache = PromptCache(max_size=3)
        for i in range(4):
            cache.put(f"prompt_{i}", MODEL, MAX_TOKENS, TEMP, SEED, _make_result(f"val{i}"))

        # prompt_0 should have been evicted (oldest)
        assert cache.get("prompt_0", MODEL, MAX_TOKENS, TEMP, SEED) is None
        assert cache.get("prompt_1", MODEL, MAX_TOKENS, TEMP, SEED) is not None
        assert cache.get("prompt_2", MODEL, MAX_TOKENS, TEMP, SEED) is not None
        assert cache.get("prompt_3", MODEL, MAX_TOKENS, TEMP, SEED) is not None

    def test_lru_access_refreshes_order(self):
        cache = PromptCache(max_size=3)
        cache.put("a", MODEL, MAX_TOKENS, TEMP, SEED, _make_result("a"))
        cache.put("b", MODEL, MAX_TOKENS, TEMP, SEED, _make_result("b"))
        cache.put("c", MODEL, MAX_TOKENS, TEMP, SEED, _make_result("c"))

        # Access 'a' to refresh it, making 'b' the oldest
        cache.get("a", MODEL, MAX_TOKENS, TEMP, SEED)
        cache.put("d", MODEL, MAX_TOKENS, TEMP, SEED, _make_result("d"))

        assert cache.get("b", MODEL, MAX_TOKENS, TEMP, SEED) is None     # b was evicted
        assert cache.get("a", MODEL, MAX_TOKENS, TEMP, SEED) is not None  # a was refreshed

    def test_make_key_deterministic(self):
        key1 = PromptCache._make_key("hello", MODEL, MAX_TOKENS, 0.5, SEED)
        key2 = PromptCache._make_key("hello", MODEL, MAX_TOKENS, 0.5, SEED)
        assert key1 == key2

    def test_make_key_different_params(self):
        key1 = PromptCache._make_key("hello", MODEL, MAX_TOKENS, 0.5, SEED)
        key2 = PromptCache._make_key("hello", MODEL, MAX_TOKENS, 0.7, SEED)
        assert key1 != key2

    def test_latency_saved_tracking(self):
        cache = PromptCache(max_size=10)
        cache.put(PROMPT, MODEL, MAX_TOKENS, TEMP, SEED, _make_result(latency=200.0))
        cache.get(PROMPT, MODEL, MAX_TOKENS, TEMP, SEED)  # hit — saves 200ms
        cache.get(PROMPT, MODEL, MAX_TOKENS, TEMP, SEED)  # hit — saves 200ms

        stats = cache.stats()
        assert stats["total_latency_saved_ms"] == pytest.approx(400.0, abs=1.0)

    def test_stats_size_and_max(self):
        cache = PromptCache(max_size=5)
        cache.put("a", MODEL, MAX_TOKENS, TEMP, SEED, _make_result())
        cache.put("b", MODEL, MAX_TOKENS, TEMP, SEED, _make_result())

        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 5


# =============================================================================
# ProfilerContext Tests
# =============================================================================

class TestProfilerContext:
    def test_section_timing(self):
        profiler = ProfilerContext()
        with profiler.section("test_phase"):
            time.sleep(0.05)

        result = profiler.summary()
        assert "test_phase" in result
        assert result["test_phase"]["count"] == 1
        assert result["test_phase"]["total_ms"] >= 40  # Allow some slack

    def test_multiple_sections(self):
        profiler = ProfilerContext()
        for _ in range(3):
            with profiler.section("api_call"):
                time.sleep(0.01)

        result = profiler.summary()
        assert result["api_call"]["count"] == 3
        assert result["api_call"]["total_ms"] >= 25

    def test_disabled_profiler_noop(self):
        profiler = ProfilerContext(enabled=False)
        with profiler.section("no_track"):
            time.sleep(0.01)

        result = profiler.summary()
        assert len(result) == 0

    def test_percentiles_in_summary(self):
        profiler = ProfilerContext()
        for _ in range(10):
            with profiler.section("fast"):
                time.sleep(0.005)

        result = profiler.summary()
        assert "p50_ms" in result["fast"]
        assert "p95_ms" in result["fast"]
        assert "mean_ms" in result["fast"]

    def test_empty_summary(self):
        profiler = ProfilerContext()
        result = profiler.summary()
        assert result == {}


# =============================================================================
# OptimizationReport Tests
# =============================================================================

class TestOptimizationReport:
    def test_to_dict(self):
        report = OptimizationReport(
            cache_stats={"hits": 5, "misses": 10},
            profiling_summary={"api_call": {"count": 15, "total_ms": 3000}},
            batch_stats={"batches_processed": 3, "total_prompts_batched": 15},
            total_wall_time_ms=5000.0,
        )
        d = report.to_dict()
        assert d["cache_stats"]["hits"] == 5
        assert d["total_wall_time_ms"] == 5000.0
        assert d["batch_stats"]["batches_processed"] == 3

    def test_to_dict_empty(self):
        report = OptimizationReport()
        d = report.to_dict()
        assert d["cache_stats"] == {}
        assert d["profiling_summary"] == {}
        assert d["batch_stats"] == {}
        assert d["total_wall_time_ms"] == 0.0
