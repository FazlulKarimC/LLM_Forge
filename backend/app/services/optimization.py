"""
Inference Optimization Services — Phase 8

Provides:
- PromptCache: LRU cache for deterministic prompt results
- ProfilerContext: Context-managed timing collector for execution phases
- OptimizationReport: Aggregated optimization metrics for an experiment
"""

import hashlib
import time
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import numpy as np

from app.services.inference.base import GenerationResult


# =============================================================================
# Prompt Cache — LRU cache for identical prompts
# =============================================================================

class PromptCache:
    """
    Thread-safe LRU cache for generation results.

    Cache key is derived from (prompt, model, max_tokens, temperature, seed).
    Only useful when same prompts run with same config (e.g. reproducibility
    runs, or RAG queries with identical retrieved context).
    """

    def __init__(self, max_size: int = 256):
        """
        Args:
            max_size: Maximum number of cached results before eviction.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, GenerationResult] = OrderedDict()
        self._lock = threading.Lock()

        # Stats
        self.hits = 0
        self.misses = 0
        self.total_latency_saved_ms = 0.0

    @staticmethod
    def _make_key(
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        seed: Optional[int],
    ) -> str:
        """Generate deterministic cache key from prompt + generation params."""
        raw = f"{prompt}|{model}|{max_tokens}|{temperature}|{seed}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        seed: Optional[int],
    ) -> Optional[GenerationResult]:
        """
        Look up a cached result. Returns None on miss.

        On hit, moves entry to end (most-recently-used) and increments stats.
        """
        key = self._make_key(prompt, model, max_tokens, temperature, seed)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                result = self._cache[key]
                self.total_latency_saved_ms += result.latency_ms
                return result
            self.misses += 1
            return None

    def put(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        seed: Optional[int],
        result: GenerationResult,
    ) -> None:
        """Store a generation result. Evicts LRU entry if at capacity."""
        key = self._make_key(prompt, model, max_tokens, temperature, seed)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)  # evict oldest
                self._cache[key] = result

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 – 1.0). Returns 0 if no lookups yet."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics as a dict."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "size": self.size,
            "max_size": self.max_size,
            "total_latency_saved_ms": round(self.total_latency_saved_ms, 2),
        }

    def clear(self) -> None:
        """Reset cache and stats."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.total_latency_saved_ms = 0.0


# =============================================================================
# Profiler — Context-managed section timer
# =============================================================================

class ProfilerContext:
    """
    Collects fine-grained timing for each execution phase.

    Usage:
        profiler = ProfilerContext()
        with profiler.section("api_call"):
            result = engine.generate(prompt, config)
        summary = profiler.summary()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._sections: Dict[str, List[float]] = {}  # section -> list of durations (ms)
        self._lock = threading.Lock()

    @contextmanager
    def section(self, name: str):
        """
        Time a named section. Yields control, then records elapsed time.

        If profiling is disabled, this is a no-op passthrough.
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                self._sections.setdefault(name, []).append(elapsed_ms)

    def add_timing(self, name: str, elapsed_ms: float) -> None:
        """Manually add a timing entry (for cases where context manager doesn't fit)."""
        if not self.enabled:
            return
        with self._lock:
            self._sections.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate per-section timings.

        Returns dict like:
        {
            "api_call": {"count": 100, "total_ms": 52000, "mean_ms": 520, "p50_ms": 510, "p95_ms": 780},
            "parsing":  {"count": 100, "total_ms": 200,   "mean_ms": 2,   "p50_ms": 1.8, "p95_ms": 3.2},
        }
        """
        result = {}
        for name, durations in self._sections.items():
            arr = np.array(durations)
            result[name] = {
                "count": len(durations),
                "total_ms": round(float(arr.sum()), 2),
                "mean_ms": round(float(arr.mean()), 2),
                "p50_ms": round(float(np.percentile(arr, 50)), 2),
                "p95_ms": round(float(np.percentile(arr, 95)), 2),
            }
        return result

    def section_names(self) -> List[str]:
        """Names of all recorded sections."""
        return list(self._sections.keys())


# =============================================================================
# Optimization Report — experiment-level summary
# =============================================================================

@dataclass
class OptimizationReport:
    """
    Aggregated optimization metrics for a single experiment execution.

    Stored in Result.raw_metrics["optimization"].
    """
    cache_stats: Dict[str, Any] = field(default_factory=dict)
    profiling_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    batch_stats: Dict[str, Any] = field(default_factory=dict)
    total_wall_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "cache_stats": self.cache_stats,
            "profiling_summary": self.profiling_summary,
            "batch_stats": self.batch_stats,
            "total_wall_time_ms": round(self.total_wall_time_ms, 2),
        }
