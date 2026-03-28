"""
Provider Stats Tracker

Thread-safe in-memory tracker of per-provider performance metrics
for adaptive routing decisions.

Records latency, cost, error rates, and tokens per provider.
Recommends providers based on routing policy.
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RoutingPolicy(str, Enum):
    """Supported routing policies for provider selection."""
    FALLBACK_CHAIN = "fallback_chain"
    CHEAPEST_FIRST = "cheapest_first"
    FASTEST_FIRST = "fastest_first"
    ADAPTIVE = "adaptive"


@dataclass
class ProviderSnapshot:
    """Live statistics for a single provider."""
    name: str
    total_requests: int = 0
    total_errors: int = 0
    latencies: List[float] = field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    @property
    def mean_latency_ms(self) -> float:
        if not self.latencies:
            return float("inf")
        return sum(self.latencies) / len(self.latencies)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return float("inf")
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def cost_per_request(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_cost_usd / self.total_requests


class ProviderStatsTracker:
    """
    Thread-safe tracker for per-provider performance metrics.
    
    Used by ProviderRouter to make policy-driven routing decisions.
    All mutable state is guarded by a threading.Lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats: Dict[str, ProviderSnapshot] = {}

    def record(
        self,
        provider_name: str,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        is_error: bool,
    ) -> None:
        """Record a single request result. Thread-safe."""
        with self._lock:
            if provider_name not in self._stats:
                self._stats[provider_name] = ProviderSnapshot(name=provider_name)
            
            snap = self._stats[provider_name]
            snap.total_requests += 1
            if is_error:
                snap.total_errors += 1
            snap.latencies.append(latency_ms)
            snap.total_tokens += tokens_in + tokens_out
            snap.total_cost_usd += cost_usd

    def recommend(self, policy: RoutingPolicy, available: List[str]) -> Optional[str]:
        """
        Recommend a provider based on policy and collected stats.
        
        Args:
            policy: The routing policy to use
            available: List of available provider names
            
        Returns:
            Recommended provider name, or None if no data
        """
        with self._lock:
            if not available:
                return None
            
            # Only consider providers we have stats for
            known = {name for name in available if name in self._stats}
            
            # If we have no stats yet, return first available
            if not known:
                return available[0]
            
            if policy == RoutingPolicy.CHEAPEST_FIRST:
                return self._select_cheapest(known)
            elif policy == RoutingPolicy.FASTEST_FIRST:
                return self._select_fastest(known)
            elif policy == RoutingPolicy.ADAPTIVE:
                # Shouldn't reach here — adaptive uses epsilon-greedy in router
                return self._select_cheapest(known)
            else:
                return available[0]

    def _select_cheapest(self, candidates: set) -> str:
        """
        Select cheapest provider. Tie-break: latency, then error rate.
        
        On free tier, all costs are often $0.00, so tie-breaking is critical.
        """
        scored = []
        for name in candidates:
            snap = self._stats[name]
            scored.append((
                snap.cost_per_request,
                snap.mean_latency_ms,
                snap.error_rate,
                name,
            ))
        scored.sort()
        return scored[0][3]

    def _select_fastest(self, candidates: set) -> str:
        """Select fastest provider by mean latency. Tie-break: error rate."""
        scored = []
        for name in candidates:
            snap = self._stats[name]
            scored.append((
                snap.mean_latency_ms,
                snap.error_rate,
                name,
            ))
        scored.sort()
        return scored[0][2]

    def summary(self) -> Dict[str, Any]:
        """Produce a serializable summary for raw_metrics storage."""
        with self._lock:
            result = {}
            for name, snap in self._stats.items():
                result[name] = {
                    "total_requests": snap.total_requests,
                    "total_errors": snap.total_errors,
                    "error_rate": round(snap.error_rate, 4),
                    "mean_latency_ms": round(snap.mean_latency_ms, 1) if snap.latencies else None,
                    "p95_latency_ms": round(snap.p95_latency_ms, 1) if snap.latencies else None,
                    "total_tokens": snap.total_tokens,
                    "total_cost_usd": round(snap.total_cost_usd, 6),
                    "cost_per_request": round(snap.cost_per_request, 6),
                }
            return result

    @classmethod
    def from_historical(
        cls,
        routing_data: Dict[str, Any],
        bucket_key: Optional[str] = None,
    ) -> "ProviderStatsTracker":
        """
        Warm-start from a previous experiment's raw_metrics["routing"].
        
        Args:
            routing_data: Dict with provider names as keys, snapshot dicts as values
            bucket_key: Optional bucket filter (unused in v1, reserved for v2)
            
        Returns:
            Pre-populated ProviderStatsTracker
        """
        tracker = cls()
        for name, data in routing_data.items():
            if not isinstance(data, dict):
                continue
            snap = ProviderSnapshot(name=name)
            snap.total_requests = data.get("total_requests", 0)
            snap.total_errors = data.get("total_errors", 0)
            snap.total_tokens = data.get("total_tokens", 0)
            snap.total_cost_usd = data.get("total_cost_usd", 0.0)
            # Reconstruct approximate latencies from mean
            mean_lat = data.get("mean_latency_ms")
            if mean_lat and snap.total_requests > 0:
                snap.latencies = [mean_lat] * snap.total_requests
            tracker._stats[name] = snap
        return tracker
