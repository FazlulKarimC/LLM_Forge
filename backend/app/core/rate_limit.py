"""
Rate Limiting for Free-Tier Protection

In-memory sliding window rate limiter.
No external dependencies (no Redis required).

Limits:
- Per-IP: configurable requests per hour
- Global: configurable max concurrent experiment runs
"""

import time
import logging
from collections import defaultdict
from threading import Lock
from typing import Optional

from fastapi import Request
from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────────
CREATE_LIMIT_PER_HOUR = 15
RUN_LIMIT_PER_HOUR = 15
MAX_CONCURRENT_RUNS = 3
WINDOW_SECONDS = 3600  # 1 hour


class SlidingWindowCounter:
    """Thread-safe sliding window rate limiter."""

    def __init__(self):
        self._lock = Lock()
        # IP -> list of timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)
        # Global concurrent run counter
        self._active_runs = 0

    def _cleanup(self, ip: str, now: float) -> None:
        """Remove expired timestamps outside the window."""
        cutoff = now - WINDOW_SECONDS
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > cutoff
        ]

    def check_rate_limit(self, ip: str, limit: int) -> Optional[int]:
        """
        Check if IP is within rate limit.

        Returns:
            None if allowed, or retry_after seconds if rate limited.
        """
        now = time.time()
        with self._lock:
            self._cleanup(ip, now)
            if len(self._requests[ip]) >= limit:
                # Calculate when the oldest request in the window expires
                oldest = min(self._requests[ip])
                retry_after = int(oldest + WINDOW_SECONDS - now) + 1
                return max(retry_after, 1)
            self._requests[ip].append(now)
            return None

    def check_concurrent_runs(self) -> bool:
        """Check if under the global concurrent run limit (read-only, for metrics)."""
        with self._lock:
            return self._active_runs < MAX_CONCURRENT_RUNS

    def try_acquire_run(self) -> bool:
        """Atomically check the concurrent run limit and increment if allowed.
        
        Returns:
            True if a run slot was acquired, False if at capacity.
        """
        with self._lock:
            if self._active_runs >= MAX_CONCURRENT_RUNS:
                return False
            self._active_runs += 1
            logger.info("Active concurrent runs: %d/%d", self._active_runs, MAX_CONCURRENT_RUNS)
            return True

    def increment_runs(self) -> None:
        """Increment active run counter."""
        with self._lock:
            self._active_runs += 1
            logger.info("Active concurrent runs: %d/%d", self._active_runs, MAX_CONCURRENT_RUNS)

    def decrement_runs(self) -> None:
        """Decrement active run counter."""
        with self._lock:
            self._active_runs = max(0, self._active_runs - 1)
            logger.info("Active concurrent runs: %d/%d", self._active_runs, MAX_CONCURRENT_RUNS)


# Global singleton
_limiter = SlidingWindowCounter()


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For from proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_response(message: str, retry_after: int) -> JSONResponse:
    """Create a 429 Too Many Requests response."""
    return JSONResponse(
        status_code=429,
        content={
            "error": True,
            "message": message,
            "retry_after": retry_after,
            "status_code": 429,
        },
        headers={"Retry-After": str(retry_after)},
    )


async def check_create_rate_limit(request: Request) -> None:
    """
    Rate limit check for experiment creation.
    Raises HTTPException(429) if rate limited.
    """
    ip = _get_client_ip(request)
    retry_after = _limiter.check_rate_limit(ip, CREATE_LIMIT_PER_HOUR)
    if retry_after is not None:
        minutes = (retry_after + 59) // 60
        logger.warning("Rate limit hit for IP %s on create (retry_after=%ds)", ip, retry_after)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. You can create a new experiment in {minutes} minute{'s' if minutes != 1 else ''}.",
            headers={"Retry-After": str(retry_after)},
        )


async def check_run_rate_limit(request: Request) -> None:
    """
    Rate limit check for experiment runs.
    Also checks global concurrent run limit.
    Raises HTTPException(429) if rate limited.
    """
    # Check global concurrent runs first
    # Atomically check and acquire a concurrent run slot
    if not _limiter.try_acquire_run():
        logger.warning("Global concurrent run limit reached (%d/%d)", MAX_CONCURRENT_RUNS, MAX_CONCURRENT_RUNS)
        raise HTTPException(
            status_code=429,
            detail=f"Server is busy — {MAX_CONCURRENT_RUNS} experiments are already running. Please wait for one to complete.",
            headers={"Retry-After": "30"},
        )

    ip = _get_client_ip(request)
    retry_after = _limiter.check_rate_limit(ip, RUN_LIMIT_PER_HOUR)
    if retry_after is not None:
        minutes = (retry_after + 59) // 60
        logger.warning("Rate limit hit for IP %s on run (retry_after=%ds)", ip, retry_after)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. You can run a new experiment in {minutes} minute{'s' if minutes != 1 else ''}.",
            headers={"Retry-After": str(retry_after)},
        )


def increment_active_runs() -> None:
    """Call when an experiment starts running."""
    _limiter.increment_runs()


def decrement_active_runs() -> None:
    """Call when an experiment finishes (success or failure)."""
    _limiter.decrement_runs()

