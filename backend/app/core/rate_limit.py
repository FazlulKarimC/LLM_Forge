"""
Rate limiting for free-tier protection.

This module keeps the lightweight per-IP sliding-window limits in memory.
The global concurrent run cap is enforced from database-backed experiment
status in the API layer, because process-local counters do not work once
RQ workers and the API server run separately.
"""

import logging
import time
from ipaddress import ip_address
from collections import defaultdict
from threading import Lock
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

CREATE_LIMIT_PER_HOUR = 15
RUN_LIMIT_PER_HOUR = 15
MAX_CONCURRENT_RUNS = 3
WINDOW_SECONDS = 3600  # 1 hour


class SlidingWindowCounter:
    """Thread-safe per-IP sliding window limiter."""

    def __init__(self):
        self._lock = Lock()
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _cleanup(self, ip: str, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        self._requests[ip] = [ts for ts in self._requests[ip] if ts > cutoff]

    def check_rate_limit(self, ip: str, limit: int) -> Optional[int]:
        """
        Check if IP is within limit.

        Returns None if allowed, otherwise retry-after seconds.
        """
        now = time.time()
        with self._lock:
            self._cleanup(ip, now)
            if len(self._requests[ip]) >= limit:
                oldest = min(self._requests[ip])
                retry_after = int(oldest + WINDOW_SECONDS - now) + 1
                return max(retry_after, 1)
            self._requests[ip].append(now)
            return None


_limiter = SlidingWindowCounter()


def _parse_ip(value: str):
    try:
        return ip_address(value.strip())
    except ValueError:
        return None


def _get_client_ip(request: Request) -> str:
    """
    Extract the client IP with conservative proxy-header handling.

    We only trust X-Forwarded-For when the immediate peer looks like an
    internal proxy. In that case, prefer the right-most globally routable
    address because trusted proxies append to the chain.
    """
    peer_host = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if not forwarded:
        return peer_host

    peer_ip = _parse_ip(peer_host)
    if peer_ip is None or peer_ip.is_global:
        return peer_host

    forwarded_ips = [
        parsed
        for raw_ip in forwarded.split(",")
        if (parsed := _parse_ip(raw_ip)) is not None
    ]
    if not forwarded_ips:
        return peer_host

    for candidate in reversed(forwarded_ips):
        if candidate.is_global:
            return str(candidate)

    return str(forwarded_ips[-1])


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
    """Rate limit check for experiment creation."""
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

    This only enforces the per-IP sliding-window limit.
    """
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
