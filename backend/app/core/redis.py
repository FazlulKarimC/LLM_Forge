"""
Redis Connection Module

Provides Redis connection and RQ queue for background task processing.
Uses Upstash Redis (cloud-hosted, serverless).

Usage:
    from app.core.redis import get_queue
    
    queue = get_queue()
    queue.enqueue(my_task, arg1, arg2)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from redis import Redis
from rq import Queue

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level state for connection caching
_connection: Optional[Redis] = None
_queues: Dict[str, Queue] = {}


@dataclass
class RedisProbeResult:
    """Structured result from a Redis PING probe."""
    healthy: bool
    latency_ms: float = 0.0
    error: Optional[str] = None


def get_redis_connection() -> Redis:
    """
    Get cached Redis connection from Upstash URL.
    
    Uses module-level caching to ensure only one connection is created
    and reused across all calls.  Includes short timeouts suitable for
    free-tier Upstash instances.
    
    Returns:
        Redis connection instance (cached)
    
    Raises:
        ValueError: If REDIS_URL is not configured
    """
    global _connection

    if _connection is not None:
        return _connection

    if not settings.REDIS_URL:
        raise ValueError(
            "REDIS_URL not configured. "
            "Set REDIS_URL environment variable with Upstash connection string."
        )

    timeout_s = settings.UPSTASH_HEALTHCHECK_TIMEOUT_MS / 1000.0

    _connection = Redis.from_url(
        settings.REDIS_URL,
        decode_responses=False,  # RQ requires bytes
        socket_connect_timeout=timeout_s,
        socket_timeout=timeout_s,
    )
    return _connection


def reset_redis_connection() -> None:
    """
    Invalidate cached Redis connection and all cached queues.

    Called by the circuit breaker when Upstash is detected as unavailable
    so the next attempt creates a fresh connection instead of reusing a
    dead one.
    """
    global _connection
    _connection = None
    _queues.clear()
    logger.info("Redis connection cache cleared")


def get_queue(name: str = "experiments") -> Queue:
    """
    Get cached RQ queue for background tasks.
    
    Returns existing Queue if already created for this name,
    otherwise creates and caches a new Queue.
    
    Args:
        name: Queue name (default: "experiments")
    
    Returns:
        RQ Queue instance (cached)
    """
    if name not in _queues:
        _queues[name] = Queue(name, connection=get_redis_connection())
    return _queues[name]


def probe_redis() -> RedisProbeResult:
    """
    Probe Upstash Redis with a PING command using short timeouts.

    Returns a structured result instead of throwing so callers can make
    decisions without try/except boilerplate.
    """
    try:
        conn = get_redis_connection()
        start = time.monotonic()
        conn.ping()
        latency = (time.monotonic() - start) * 1000
        return RedisProbeResult(healthy=True, latency_ms=round(latency, 1))
    except Exception as exc:
        return RedisProbeResult(healthy=False, error=str(exc)[:200])
