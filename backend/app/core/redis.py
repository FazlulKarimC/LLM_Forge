"""
Redis Connection Module

Provides Redis connection and RQ queue for background task processing.
Uses Upstash Redis (cloud-hosted, serverless).

Usage:
    from app.core.redis import get_queue
    
    queue = get_queue()
    queue.enqueue(my_task, arg1, arg2)
"""

from functools import lru_cache
from typing import Dict

from redis import Redis
from rq import Queue

from app.core.config import settings

# Module-level cache for queue instances
_queues: Dict[str, Queue] = {}


@lru_cache(maxsize=1)
def get_redis_connection() -> Redis:
    """
    Get cached Redis connection from Upstash URL.
    
    Uses lru_cache to ensure only one connection is created
    and reused across all calls.
    
    Returns:
        Redis connection instance (cached)
    
    Raises:
        ValueError: If REDIS_URL is not configured
    """
    if not settings.REDIS_URL:
        raise ValueError(
            "REDIS_URL not configured. "
            "Set REDIS_URL environment variable with Upstash connection string."
        )
    
    return Redis.from_url(
        settings.REDIS_URL,
        decode_responses=False,  # RQ requires bytes
    )


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
