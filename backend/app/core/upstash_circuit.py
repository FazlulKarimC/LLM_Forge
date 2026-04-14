"""
Upstash Circuit Breaker

In-process circuit breaker that protects against repeated calls to a dead
or archived Upstash Redis instance.  Three states:

    closed   – Upstash used normally.
    open     – Upstash skipped; all dispatches go inline.
    half_open – One probe allowed; success closes, failure re-opens.

The circuit opens for UPSTASH_CIRCUIT_OPEN_MINUTES (default 30 min) on
hard availability failures: connection errors, timeouts, auth failures,
and Upstash "archived / not found / gone" responses.

Application-layer bugs (bad args, serialization errors) do NOT trip it.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Patterns that indicate the Upstash instance was archived, deleted, or gone
_ARCHIVED_PATTERNS = re.compile(
    r"archived|not\s*found|gone|noauth|wrongpass|invalid\s*password",
    re.IGNORECASE,
)


@dataclass
class UpstashCircuitState:
    """Mutable in-process circuit breaker state."""
    state: str = "closed"             # closed | open | half_open
    opened_at: float = 0.0            # monotonic timestamp when circuit opened
    reopen_at: float = 0.0            # monotonic timestamp when half-open probe allowed
    last_failure_reason: Optional[str] = None
    last_success_at: float = 0.0

    # Cached probe outcome so we don't ping Upstash on every request
    _probe_cache_valid_until: float = 0.0
    _probe_cache_healthy: Optional[bool] = None


# Global singleton — process-local, no persistence needed.
_circuit = UpstashCircuitState()


def _open_duration_seconds() -> float:
    return settings.UPSTASH_CIRCUIT_OPEN_MINUTES * 60


def _probe_cache_ttl() -> float:
    return float(settings.UPSTASH_HEALTHCHECK_CACHE_SECONDS)


# ── Public API ──────────────────────────────────────────────────────────


def is_open() -> bool:
    """Return True if the circuit is open (Upstash should be skipped)."""
    if _circuit.state == "closed":
        return False
    if _circuit.state == "open":
        if time.monotonic() >= _circuit.reopen_at:
            _circuit.state = "half_open"
            logger.info("Upstash circuit → half_open (probe window)")
            return False          # allow one probe
        return True
    # half_open — allow the probe
    return False


def allow_probe() -> bool:
    """Return True if a health probe should be attempted right now."""
    if _circuit.state == "closed":
        # In closed state, use cached probe to avoid spamming PING
        if time.monotonic() < _circuit._probe_cache_valid_until:
            return False  # cached result still fresh
        return True
    if _circuit.state == "half_open":
        return True
    return False  # open — skip


def record_success() -> None:
    """Record a successful Upstash interaction."""
    if _circuit.state == "half_open":
        logger.info("Upstash circuit → closed (probe succeeded)")
    _circuit.state = "closed"
    _circuit.last_success_at = time.monotonic()
    _circuit.last_failure_reason = None
    _circuit._probe_cache_valid_until = time.monotonic() + _probe_cache_ttl()
    _circuit._probe_cache_healthy = True


def record_failure(exc: BaseException) -> None:
    """
    Classify the failure and, if it's an Upstash availability problem,
    open the circuit for UPSTASH_CIRCUIT_OPEN_MINUTES.
    """
    reason = classify_upstash_failure(exc)
    if reason is None:
        # Application-level bug — don't open the circuit
        logger.debug("Non-circuit failure (app bug): %s", exc)
        return

    from app.core.redis import reset_redis_connection

    now = time.monotonic()
    _circuit.state = "open"
    _circuit.opened_at = now
    _circuit.reopen_at = now + _open_duration_seconds()
    _circuit.last_failure_reason = reason
    _circuit._probe_cache_valid_until = now + _probe_cache_ttl()
    _circuit._probe_cache_healthy = False

    # Kill the stale cached connection so next probe creates a fresh one
    reset_redis_connection()

    logger.warning(
        "Upstash circuit OPENED for %d min — reason: %s",
        settings.UPSTASH_CIRCUIT_OPEN_MINUTES,
        reason,
    )


def classify_upstash_failure(exc: BaseException) -> Optional[str]:
    """
    Decide whether *exc* is an Upstash availability failure.

    Returns a short human-readable reason string if the circuit should open,
    or None if this is an application-level bug that should not trip it.
    """
    from redis.exceptions import (
        ConnectionError as RedisConnectionError,
        TimeoutError as RedisTimeoutError,
        AuthenticationError as RedisAuthError,
        ResponseError as RedisResponseError,
    )

    if isinstance(exc, RedisConnectionError):
        return f"ConnectionError: {str(exc)[:120]}"
    if isinstance(exc, RedisTimeoutError):
        return f"TimeoutError: {str(exc)[:120]}"
    if isinstance(exc, RedisAuthError):
        return f"AuthenticationError: {str(exc)[:120]}"
    if isinstance(exc, RedisResponseError):
        msg = str(exc)
        if _ARCHIVED_PATTERNS.search(msg):
            return f"ResponseError (archived/gone): {msg[:120]}"
        return None   # other ResponseError = app bug
    if isinstance(exc, OSError):
        # DNS failures, TLS errors, etc.
        return f"OSError: {str(exc)[:120]}"

    return None  # unknown / app-level error — don't trip circuit


def get_circuit_snapshot() -> dict:
    """Return a JSON-safe snapshot of the circuit state for readiness endpoints."""
    return {
        "state": _circuit.state,
        "last_failure_reason": _circuit.last_failure_reason,
        "open_minutes": settings.UPSTASH_CIRCUIT_OPEN_MINUTES,
    }


def reset_circuit() -> None:
    """Force-close the circuit. Useful for tests."""
    _circuit.state = "closed"
    _circuit.opened_at = 0.0
    _circuit.reopen_at = 0.0
    _circuit.last_failure_reason = None
    _circuit._probe_cache_valid_until = 0.0
    _circuit._probe_cache_healthy = None
