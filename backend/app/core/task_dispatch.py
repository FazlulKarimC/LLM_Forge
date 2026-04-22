"""
Task Dispatch Abstraction

Centralizes the decision of how to execute an experiment:
  - InlineDispatchBackend: always uses FastAPI BackgroundTasks
  - UpstashRQDispatchBackend: always enqueues via RQ (fails hard)
  - AutoDispatchBackend: tries RQ if Upstash + worker healthy, else inline

The run_experiment route calls dispatch_experiment() and receives a
DispatchResult telling it what happened. This keeps the route thin.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable
from uuid import UUID

from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Data ────────────────────────────────────────────────────────────────


@dataclass
class DispatchResult:
    """Outcome of a dispatch attempt."""
    backend_used: str          # "rq" | "inline" | "inline_fallback"
    fallback_reason: Optional[str] = None
    circuit_state: str = "closed"
    worker_available: Optional[bool] = None


# ── Protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class DispatchBackend(Protocol):
    async def dispatch(
        self,
        background_tasks: BackgroundTasks,
        experiment_id: UUID,
        db: Optional[AsyncSession] = None,
        custom_base_url: Optional[str] = None,
        custom_api_key: Optional[str] = None,
    ) -> DispatchResult: ...


def _schedule_inline_execution(
    background_tasks: BackgroundTasks,
    experiment_id: UUID,
    custom_base_url: Optional[str] = None,
    custom_api_key: Optional[str] = None,
) -> None:
    """Schedule inline execution via FastAPI BackgroundTasks."""
    from app.api.experiments import _execute_inline

    background_tasks.add_task(
        _execute_inline, experiment_id, custom_base_url, custom_api_key
    )


# ── Inline Backend ──────────────────────────────────────────────────────


class InlineDispatchBackend:
    """Always run inline via FastAPI BackgroundTasks."""

    async def dispatch(
        self,
        background_tasks: BackgroundTasks,
        experiment_id: UUID,
        db: Optional[AsyncSession] = None,
        custom_base_url: Optional[str] = None,
        custom_api_key: Optional[str] = None,
    ) -> DispatchResult:
        _schedule_inline_execution(
            background_tasks, experiment_id, custom_base_url, custom_api_key
        )
        return DispatchResult(backend_used="inline")


# ── Upstash/RQ Backend ─────────────────────────────────────────────────


class UpstashRQDispatchBackend:
    """Always enqueue via RQ — fails hard if Redis is unavailable."""

    async def dispatch(
        self,
        background_tasks: BackgroundTasks,
        experiment_id: UUID,
        db: Optional[AsyncSession] = None,
        custom_base_url: Optional[str] = None,
        custom_api_key: Optional[str] = None,
    ) -> DispatchResult:
        from app.core.redis import get_queue
        from app.tasks.experiment_tasks import run_experiment_task

        queue = get_queue()
        queue.enqueue(
            run_experiment_task,
            str(experiment_id),
            custom_base_url=custom_base_url,
            custom_api_key=custom_api_key,
        )
        return DispatchResult(backend_used="rq")


# ── Auto Backend (the smart one) ───────────────────────────────────────


class AutoDispatchBackend:
    """
    Decision tree (evaluated top-to-bottom):

    1. No REDIS_URL configured          → inline
    2. Circuit is open                   → inline  (skip probe)
    3. Probe window: health-check fails  → open circuit, inline
    4. No recent worker heartbeat        → inline  (don't open circuit)
    5. Enqueue to RQ                     → rq
       5a. Enqueue throws avail. error   → open circuit, inline fallback
    """

    async def dispatch(
        self,
        background_tasks: BackgroundTasks,
        experiment_id: UUID,
        db: Optional[AsyncSession] = None,
        custom_base_url: Optional[str] = None,
        custom_api_key: Optional[str] = None,
    ) -> DispatchResult:
        from app.core import upstash_circuit
        from app.core.redis import probe_redis

        # 1. No Redis URL
        if not settings.REDIS_URL:
            logger.info("Auto dispatch → inline (no REDIS_URL)")
            _schedule_inline_execution(
                background_tasks, experiment_id, custom_base_url, custom_api_key
            )
            return DispatchResult(
                backend_used="inline",
                fallback_reason="REDIS_URL not configured",
                circuit_state=upstash_circuit.get_circuit_snapshot()["state"],
            )

        # 2. Circuit open?
        if upstash_circuit.is_open():
            reason = upstash_circuit.get_circuit_snapshot()["last_failure_reason"]
            logger.info("Auto dispatch → inline (circuit open: %s)", reason)
            _schedule_inline_execution(
                background_tasks, experiment_id, custom_base_url, custom_api_key
            )
            return DispatchResult(
                backend_used="inline",
                fallback_reason=f"Circuit open: {reason}",
                circuit_state="open",
            )

        # 3. Probe if allowed
        if upstash_circuit.allow_probe():
            probe = probe_redis()
            if not probe.healthy:
                upstash_circuit.record_failure(
                    ConnectionError(probe.error or "Redis probe failed")
                )
                logger.warning(
                    "Auto dispatch → inline (probe failed: %s)", probe.error
                )
                _schedule_inline_execution(
                    background_tasks, experiment_id, custom_base_url, custom_api_key
                )
                return DispatchResult(
                    backend_used="inline",
                    fallback_reason=f"Redis probe failed: {probe.error}",
                    circuit_state="open",
                )
            upstash_circuit.record_success()

        # 4. Worker heartbeat check (async — uses the route's DB session)
        worker_alive = await _check_worker_heartbeat_async(db)
        if not worker_alive:
            logger.info("Auto dispatch → inline (no recent worker heartbeat)")
            _schedule_inline_execution(
                background_tasks, experiment_id, custom_base_url, custom_api_key
            )
            return DispatchResult(
                backend_used="inline",
                fallback_reason="No recent worker heartbeat",
                circuit_state=upstash_circuit.get_circuit_snapshot()["state"],
                worker_available=False,
            )

        # 5. Enqueue via RQ
        try:
            result = await UpstashRQDispatchBackend().dispatch(
                background_tasks, experiment_id, db, custom_base_url, custom_api_key
            )
            result.circuit_state = upstash_circuit.get_circuit_snapshot()["state"]
            result.worker_available = True
            return result
        except Exception as exc:
            # 5a. Enqueue failed — classify and maybe open circuit
            upstash_circuit.record_failure(exc)
            logger.warning(
                "Auto dispatch → inline fallback (enqueue failed: %s)", exc
            )
            _schedule_inline_execution(
                background_tasks, experiment_id, custom_base_url, custom_api_key
            )
            return DispatchResult(
                backend_used="inline_fallback",
                fallback_reason=f"Enqueue failed: {str(exc)[:200]}",
                circuit_state=upstash_circuit.get_circuit_snapshot()["state"],
                worker_available=True,
            )


async def _check_worker_heartbeat_async(db: Optional[AsyncSession] = None) -> bool:
    """
    Check the worker_heartbeats table using the provided async session.

    If no session is provided, creates a fresh one. If the DB is
    unreachable, assumes no worker (fail-safe: go inline).
    """
    from app.core.worker_heartbeat import has_recent_worker_heartbeat

    try:
        if db is not None:
            return await has_recent_worker_heartbeat(db)
        else:
            # Fallback: create own session (used from readiness checks)
            from app.core.database import async_session_maker
            async with async_session_maker() as session:
                return await has_recent_worker_heartbeat(session)
    except Exception as exc:
        logger.warning("Worker heartbeat check failed: %s", exc)
        return False


# ── Factory ─────────────────────────────────────────────────────────────


def _get_backend() -> DispatchBackend:
    """Return the dispatch backend based on QUEUE_BACKEND_MODE."""
    mode = settings.QUEUE_BACKEND_MODE.lower()
    if mode == "inline":
        return InlineDispatchBackend()
    if mode == "upstash":
        return UpstashRQDispatchBackend()
    return AutoDispatchBackend()


async def dispatch_experiment(
    background_tasks: BackgroundTasks,
    experiment_id: UUID,
    db: Optional[AsyncSession] = None,
    custom_base_url: Optional[str] = None,
    custom_api_key: Optional[str] = None,
) -> DispatchResult:
    """
    Top-level dispatch entry point used by the run_experiment route.
    Delegates to the configured backend.
    """
    backend = _get_backend()
    result = await backend.dispatch(
        background_tasks, experiment_id, db, custom_base_url, custom_api_key
    )
    logger.info(
        "Experiment %s dispatched via %s (reason=%s, circuit=%s, worker=%s)",
        experiment_id,
        result.backend_used,
        result.fallback_reason,
        result.circuit_state,
        result.worker_available,
    )
    return result


# ── Readiness snapshot ──────────────────────────────────────────────────


async def get_dispatch_readiness_snapshot() -> dict:
    """
    Return a structured readiness snapshot for the /ready endpoint.
    Checks all three conditions: Redis config, circuit state, worker liveness.
    """
    from app.core import upstash_circuit

    snapshot = {
        "task_dispatch": "healthy",
        "upstash": "not_configured",
        "rq_worker": "not_configured",
    }

    mode = settings.QUEUE_BACKEND_MODE.lower()

    if mode == "inline":
        snapshot["task_dispatch"] = "inline_only"
        return snapshot

    if not settings.REDIS_URL:
        snapshot["task_dispatch"] = "fallback_inline"
        snapshot["upstash"] = "not_configured"
        snapshot["rq_worker"] = "not_configured"
        return snapshot

    # Upstash status
    circuit = upstash_circuit.get_circuit_snapshot()
    if circuit["state"] == "open":
        snapshot["upstash"] = "circuit_open"
        snapshot["task_dispatch"] = "fallback_inline"
    elif circuit["state"] == "half_open":
        snapshot["upstash"] = "half_open"
        snapshot["task_dispatch"] = "fallback_inline"
    else:
        snapshot["upstash"] = "healthy"

    # Worker status (async — uses its own session)
    worker_alive = await _check_worker_heartbeat_async()
    if worker_alive:
        snapshot["rq_worker"] = "healthy"
    else:
        snapshot["rq_worker"] = "worker_missing"
        snapshot["task_dispatch"] = "fallback_inline"

    # If both healthy and circuit closed, dispatch is healthy
    if snapshot["upstash"] == "healthy" and snapshot["rq_worker"] == "healthy":
        snapshot["task_dispatch"] = "healthy"

    return snapshot
