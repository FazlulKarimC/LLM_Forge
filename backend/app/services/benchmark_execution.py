"""Bounded hosted execution, independent of any queue or permanent worker."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from threading import Event, Lock
import time

from fastapi import HTTPException
from sqlalchemy import delete, select, update

from app.core.config import settings
from app.models.benchmark_reservation import BenchmarkReservation
from app.models.evaluation_execution import EvaluationCapacity


class ExecutionInterrupted(Exception):
    """An attempt no longer owns execution permission."""


class BenchmarkBudgetExceeded(Exception):
    """The attempt reached its explicit call or time budget."""


async def run_blocking(function, *args):
    """Keep ownership until an issued synchronous call exits on cancellation."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        owner = getattr(function, "__self__", None)
        cancel_pending = getattr(owner, "cancel_pending", None)
        if cancel_pending:
            cancel_pending()
        # Cancelling to_thread does not stop its thread or its HTTP request.
        # Drain the bounded transport call before releasing the reservation.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue  # Repeated cancellation still cannot stop the thread.
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # Consume an error received while draining.
        raise cancellation


def utcnow():
    return datetime.now(timezone.utc)


def aware(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


async def reserve_execution(db, *, project_id, kind, resource_id, attempt=None):
    """Claim the single global slot atomically in the caller's transaction.

    The existing capacity row is a mutex, held only during admission. Core-table
    queries deliberately count reservations across projects; no case data leaves
    this internal function. PostgreSQL serializes contenders on the mutex.
    """
    capacity = await db.scalar(
        select(EvaluationCapacity).where(EvaluationCapacity.id == 1).with_for_update()
    )
    if capacity is None:
        raise RuntimeError("Execution capacity is missing; apply database migrations")
    now = utcnow()
    table = BenchmarkReservation.__table__
    active = await db.scalar(
        select(table.c.id)
        .where(table.c.released_at.is_(None), table.c.expires_at > now)
        .limit(1)
    )
    if active is not None:
        raise HTTPException(
            429,
            "Server is busy with another benchmark or model job. Try again after it finishes.",
            headers={"Retry-After": "30"},
        )
    from app.models.evaluation_execution import EvaluationExecution

    executions = EvaluationExecution.__table__
    if await db.scalar(
        select(executions.c.id)
        .where(executions.c.status == "running", executions.c.lease_until > now)
        .limit(1)
    ):
        raise HTTPException(
            429,
            "Server is busy with a legacy evaluation. Try again after it finishes.",
            headers={"Retry-After": "30"},
        )
    await db.execute(delete(table).where(table.c.expires_at < now - timedelta(days=1)))
    deadline = now + timedelta(seconds=settings.BENCHMARK_MAX_SECONDS)
    reservation = BenchmarkReservation(
        project_id=project_id,
        kind=kind,
        resource_id=str(resource_id),
        attempt=attempt,
        deadline_at=deadline,
        # Allow an issued call to finish before an abandoned slot can be reused.
        expires_at=deadline
        + timedelta(seconds=settings.BENCHMARK_CALL_TIMEOUT_SECONDS + 30),
    )
    db.add(reservation)
    await db.flush()
    return reservation


async def release_execution(db, *, kind, resource_id, attempt=None):
    query = update(BenchmarkReservation).where(
        BenchmarkReservation.kind == kind,
        BenchmarkReservation.resource_id == str(resource_id),
        BenchmarkReservation.released_at.is_(None),
    )
    if attempt is not None:
        query = query.where(BenchmarkReservation.attempt == attempt)
    await db.execute(query.values(released_at=utcnow()))


async def execution_deadline(db, *, kind, resource_id, attempt=None):
    query = select(BenchmarkReservation).where(
        BenchmarkReservation.kind == kind,
        BenchmarkReservation.resource_id == str(resource_id),
        BenchmarkReservation.released_at.is_(None),
    )
    if attempt is not None:
        query = query.where(BenchmarkReservation.attempt == attempt)
    row = await db.scalar(query)
    if row is None or aware(row.deadline_at) <= utcnow():
        raise BenchmarkBudgetExceeded(
            "Execution reservation expired; start a new attempt explicitly"
        )
    return aware(row.deadline_at)


class BudgetedEngine:
    """Serialize generation and check ownership before every provider call.

    Calls happen on a worker thread; ownership checks run on the API event loop.
    A batch is deliberately sequential here, and never replays a partial batch.
    The underlying HTTP clients also have timeouts and no automatic retries.
    """

    def __init__(self, engine, *, deadline_at, ensure_active=None, max_calls=None):
        self.engine = engine
        self.loop = asyncio.get_running_loop()
        self.ensure_active = ensure_active
        self.max_calls = (
            max_calls if max_calls is not None else settings.BENCHMARK_MAX_CALLS
        )
        self.calls = 0
        self.deadline = time.monotonic() + max(
            0, (aware(deadline_at) - utcnow()).total_seconds()
        )
        self.lock = Lock()
        self.stop_requested = Event()
        self.call_log = []

    def cancel_pending(self):
        self.stop_requested.set()

    def check(self):
        if self.stop_requested.is_set():
            raise ExecutionInterrupted("Execution was cancelled")
        if time.monotonic() >= self.deadline:
            raise BenchmarkBudgetExceeded(
                "Benchmark time budget reached; completed cases were kept"
            )
        if self.ensure_active:
            asyncio.run_coroutine_threadsafe(self.ensure_active(), self.loop).result(
                timeout=10
            )

    def generate(self, prompt, config):
        with self.lock:
            self.check()
            if len(prompt.encode()) > 8 * 1024:
                raise BenchmarkBudgetExceeded("Compiled prompt exceeds the 8 KiB hosted input budget")
            if self.calls >= self.max_calls:
                raise BenchmarkBudgetExceeded(
                    "Benchmark call budget reached; completed cases were kept"
                )
            self.calls += 1  # An ambiguous response still consumes the call budget.
            result = self.engine.generate(prompt, config)
            self.call_log.append({
                "served_provider": result.served_provider,
                "served_model": result.served_model,
                "backend_provider": result.backend_provider,
                "requested_model": getattr(self.engine, "model_name", None),
                "usage_source": result.usage_source,
                "cost_source": result.cost_source,
                "tokens_input": result.tokens_input, "tokens_output": result.tokens_output,
                "cost_usd": result.cost_usd, "latency_ms": result.latency_ms,
                "latency_kind": result.latency_kind,
            })
            return result

    def generate_batch(self, prompts, config, max_workers=1):
        return [self.generate(prompt, config) for prompt in prompts]

    def __getattr__(self, name):
        return getattr(self.engine, name)
