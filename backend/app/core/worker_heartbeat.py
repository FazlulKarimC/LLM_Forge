"""
Worker Heartbeat Helpers

Provides async helpers for:
- touch_worker_heartbeat: upsert a heartbeat row
- has_recent_worker_heartbeat: check if any worker is alive
- cleanup_stale_worker_heartbeats: delete expired rows
- touch_worker_heartbeat_sync: sync wrapper for use in worker.py
"""

import asyncio
import logging
import socket
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.config import settings

logger = logging.getLogger(__name__)


async def touch_worker_heartbeat(
    session: AsyncSession,
    worker_id: str,
    backend: str = "rq",
    queue_name: str = "experiments",
    hostname: Optional[str] = None,
) -> None:
    """Upsert a heartbeat row for the given worker."""
    from app.models.worker_heartbeat import WorkerHeartbeatRecord  # noqa: F811

    now = datetime.now(timezone.utc)
    hostname = hostname or socket.gethostname()

    # Use PostgreSQL INSERT ... ON CONFLICT for atomic upsert
    stmt = pg_insert(WorkerHeartbeatRecord).values(
        worker_id=worker_id,
        backend=backend,
        queue_name=queue_name,
        hostname=hostname,
        created_at=now,
        updated_at=now,
    ).on_conflict_do_update(
        index_elements=["worker_id"],
        set_={"updated_at": now, "hostname": hostname},
    )
    await session.execute(stmt)
    await session.commit()


async def has_recent_worker_heartbeat(
    session: AsyncSession,
    max_age_seconds: Optional[int] = None,
    backend: str = "rq",
    queue_name: str = "experiments",
) -> bool:
    """Return True if at least one worker has checked in recently."""
    from app.models.worker_heartbeat import WorkerHeartbeatRecord

    if max_age_seconds is None:
        max_age_seconds = settings.RQ_WORKER_HEARTBEAT_TTL_SECONDS

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    result = await session.execute(
        select(WorkerHeartbeatRecord.worker_id)
        .where(
            WorkerHeartbeatRecord.backend == backend,
            WorkerHeartbeatRecord.queue_name == queue_name,
            WorkerHeartbeatRecord.updated_at >= cutoff,
        )
        .limit(1)
    )
    return result.first() is not None


async def cleanup_stale_worker_heartbeats(
    session: AsyncSession,
    max_age_seconds: Optional[int] = None,
) -> int:
    """Delete heartbeat rows older than max_age and return count deleted."""
    from app.models.worker_heartbeat import WorkerHeartbeatRecord

    if max_age_seconds is None:
        max_age_seconds = settings.RQ_WORKER_HEARTBEAT_TTL_SECONDS * 4  # generous

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
    result = await session.execute(
        delete(WorkerHeartbeatRecord).where(
            WorkerHeartbeatRecord.updated_at < cutoff,
        )
    )
    await session.commit()
    return result.rowcount  # type: ignore[return-value]


def touch_worker_heartbeat_sync(
    worker_id: str,
    backend: str = "rq",
    queue_name: str = "experiments",
    hostname: Optional[str] = None,
) -> None:
    """
    Sync wrapper for heartbeat upsert — used from the RQ worker script
    which runs its own event loop per heartbeat tick.
    """
    from app.core.database import async_session_maker

    async def _inner() -> None:
        async with async_session_maker() as session:
            await touch_worker_heartbeat(
                session, worker_id, backend, queue_name, hostname
            )

    asyncio.run(_inner())
