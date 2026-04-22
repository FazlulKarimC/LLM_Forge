"""Persistent background-job metadata store backed by the primary database."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import delete, select

from app.core.database import async_session_maker
from app.models.background_job import BackgroundJobRecord

_JOB_TTL = timedelta(hours=6)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _cleanup_expired_jobs(session) -> None:
    cutoff = _now() - _JOB_TTL
    await session.execute(
        delete(BackgroundJobRecord).where(BackgroundJobRecord.updated_at < cutoff)
    )


async def create_job(kind: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    async with async_session_maker() as session:
        await _cleanup_expired_jobs(session)
        job = BackgroundJobRecord(
            job_id=uuid4().hex,
            kind=kind,
            status="queued",
            job_metadata=metadata or {},
            result=None,
            error=None,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job.to_payload()


async def update_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    async with async_session_maker() as session:
        await _cleanup_expired_jobs(session)
        result = await session.execute(
            select(BackgroundJobRecord).where(BackgroundJobRecord.job_id == job_id)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None

        for key, value in updates.items():
            if key == "metadata":
                job.job_metadata = value
            else:
                setattr(job, key, value)
        job.updated_at = _now()
        await session.commit()
        await session.refresh(job)
        return job.to_payload()


async def mark_job_running(job_id: str) -> Optional[Dict[str, Any]]:
    return await update_job(job_id, status="running", error=None)


async def mark_job_completed(job_id: str, result: Any) -> Optional[Dict[str, Any]]:
    return await update_job(job_id, status="completed", result=result, error=None)


async def mark_job_failed(job_id: str, error: str) -> Optional[Dict[str, Any]]:
    return await update_job(job_id, status="failed", error=error)


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    async with async_session_maker() as session:
        await _cleanup_expired_jobs(session)
        await session.commit()
        result = await session.execute(
            select(BackgroundJobRecord).where(BackgroundJobRecord.job_id == job_id)
        )
        job = result.scalar_one_or_none()
        return job.to_payload() if job is not None else None
