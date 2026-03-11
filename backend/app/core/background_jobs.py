"""Simple in-memory job store for long-running background tasks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, Optional
from uuid import uuid4

_JOB_TTL = timedelta(hours=6)
_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _cleanup_locked() -> None:
    cutoff = _now() - _JOB_TTL
    expired_job_ids = [
        job_id
        for job_id, job in _JOBS.items()
        if datetime.fromisoformat(job["updated_at"]) < cutoff
    ]
    for job_id in expired_job_ids:
        _JOBS.pop(job_id, None)


def create_job(kind: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = _now().isoformat()
    job = {
        "job_id": uuid4().hex,
        "kind": kind,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "metadata": metadata or {},
        "result": None,
        "error": None,
    }

    with _LOCK:
        _cleanup_locked()
        _JOBS[job["job_id"]] = job

    return dict(job)


def update_job(job_id: str, **updates: Any) -> Optional[Dict[str, Any]]:
    with _LOCK:
        _cleanup_locked()
        job = _JOBS.get(job_id)
        if job is None:
            return None

        job.update(updates)
        job["updated_at"] = _now().isoformat()
        return dict(job)


def mark_job_running(job_id: str) -> Optional[Dict[str, Any]]:
    return update_job(job_id, status="running", error=None)


def mark_job_completed(job_id: str, result: Any) -> Optional[Dict[str, Any]]:
    return update_job(job_id, status="completed", result=result, error=None)


def mark_job_failed(job_id: str, error: str) -> Optional[Dict[str, Any]]:
    return update_job(job_id, status="failed", error=error)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        _cleanup_locked()
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None