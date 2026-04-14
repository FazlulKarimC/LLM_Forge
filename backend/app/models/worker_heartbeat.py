"""
Worker Heartbeat SQLAlchemy Model

Tracks liveness of RQ workers so the API can decide whether to dispatch
to the queue or fall back to inline execution.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class WorkerHeartbeatRecord(Base):
    """Tracks RQ worker liveness via periodic heartbeats."""

    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    backend: Mapped[str] = mapped_column(String(32), nullable=False, default="rq")
    queue_name: Mapped[str] = mapped_column(String(64), nullable=False, default="experiments")
    hostname: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_worker_heartbeats_backend_queue", "backend", "queue_name"),
        Index("ix_worker_heartbeats_updated_at", "updated_at"),
    )
