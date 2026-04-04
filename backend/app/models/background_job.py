"""
Background Job SQLAlchemy Model

Stores asynchronous job state for long-running API-triggered tasks.
This keeps pollable jobs durable across restarts on free-tier hosting.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class BackgroundJobRecord(Base):
    """Durable background job state persisted in the primary database."""

    __tablename__ = "background_jobs"

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    job_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    def to_payload(self) -> dict:
        """Return the API contract expected by the frontend."""
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.job_metadata or {},
            "result": self.result,
            "error": self.error,
        }
