"""Short-lived execution reservations; never credentials or case content."""

from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base
from app.models.prompt import now


class BenchmarkReservation(Base):
    __tablename__ = "benchmark_reservations"
    __table_args__ = (
        Index("ix_benchmark_reservation_active", "released_at", "expires_at"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind = Column(String(32), nullable=False)
    resource_id = Column(String(64), nullable=False, index=True)
    attempt = Column(Integer, nullable=True)
    deadline_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    released_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=now)
