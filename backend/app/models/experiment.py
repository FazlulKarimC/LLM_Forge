"""
Experiment SQLAlchemy Model

Stores experiment configurations and status.
Each experiment contains the full config needed for reproduction.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, String, DateTime, JSON, Enum, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base
from app.schemas.experiment import ExperimentStatus


class Experiment(Base):
    """
    Experiment database model.
    
    Stores experiment configuration and execution state.
    Supports soft delete via deleted_at column.
    """
    __tablename__ = "experiments"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    
    # Basic info
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Configuration (stored as JSON for flexibility)
    config = Column(JSON, nullable=False)
    
    # Denormalized fields for filtering
    method = Column(String(50), nullable=False, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    dataset_name = Column(String(255), nullable=False, index=True)
    
    # Status tracking
    status = Column(
        Enum(ExperimentStatus),
        default=ExperimentStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Reproducibility metadata (P1 #11)
    dataset_hash = Column(String(64), nullable=True)  # SHA256 of dataset content
    sample_ids = Column(JSON, nullable=True)           # List of sampled example IDs
    current_attempt = Column(Integer, default=1, server_default="1", nullable=False)
    
    # Soft delete
    deleted_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Relationships
    results = relationship("Result", back_populates="experiment", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name}, status={self.status})>"
    
    @property
    def is_deleted(self) -> bool:
        """Check if experiment is soft deleted."""
        return self.deleted_at is not None
