from datetime import datetime
from typing import Dict, Optional

from sqlalchemy import JSON, Index, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ReviewedComment(Base):
    __tablename__ = "reviewed_comments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    comment_text: Mapped[str] = mapped_column(Text, nullable=False)
    original_predictions: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    reviewed_labels: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    moderator_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default=text("'api'")
    )
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )

    __table_args__ = (
        Index("idx_reviewed_comments_status", "status"),
        Index("idx_reviewed_comments_reviewed_at", "reviewed_at"),
        Index("idx_reviewed_comments_source", "source"),
        Index("idx_reviewed_comments_created_at", "created_at"),
    )

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "comment_text": self.comment_text,
            "original_predictions": self.original_predictions,
            "reviewed_labels": self.reviewed_labels,
            "moderator_id": self.moderator_id,
            "reviewed_at": self.reviewed_at,
            "created_at": self.created_at,
            "source": self.source,
            "model_version": self.model_version,
            "status": self.status,
        }
