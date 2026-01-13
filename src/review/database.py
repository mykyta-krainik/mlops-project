import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select

from src.data.database import DatabaseManager
from src.data.models import ReviewedComment

logger = logging.getLogger(__name__)


class ReviewDatabase:
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 5432,
        database: str = "reviews",
        user: Optional[str] = None,
        password: Optional[str] = None,
        secret_arn: Optional[str] = None,
    ):
        self._db_manager = DatabaseManager(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            secret_arn=secret_arn or os.environ.get("DB_SECRET_ARN"),
            use_lambda_pool=False,
        )

    def initialize_schema(self) -> None:
        self._db_manager.create_tables()
        logger.info("Database schema initialized")

    def add_pending_review(
        self,
        comment_text: str,
        predictions: Dict[str, float],
        model_version: str,
        source: str = "api",
    ) -> int:
        with self._db_manager.get_session() as session:
            comment = ReviewedComment(
                comment_text=comment_text,
                original_predictions=predictions,
                model_version=model_version,
                source=source,
                status="pending",
            )
            session.add(comment)
            session.flush()
            review_id = comment.id
            logger.info(f"Added pending review with id {review_id}")
            return review_id

    def get_pending_reviews(
        self, limit: int = 10, offset: int = 0
    ) -> List[Dict[str, Any]]:
        with self._db_manager.get_session() as session:
            stmt = (
                select(ReviewedComment)
                .where(ReviewedComment.status == "pending")
                .order_by(ReviewedComment.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            results = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": comment.id,
                    "comment_text": comment.comment_text,
                    "original_predictions": comment.original_predictions,
                    "model_version": comment.model_version,
                    "source": comment.source,
                    "created_at": comment.created_at,
                }
                for comment in results
            ]

    def get_pending_count(self) -> int:
        with self._db_manager.get_session() as session:
            stmt = select(func.count()).select_from(ReviewedComment).where(
                ReviewedComment.status == "pending"
            )
            count = session.execute(stmt).scalar()
            return count or 0

    def submit_review(
        self,
        review_id: int,
        reviewed_labels: Dict[str, int],
        moderator_id: str,
    ) -> bool:
        with self._db_manager.get_session() as session:
            stmt = select(ReviewedComment).where(
                ReviewedComment.id == review_id,
                ReviewedComment.status == "pending"
            )
            comment = session.execute(stmt).scalar_one_or_none()
            
            if comment:
                comment.reviewed_labels = reviewed_labels
                comment.moderator_id = moderator_id
                comment.reviewed_at = datetime.now()
                comment.status = "reviewed"
                logger.info(f"Submitted review for id {review_id}")
                return True
            return False

    def skip_review(self, review_id: int, moderator_id: str) -> bool:
        with self._db_manager.get_session() as session:
            stmt = select(ReviewedComment).where(
                ReviewedComment.id == review_id,
                ReviewedComment.status == "pending"
            )
            comment = session.execute(stmt).scalar_one_or_none()
            
            if comment:
                comment.moderator_id = moderator_id
                comment.reviewed_at = datetime.now()
                comment.status = "skipped"
                return True
            return False

    def get_reviewed_comments(
        self, since: Optional[datetime] = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        with self._db_manager.get_session() as session:
            stmt = (
                select(ReviewedComment)
                .where(ReviewedComment.status == "reviewed")
            )
            
            if since:
                stmt = stmt.where(ReviewedComment.reviewed_at > since)
            
            stmt = stmt.order_by(ReviewedComment.reviewed_at.desc()).limit(limit)
            
            results = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": comment.id,
                    "comment_text": comment.comment_text,
                    "reviewed_labels": comment.reviewed_labels,
                    "model_version": comment.model_version,
                    "reviewed_at": comment.reviewed_at,
                }
                for comment in results
            ]

    def get_statistics(self) -> Dict[str, Any]:
        with self._db_manager.get_session() as session:
            stmt = select(
                ReviewedComment.status,
                func.count().label("count")
            ).group_by(ReviewedComment.status)
            
            status_results = session.execute(stmt).all()
            status_counts = {row.status: row.count for row in status_results}
            
            seven_days_ago = datetime.now() - timedelta(days=7)
            stmt = (
                select(
                    func.date(ReviewedComment.reviewed_at).label("date"),
                    func.count().label("count")
                )
                .where(
                    ReviewedComment.status == "reviewed",
                    ReviewedComment.reviewed_at > seven_days_ago
                )
                .group_by(func.date(ReviewedComment.reviewed_at))
                .order_by(func.date(ReviewedComment.reviewed_at))
            )
            
            daily_results = session.execute(stmt).all()
            daily_reviews = [
                {"date": str(row.date), "count": row.count}
                for row in daily_results
            ]
            
            stmt = (
                select(
                    ReviewedComment.moderator_id,
                    func.count().label("count")
                )
                .where(
                    ReviewedComment.status == "reviewed",
                    ReviewedComment.moderator_id.is_not(None)
                )
                .group_by(ReviewedComment.moderator_id)
                .order_by(func.count().desc())
                .limit(10)
            )
            
            moderator_results = session.execute(stmt).all()
            top_moderators = [
                {"moderator": row.moderator_id, "count": row.count}
                for row in moderator_results
            ]
            
            return {
                "status_counts": status_counts,
                "daily_reviews": daily_reviews,
                "top_moderators": top_moderators,
                "total_pending": status_counts.get("pending", 0),
                "total_reviewed": status_counts.get("reviewed", 0),
            }
