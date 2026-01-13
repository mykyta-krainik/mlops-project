"""
Pydantic schemas for the review system API.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ReviewLabels(BaseModel):
    """Labels for a reviewed comment."""

    toxic: int = Field(0, ge=0, le=1, description="Is the comment toxic?")
    severe_toxic: int = Field(0, ge=0, le=1, description="Is it severely toxic?")
    obscene: int = Field(0, ge=0, le=1, description="Is it obscene?")
    threat: int = Field(0, ge=0, le=1, description="Does it contain threats?")
    insult: int = Field(0, ge=0, le=1, description="Is it insulting?")
    identity_hate: int = Field(0, ge=0, le=1, description="Contains identity hate?")


class PendingReview(BaseModel):
    """A comment pending review."""

    id: int
    comment_text: str
    original_predictions: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    source: str = "api"
    created_at: Optional[datetime] = None


class SubmitReviewRequest(BaseModel):
    """Request to submit a review."""

    review_id: int = Field(..., description="ID of the review to submit")
    labels: ReviewLabels = Field(..., description="Reviewed labels")


class SubmitReviewResponse(BaseModel):
    """Response after submitting a review."""

    success: bool
    message: str
    review_id: int


class ReviewStatistics(BaseModel):
    """Review system statistics."""

    total_pending: int
    total_reviewed: int
    status_counts: Dict[str, int]
    daily_reviews: List[Dict]
    top_moderators: List[Dict]


class PendingReviewsResponse(BaseModel):
    """Response containing pending reviews."""

    reviews: List[PendingReview]
    total_pending: int
    page: int
    limit: int
