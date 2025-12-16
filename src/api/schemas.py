from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ModerationAction(str, Enum):
    BAN = "BAN"
    REVIEW = "REVIEW"
    ALLOW = "ALLOW"


class PredictRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="Comment text to classify")


class BatchPredictRequest(BaseModel):
    comments: List[str] = Field(
        ..., min_items=1, max_items=100, description="List of comments to classify"
    )


class PredictionResult(BaseModel):
    comment: str = Field(..., description="Original comment text")
    predictions: Dict[str, float] = Field(
        ..., description="Probability scores for each toxic category"
    )
    is_toxic: bool = Field(..., description="Whether comment is toxic (any category > 0.5)")
    moderation_action: ModerationAction = Field(
        ..., description="Recommended moderation action"
    )
    model_version: str = Field(..., description="Version of the model used")


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    version: str
    model_type: str
    target_labels: List[str]
    loaded: bool
    source: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

