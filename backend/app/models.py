"""
Pydantic request/response models for API validation and documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# WenSocket Models

class RankingUpdate(BaseModel):
    """Single ranking update message."""
    rank: int
    player_id: str
    fit_score: float
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)

class RankingStreamMessage(BaseModel):
    """Message sent over WebSocket during streaming."""
    type: str = Field(..., description="Message type: 'update', 'progress', 'complete', 'error'")
    data: Optional[Dict[str, Any]] = None
    batch_number: int = 0
    total_batches: Optional[int] = None
    course_id: Optional[str] = None
    tournament_name: Optional[str] = None
    progress_percent: Optional[int] = None
    message: Optional[str] = None

class PredictionUpdate(BaseModel):
    """Single prediction update message."""
    player_id: str
    course_id: str
    fit_score: Optional[float]
    make_cut_prob: float
    top_10_prob: float
    win_prob: float
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)

# Prediction Models

class PredictionRequest(BaseModel):
    """Request for player outcome prediction."""
    player_id: str = Field(..., description="Player Identifier")
    course_id: str = Field(..., description="Course Identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "scheffler_scottie",
                "course_id": "augusta_national"
            }
        }

class OutcomeProbabilities(BaseModel):
    """Outcome probability predictions."""
    make_cut: float = Field(..., ge=0, le=1, description="Probability of making the cut")
    top_10: float = Field(..., ge=0, le=1, description="Probability of finishing top 10")
    win: float = Field(..., ge=0, le=1, description="Probability of winning")

class PredictionResponse(BaseModel):
    """Response with prediction results and explanations."""
    player_id: str
    course_id: str
    fit_score: Optional[float] = Field(None, description="Course fit score (strokes)")
    outcome_probabilities: OutcomeProbabilities
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")
    top_features: List[Dict[str, Any]] = Field(default_factory=list, description="Top contributing features")
    timestamp: datetime = Field(default_factory=datetime.now)

# Ranking Models

class RankingRequest(BaseModel):
    """Request for tournament rankings."""
    tournament_name: str = Field(..., description="Tournament identifier")
    courses: List[str] = Field(..., description="Course IDs in tournament")
    aggregation_method: str = Field(default="average", description="How to aggregate course rankings")
    top_n: int = Field(default=50, ge=1, le=200, description="Number of top players to return")
    
    class Config:
        json_schema_extra = {
            "example": {
                "tournament_name": "masters_2024",
                "courses": ["augusta_national"],
                "aggregation_method": "average",
                "top_n": 100
            }
        }

class PlayerRanking(BaseModel):
    """Single player ranking result."""
    rank: int
    player_id: str
    player_name: Optional[str] = None
    fit_score: float = Field(..., description="Course fit score")
    ranking_score: float = Field(..., description="Overall ranking score")
    win_probability: float = Field(..., description="Estimated win probability")


class RankingResponse(BaseModel):
    """Response with tournament rankings."""
    tournament_name: str
    course_id: str
    rankings: List[PlayerRanking]
    aggregate_ranking: Optional[List[PlayerRanking]] = None
    generated_at: datetime = Field(default_factory=datetime.now)

# Explanation Models

class FeatureImportance(BaseModel):
    """Feature importance entry."""
    feature: str
    importance: float = Field(..., ge=0, description="Importance score")
    contribution: Optional[str] = Field(None, description="Direction: positive/negative")


class LocalExplanation(BaseModel):
    """Local SHAP explanation for single prediction."""
    player_id: str
    course_id: str
    prediction: float
    top_features: List[FeatureImportance]
    base_value: Optional[float] = None


class ExplanationRequest(BaseModel):
    """Request for model explanation."""
    player_id: str = Field(..., description="Player identifier")
    course_id: str = Field(..., description="Course identifier")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of top features")


class ExplanationResponse(BaseModel):
    """Response with SHAP explanations."""
    player_id: str
    course_id: str
    local_explanation: LocalExplanation
    global_feature_importance: List[FeatureImportance]
    timestamp: datetime = Field(default_factory=datetime.now)

# Health & Status Models

class ModelStatus(BaseModel):
    """Status of a single model."""
    loaded: bool
    model_type: Optional[str] = None
    last_loaded: Optional[datetime] = None

class HealthResponse(BaseModel):
    """API health status."""
    status: str = Field(..., description="healthy/degraded/unhealthy")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    models: Dict[str, ModelStatus]
    uptime_seconds: Optional[int] = None

# Error Models

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = "Validation Error"
    details: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.now)