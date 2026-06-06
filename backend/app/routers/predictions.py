"""
Predictions router - player outcome predictions.
"""

from multiprocessing import reduction

from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime
import logging

from backend.app.models import PredictionRequest, PredictionResponse, OutcomeProbabilities, CourseHistory
from backend.app.services.outcomes import OutcomeService
from backend.app.services.course_fit import CourseFitService
from backend.app.services.data import DataService
from backend.app.services.features import FeatureService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["predictions"])

@router.post(
    "/player-outcome",
    response_model=PredictionResponse,
    summary="Predict player tournament outcomes",
    description="Predict make cut, top-10, and win probabilities for a plyer at a course"
)
async def predict_player_outcome(request: PredictionRequest):
    """Predict tournament outcomes for a plyer at a specific course."""
    try:
        if not DataService.player_exists(request.player_id):
            raise HTTPException(status_code=404, detail=f"Player '{request.player_id}' not found")
        if not DataService.course_exists(request.course_id):
            raise HTTPException(status_code=404, detail=f"Course '{request.course_id}' not found")

        # Get course fit score
        fit_result = CourseFitService.predict_player_course_fit(
            request.player_id,
            request.course_id
        )

        # Get outcome probabilities
        outcome_result = OutcomeService.predict_player_outcomes(
            request.player_id,
            request.course_id
        )

        if not outcome_result:
            raise HTTPException(
                status_code=500,
                detail="Could not generate prediction"
            )

        # Get course history features
        features = FeatureService.create_player_course_features(request.player_id, request.course_id)
        course_history = None
        if features:
            course_history = CourseHistory(
                sg_avg=features["course_avg_sg"],
                appearances=features["rounds_at_course"],
            )

        return PredictionResponse(
            player_id=request.player_id,
            course_id=request.course_id,
            fit_score=fit_result["fit_score"] if fit_result else None,
            outcome_probabilities=OutcomeProbabilities(
                make_cut=outcome_result["make_cut_probability"],
                top_10=outcome_result["top_10_probability"],
                win=outcome_result["win_probability"]
            ),
            confidence=outcome_result["overall_confidence"],
            top_features=[
                {
                    "feature": "sg_total_last_10",
                    "importance": 0.28,
                    "value": 0.45
                },
                {
                    "feature": "course_avg_sg",
                    "importance": 0.18,
                    "value": 0.12
                },
            ],
            course_history=course_history,
            timestamp=datetime.now()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_player_outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/historical/{player_id}",
    summary="Get player prediction history",
    description="Retrieve historical predictions for a player"
)
async def get_prediction_history(player_id: str, limit: int = 10):
    """Get historical predictions for a player."""
    try:
        return {
            "player_id": player_id,
            "prediction_count": 0,
            "predictions": [],
            "note": "Prediction history not yet implemented - requires database"
        }
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))