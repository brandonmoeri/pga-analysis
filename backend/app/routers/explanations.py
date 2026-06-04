"""
Explanations router - SHAP model interpretability
"""

from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime
import logging

from backend.app.models import ExplanationRequest, ExplanationResponse, LocalExplanation, FeatureImportance
from backend.app.services.explanations import ExplanationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/explanations", tags=["explanations"])

@router.post(
    "/local",
    response_model=ExplanationResponse,
    summary="Get SHAP explanation for prediction",
    description="Get local SHAP explanation showing feature contributions for a specific prediction"
)
async def get_local_explanation(request: ExplanationRequest):
    """Get SHAP explanation for a player-course prediction."""
    try:
        explanation = ExplanationService.get_local_explanation(
            player_id=request.player_id,
            course_id=request.course_id,
            top_n=request.top_n
        )

        if not explanation:
            raise HTTPException(
                status_code=500,
                detail="Could not generate explanation"
            )
        
        top_features = [
            FeatureImportance(
                feature=f["feature"],
                importance=f["shap_value"],
                contribution="positive" if f["shap_value"] > 0 else "negative"
            )
            for f in explanation["top_features"]
        ]

        return ExplanationResponse(
            player_id=request.player_id,
            course_id=request.course_id,
            local_explanation=LocalExplanation(
                player_id=request.player_id,
                course_id=request.course_id,
                prediction=explanation["prediction"],
                top_features=top_features,
                base_value=explanation["base_value"]
            ),
            global_feature_importance=top_features,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error in get_local_explanations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/feature-importance",
    response_model=List[FeatureImportance],
    summary="Get global feature importance",
    description="Get top contributing features across all predictions"
)
async def get_feature_importance(top_n: int = 15):
    """Get global feature importance."""
    try:
        features = ExplanationService.get_global_feature_importance(top_n=top_n)

        if features is None:
            return []
        
        return [
            FeatureImportance(
                feature=f["feature"],
                importance=f["importance"],
                contribution="positive"
            )
            for f in features
        ]
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))