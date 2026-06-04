"""
Course fit prediction service using the trained model.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from backend.app.utils.model_loader import ModelCache

logger = logging.getLogger(__name__)

class CourseFitService:
    """Handle course fit predictions."""

    @classmethod
    def predict_player_course_fit(cls, player_id: int, course_id: int) -> Optional[float]:
        """
        Predict player-course fit score.

        Args:
            player_id: Player identifier
            course_id: Course identifier
        
        Returns:
            Dictionary with prediction and confidence
        """
        try:
            model = ModelCache.get_model("course_fit")
            if model is None:
                logger.warning("Course fit model not loaded")
                return None
            
            # In production, construct feature vector from player_id and course_id
            # For now, return mock prediction
            fit_score = np.random.normal(0, 0.85) # Mean 0, std dev ~0.85 strokes

            return {
                "player_id": player_id,
                "course_id": course_id,
                "fit_score": float(fit_score),
                "confidence": 0.72, # Average R^2 from training
                "interpretation": "Higher is better" if fit_score > 0 else "Lower than average"
            }
        except Exception as e:
            logger.error(f"Error predicting course fit: {e}")
            return None
        
    @classmethod
    def get_feature_importance(cls, top_n: int = 15) -> Optional[List[Dict]]:
        """Get top contributing features."""
        try:
            model = ModelCache.get_model("course_fit")
            if model is None or not hasattr(model, 'feature_importances_'):
                logger.warning("Cannot extract feature importance")
                return None
            
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]

            features = [
                {"rank": i+1, "feature": f"feature_{idx}", "importance": float(importances[idx])}
                for i, idx in enumerate(top_indices)
            ]
            return features
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None