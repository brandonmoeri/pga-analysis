"""
Model explanation service (SHAP interpretability).
"""

import logging
from typing import Optional, Dict, List
from backend.app.utils.model_loader import ModelCache

logger = logging.getLogger(__name__)


class ExplanationService:
    """Handle model explanations."""
    
    @classmethod
    def get_local_explanation(
        cls,
        player_id: str,
        course_id: str,
        top_n: int = 10
    ) -> Optional[Dict]:
        """
        Get SHAP explanation for player-course prediction.
        
        Args:
            player_id: Player identifier
            course_id: Course identifier
            top_n: Number of top features to explain
        
        Returns:
            Dictionary with feature contributions
        """
        try:
            model = ModelCache.get_model("course_fit")
            if model is None:
                return None
            
            # Mock SHAP values for top features
            return {
                "player_id": player_id,
                "course_id": course_id,
                "base_value": 0.0,
                "prediction": 0.45,
                "top_features": [
                    {"feature": "sg_total_last_10", "shap_value": 0.35, "feature_value": 0.45},
                    {"feature": "course_avg_sg", "shap_value": 0.18, "feature_value": 0.12},
                    {"feature": "sg_total_last_5", "shap_value": 0.15, "feature_value": 0.52},
                    {"feature": "sg_total_momentum", "shap_value": 0.12, "feature_value": 0.08},
                    {"feature": "strokes_gained_ott", "shap_value": 0.10, "feature_value": 0.15},
                    {"feature": "strokes_gained_app", "shap_value": 0.09, "feature_value": 0.22},
                    {"feature": "strokes_gained_putt", "shap_value": 0.08, "feature_value": 0.13},
                    {"feature": "strokes_gained_arg", "shap_value": -0.05, "feature_value": -0.05},
                ][:top_n]
            }
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return None
    
    @classmethod
    def get_global_feature_importance(cls, top_n: int = 15) -> Optional[List[Dict]]:
        """Get global feature importance across all predictions."""
        try:
            model = ModelCache.get_model("course_fit")
            if model is None or not hasattr(model, 'feature_importances_'):
                return None
            
            # Mock global importance
            return [
                {"rank": 1, "feature": "sg_total_last_10", "importance": 0.28},
                {"rank": 2, "feature": "course_avg_sg", "importance": 0.18},
                {"rank": 3, "feature": "sg_total_last_5", "importance": 0.15},
                {"rank": 4, "feature": "sg_total_momentum", "importance": 0.12},
                {"rank": 5, "feature": "strokes_gained_ott", "importance": 0.09},
                {"rank": 6, "feature": "strokes_gained_app", "importance": 0.08},
                {"rank": 7, "feature": "strokes_gained_putt", "importance": 0.05},
                {"rank": 8, "feature": "course_length_yards", "importance": 0.03},
                {"rank": 9, "feature": "rounds_at_course", "importance": 0.02},
            ][:top_n]
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None