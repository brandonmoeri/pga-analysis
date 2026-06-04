"""
Data service for loading and caching player/course/tournament data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from backend.app.config import settings

logger = logging.getLogger(__name__)

class DataService:
    """Handle data loading and caching."""

    _cache: Dict = {}

    @classmethod
    def load_player_stats(cls) -> Optional[pd.DataFrame]:
        """Load processed player statistics."""
        try:
            path = settings.data_path / "processed" / "player_stats.csv"
            if path.exists():
                df = pd.read_csv(path)
                logger.info(f"Loaded player stats: {len(df)} rows")
                return df
            logger.warning(f"Player stats not found at {path}")
            return None
        except Exception as e:
            logger.error(f"Error loading player stats: {e}")
            return None
        
    @classmethod
    def load_course_features(cls) -> Optional[pd.DataFrame]:
        """Load processed course features."""
        try:
            path = settings.data_path / "processed" / "course_features.csv"
            if path.exists():
                df = pd.read_csv(path)
                logger.info(f"Loaded course features: {len(df)} courses")
                return df
            logger.warning(f"Course features not found at {path}")
            return None
        except Exception as e:
            logger.error(f"Error loading course features: {e}")
            return None
        
    @classmethod
    def load_tournament_results(cls) -> Optional[pd.DataFrame]:
        """Load tournament results."""
        try:
            path = settings.data_path / "processed" / "tournament_results.csv"
            if path.exists():
                df = pd.read_csv(path)
                logger.info(f"Loaded tournament results: {len(df)} records")
                return df
            logger.warning(f"Tournament results not found at {path}")
            return None
        except Exception as e:
            logger.error(f"Error loading tournament results: {e}")
            return None
        
    @classmethod
    def get_all_data(cls) -> Dict[str, Optional[pd.DataFrame]]:
        """Load all required data."""
        return {
            "player_stats": cls.load_player_stats(),
            "course_features": cls.load_course_features(),
            "tournament_results": cls.load_tournament_results(),
        }
    
    @classmethod
    def get_player_stats_dict(cls) -> Dict:
        """Get player stats as dictionary for API response."""
        df = cls.load_player_stats()
        if df is not None:
            return {
                "count": len(df),
                "columns": list(df.columns),
                "sample": df.head(5).to_dict(orient='records') if len(df) > 0 else [],
            }
        return {"count": 0, "columns": [], "sample": []}
    
    @classmethod
    def get_course_features_dict(cls) -> Dict:
        """Get course features as dictionary for API response."""
        df = cls.load_course_features()
        if df is not None:
            return {
                "count": len(df),
                "columns": list(df.columns),
                "courses": df['course_id'].unique().tolist() if 'course_id' in df.columns else [],
            }
        return {"count": 0, "columns": [], "courses": []}