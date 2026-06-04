"""
Model loading and caching utilities.
Manages lifecycle of ML models in memory.
"""

import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Thread-safe cache for loaded ML models.
    Stores models in memory for fast inference.
    """

    _instance = None
    _models: Dict[str, Any] = {}
    _loaded_at: Optional[datetime] = None
    _load_error: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load_models(cls, model_paths: Dict[str, Path]) -> Dict[str, bool]:
        """
        Load all models from disk.

        Args:
            model_paths: Dict mapping model names to file paths

        Returns:
            Dict indicating which models loaded successfully
        """
        instance = cls()
        logger.info("Starting model loading...")

        load_status = {}

        try:
            for model_name, model_path in model_paths.items():
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        instance._models[model_name] = pickle.load(f)
                    logger.info(f"{model_name} loaded from {model_path}")
                    load_status[model_name] = True
                else:
                    logger.warning(f"Model not found: {model_path}")
                    load_status[model_name] = False
            
            instance._loaded_at = datetime.now()
            logger.info("Model loading completed.")
        
        except Exception as e:
            error_msg = f"Error loading models: {str(e)}"
            logger.error(error_msg, exc_info=True)
            instance._load_error = error_msg
        
        return load_status
    
    @classmethod
    def get_model(cls, model_name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        instance = cls()
        return instance._models.get(model_name)
    
    @classmethod
    def get_all_models(cls) -> Dict[str, Any]:
        """Get all loaded models."""
        instance = cls()
        return instance._models.copy()
    
    @classmethod
    def is_model_loaded(cls, model_name: str) -> bool:
        """Check if a model is loaded."""
        instance = cls()
        return model_name in instance._models and instance._models[model_name] is not None
    
    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Get status of all models."""
        instance = cls()
        return {
            "models_loaded": len(instance._models),
            "model_names": list(instance._models.keys()),
            "loaded_at": instance._loaded_at.isoformat() if instance._loaded_at else None,
            "load_error": instance._load_error,
            "details": {
                name: {
                    "loaded": model is not None,
                    "type": type(model).__name__ if model is not None else None,
                }
                for name, model in instance._models.items()
            }
        }
    
    @classmethod
    def clear(cls):
        """Clear all models from cache."""
        instance = cls()
        instance._models.clear()
        instance._loaded_at = None
        instance._load_error = None
        logger.info("Model cache cleared")


def get_model_cache() -> ModelCache:
    """Dependency injection for ModelCache singleton."""
    return ModelCache()