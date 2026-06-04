"""
Configuration management for FastAPI application.
Handles environment variables, model paths, and app settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
import os
from typing import List

class Settings(BaseSettings):
    """Application settings from environment variables."""

    # App
    app_name: str = "PGA Analysis API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # CORS
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://localhost:8001",
    ]

    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]

    # Add production domain if available
    prod_domain: str = os.getenv("PROD_DOMAIN", "")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Model paths
    base_path: Path = Path(__file__).parent.parent.parent
    models_path: Path = base_path / "models"
    data_path: Path = base_path / "data"

    # Model files
    course_fit_model_path: Path = models_path / "course_fit_model.pkl"
    outcome_made_cut_model_path: Path = models_path / "outcome_made_cut.pkl"
    outcome_top_10_model_path: Path = models_path / "outcome_top_10.pkl"
    outcome_win_model_path: Path = models_path / "outcome_win.pkl"

    # WebSocket
    ws_max_connections: int = 100
    ws_timeout: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins, including production domain if set."""
        origins = self.cors_origins.copy()
        if self.environment == "production" and self.prod_domain:
            origins.append(self.prod_domain)
        return origins
    
# Global settings instance
settings = Settings()