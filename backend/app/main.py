"""
FastAPI application entry point for PGA Analysis backend.
Manages model loading, CORS, and routing.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from backend.app.config import settings
from backend.app.utils.model_loader import ModelCache

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI application lifecycle.
    Loads models on startup, cleans up on shutdown.
    """
    # Startup
    logger.info("FastAPI application starting...")
    
    model_paths = {
        "course_fit": settings.course_fit_model_path,
        "outcome_made_cut": settings.outcome_made_cut_model_path,
        "outcome_top_10": settings.outcome_top_10_model_path,
        "outcome_win": settings.outcome_win_model_path,
    }
    
    ModelCache.load_models(model_paths)
    
    yield
    
    # Shutdown
    logger.info("FastAPI application shutting down...")
    ModelCache.clear()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Machine Learning API for PGA Tour predictions",
    version=settings.app_version,
    lifespan=lifespan,
)

# Configure CORS
cors_origins = settings.get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

logger.info(f"CORS enabled for origins: {cors_origins}")


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API documentation link."""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
async def health_check():
    """Check API and model health."""
    model_cache = ModelCache()
    status = model_cache.get_status()
    
    all_models_loaded = all(
        model_cache.is_model_loaded(name)
        for name in ["course_fit", "outcome_made_cut", "outcome_top_10", "outcome_win"]
    )
    
    return {
        "status": "healthy" if all_models_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "models": status["details"],
        "models_loaded_at": status["loaded_at"],
        "load_error": status["load_error"],
    }


@app.get("/api/health/models")
async def model_status():
    """Detailed model status and metadata."""
    model_cache = ModelCache()
    return model_cache.get_status()


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


# ============================================================================
# Route Registration (Phase 4)
# ============================================================================

# Placeholder for routers to be imported and registered
# from app.routers import predictions, rankings, explanations, data
# app.include_router(predictions.router, prefix="/api")
# app.include_router(rankings.router, prefix="/api")
# app.include_router(explanations.router, prefix="/api")
# app.include_router(data.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )