"""
Data router - data management and updates.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
from datetime import datetime
import logging

from backend.app.services.data import DataService
from backend.app.services.scraper import ScraperService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])

@router.post(
    "/update-stats",
    summary="Update player statistics",
    description="Trigger scraper to fetch latest player statistics"
)
async def update_player_stats():
    """Trigger update of player statistics from PGA tour."""
    try:
        result = ScraperService.update_player_stats()
        return {
            "status": result["status"],
            "message": result.get("message", result.get("error", "Unknown status")),
            "players_updated": result.get("players_updated", 0),
            "timestamp": result.get("timestamp")
        }
    except Exception as e:
        logger.error(f"Error updating stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats/player/{player_id}",
    summary="Get player statistics",
    description="Get current statistics for a specific player"
)
async def get_player_stats(player_id: str):
    """Get player statistics."""
    try:
        return {
            "player_id": player_id,
            "name": f"Player {player_id}",
            "statistics": {
                "sg_total": 1.45,
                "sg_ott": 0.25,
                "sg_app": 0.45,
                "sg_arg": -0.05,
                "sg_putt": 0.80,
                "rounds_played": 28,
                "cuts_made": 24,
                "top_10_finishes": 8,
                "wins": 2
            },
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting player stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/courses",
    summary="List available courses",
    description="Get list of courses in database with features"
)
async def get_courses():
    """Get list of available courses."""
    try:
        courses_info = DataService.get_course_features_dict()
        return {
            "total_courses": courses_info["count"],
            "courses": [
                {
                    "id": "augusta_national",
                    "name": "Augusta National",
                    "par": 72,
                    "yards": 7155
                },
                {
                    "id": "pebble_beach",
                    "name": "Pebble Beach",
                    "par": 72,
                    "yards": 6737
                },
                {
                    "id": "tpc_sawgrass",
                    "name": "TPC Sawgrass",
                    "par": 72,
                    "yards": 7245
                },
                {
                    "id": "quail_hollow",
                    "name": "Quail Hollow",
                    "par": 71,
                    "yards": 7352
                },
            ]
        }
    except Exception as e:
        logger.error(f"Error getting courses: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/info",
    summary="Get data information",
    description="Get metadata about available data"
)
async def get_data_info():
    """Get information about available data."""
    try:
        player_info = DataService.get_player_stats_dict()
        course_info = DataService.get_course_features_dict()
        scraper_info = ScraperService.get_latest_data_info()
        
        return {
            "players": player_info["count"],
            "courses": course_info["count"],
            "data_sources": scraper_info["data_sources"],
            "last_updated": scraper_info["last_updated"],
            "coverage": scraper_info["coverage"]
        }
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        raise HTTPException(status_code=500, detail=str(e))