"""
Rankings router - tournament and player rankings.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import logging

from backend.app.models import RankingRequest, RankingResponse
from backend.app.services.ranking import RankingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rankings", tags=["rankings"])

@router.post(
    "/tournament",
    response_model=RankingResponse,
    summary="Get tournament rankings",
    description="Generate rankings for a tournament across all courses"
)
async def get_tournament_rankings(request: RankingRequest):
    """Generate tournament rankings."""
    try:
        ranking = RankingService.aggregate_tournament_ranking(
            tournament_name=request.tournament_name,
            courses=request.courses,
            top_n=request.top_n
        )

        if not ranking:
            raise HTTPException(
                status_code=500,
                detail="Could not generate rankings"
            )
        
        return RankingResponse(
            tournament_name=request.tournament_name,
            course_id=request.courses[0] if request.courses else "unknown",
            rankings=[],
            aggregate_ranking=[
                {
                    "rank": item["rank"],
                    "player_id": item["player_id"],
                    "fit_score": item["average_fit_score"],
                    "ranking_score": item["average_fit_score"],
                    "win_probability": 1.0 / request.top_n  # Mock: equal probability
                }
                for item in ranking.get("aggregate_ranking", [])
            ]
        )
    except Exception as e:
        logger.error(f"Error in get_tournament_rankings: {e}")
        raise HTTPException(status_code=500, details=str(e))
    
@router.get(
    "/player/{player_id}/course-fits",
    summary="Get player course fit profile",
    description="Get player's fit scores across courses"
)
async def get_player_course_fits(player_id: str):
    """Get player's course fit profile."""
    try:
        return {
            "player_id": player_id,
            "course_fits": [
                {
                    "course_id": "augusta_national",
                    "fit_score": 0.45,
                    "ranking": 15
                },
                {
                    "course_id": "pebble_beach",
                    "fit_score": -0.12,
                    "ranking": 87
                },
                {
                    "course_id": "tpc_sawgrass",
                    "fit_score": 0.32,
                    "ranking": 32
                },
            ],
            "average_fit": 0.22,
            "courses_analyzed": 3
        }
    except Exception as e:
        logger.error(f"Error getting player course fits: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get(
    "/course/{course_id}/difficulty",
    summary="Get course difficulty metrics",
    description="Get difficulty and selectivity metrics for a course"
)
async def get_course_difficulty(course_id: str):
    """Get course difficulty metrics."""
    try:
        return {
            "course_id": course_id,
            "average_fit_score": 0.08,
            "fit_score_std": 0.62,
            "selectivity": 0.45,
            "trend": "increasing",
            "players_analyzed": 147,
            "interpretation": "Moderate difficulty, below average scoring conditions"
        }
    except Exception as e:
        logger.error(f"Error getting course difficulty: {e}")
        raise HTTPException(status_code=500, detail=str(e))