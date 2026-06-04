"""
Tournament ranking service.
"""

import logging
from typing import Optional, Dict, List
from backend.app.services.course_fit import CourseFitService

logger = logging.getLogger(__name__)


class RankingService:
    """Handle tournament rankings."""
    
    @classmethod
    def rank_players_for_course(
        cls,
        course_id: str,
        player_ids: Optional[List[str]] = None,
        top_n: int = 50
    ) -> Optional[Dict]:
        """
        Rank players for a specific course.
        
        Args:
            course_id: Course identifier
            player_ids: List of player IDs to rank (if None, use all)
            top_n: Number of top players to return
        
        Returns:
            Dictionary with ranking results
        """
        try:
            if player_ids is None:
                player_ids = [f"player_{i}" for i in range(1, 151)]  # Mock data
            
            rankings = []
            for rank, player_id in enumerate(player_ids[:top_n], 1):
                fit_result = CourseFitService.predict_player_course_fit(player_id, course_id)
                if fit_result:
                    rankings.append({
                        "rank": rank,
                        "player_id": player_id,
                        "fit_score": fit_result["fit_score"],
                        "confidence": fit_result["confidence"]
                    })
            
            return {
                "course_id": course_id,
                "total_ranked": len(rankings),
                "rankings": rankings,
                "method": "course_fit_model"
            }
        except Exception as e:
            logger.error(f"Error ranking players: {e}")
            return None
    
    @classmethod
    def aggregate_tournament_ranking(
        cls,
        tournament_name: str,
        courses: List[str],
        top_n: int = 50
    ) -> Optional[Dict]:
        """
        Aggregate rankings across multiple tournament courses.
        
        Args:
            tournament_name: Tournament identifier
            courses: List of course IDs
            top_n: Number of top players in aggregate
        
        Returns:
            Dictionary with aggregate ranking
        """
        try:
            course_rankings = {}
            player_scores = {}
            
            # Get rankings for each course
            for course_id in courses:
                ranking = cls.rank_players_for_course(course_id, top_n=top_n)
                if ranking:
                    course_rankings[course_id] = ranking
                    
                    # Aggregate scores
                    for item in ranking["rankings"]:
                        player_id = item["player_id"]
                        if player_id not in player_scores:
                            player_scores[player_id] = []
                        player_scores[player_id].append(item["fit_score"])
            
            # Calculate average scores
            aggregate = []
            for rank, (player_id, scores) in enumerate(
                sorted(
                    player_scores.items(),
                    key=lambda x: sum(x[1]) / len(x[1]),
                    reverse=True
                )[:top_n],
                1
            ):
                avg_score = sum(scores) / len(scores)
                aggregate.append({
                    "rank": rank,
                    "player_id": player_id,
                    "average_fit_score": float(avg_score),
                    "courses_analyzed": len(scores)
                })
            
            return {
                "tournament": tournament_name,
                "courses_analyzed": len(course_rankings),
                "aggregate_ranking": aggregate,
                "aggregation_method": "mean_fit_score"
            }
        except Exception as e:
            logger.error(f"Error aggregating tournament ranking: {e}")
            return None