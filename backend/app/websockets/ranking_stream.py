"""
WebSocket connection manager and streaming logic for rankings and predictions.
"""

import logging
import asyncio
from typing import List, Set, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import json

from backend.app.models import RankingStreamMessage, PredictionUpdate, RankingUpdate
from backend.app.services.ranking import RankingService
from backend.app.services.outcomes import OutcomeService
from backend.app.services.course_fit import CourseFitService

logger = logging.getLogger(__name__)

BATCH_SIZE = 10 # Send updated in batches of 10 players

class ConnectionManager:
    """Manage WebSocket connections and broadcasting."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.active_streams: Dict[str, bool] = {}

    async def connect(self, websocket: WebSocket, stream_id: str):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        if stream_id not in self.active_connections:
            self.active_connections[stream_id] = []
        self.active_connections[stream_id].append(websocket)
        logger.info(f"Client connected to stream: {stream_id}")

    def disconnect(self, websocket: WebSocket, stream_id: str):
        """Remove a disconnected client."""
        if stream_id in self.active_connections:
            self.active_connections[stream_id].remove(websocket)
            if not self.active_connections[stream_id]:
                del self.active_connections[stream_id]
                if stream_id in self.active_streams:
                    del self.active_streams[stream_id]
        logger.info(f"Client disconnected from stream: {stream_id}")

    async def broadcast(self, stream_id: str, message: RankingStreamMessage):
        """Broadcast message to all clients in a stream"""
        if stream_id not in self.active_connections:
            return
        
        disconnected = []
        message_data = message.model_dump(mode="json")

        for websocket in self.active_connections[stream_id]:
            try:
                await websocket.send_json(message_data)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws, stream_id)

    def is_streaming(self, stream_id: str) -> bool:
        """Check if stream is currently active."""
        return self.active_streams.get(stream_id, False)
    
    def set_streaming(self, stream_id: str, value: bool):
        """Set streaming status."""
        self.active_streams[stream_id] = value

class RankingStreamService:
    """Service for streaming ranking computations."""

    @staticmethod
    async def stream_tournament_rankings(
        manager: ConnectionManager,
        stream_id: str,
        tournament_name: str,
        courses: List[str],
        top_n: int = 100,
    ):
        """
        Stream tournament rankings to all connected clients.

        Args:
            manager: ConnectionManager instance
            stream_id: Unique stream identifier
            tournament_name: Tournament name
            courses: List of course IDs
            top_n: Number of top players
        """
        manager.set_streaming(stream_id, True)

        try:
            # Get all player IDs (in production, query database)
            player_ids = [f"player_{i}" for i in range(1, 201)]
            all_rankings = []

            total_batches = (len(player_ids) + BATCH_SIZE - 1) // BATCH_SIZE

            # Stream progress message
            await manager.broadcast(
                stream_id,
                RankingStreamMessage(
                    type="progress",
                    tournament_name=tournament_name,
                    message=f"Starting ranking computation for {len(courses)} courses...",
                    progress_percent=0
                )
            )

            # Process players in batches
            for batch_num, i in enumerate(range(0, len(player_ids), BATCH_SIZE)):
                if not manager.is_streaming(stream_id):
                    break

                batch_players = player_ids[i:i + BATCH_SIZE]
                batch_rankings = []

                # Rank batch of players across all courses
                for course_id in courses:
                    course_batch_rankings = await RankingStreamService._rank_batch(
                        batch_players, course_id
                    )
                    batch_rankings.extend(course_batch_rankings)

                # Sort batch by score
                batch_rankings.sort(
                    key=lambda x: x["fit_score"],
                    reverse=True
                )
                all_rankings.extend(batch_rankings)

                # Send batch update
                progress = int((batch_num / total_batches) * 100)
                await manager.broadcast(
                    stream_id,
                    RankingStreamMessage(
                        type="update",
                        data={
                            "rankings": [
                                RankingUpdate(**r).model_dump(mode="json")
                                for r in batch_rankings
                            ]
                        },
                        batch_number=batch_num + 1,
                        total_batches=total_batches,
                        tournament_name=tournament_name,
                        progress_percent=progress,
                    )
                )

                # Small delay to allow cancellation
                await asyncio.sleep(0.1)

            # Final aggregation
            aggregate_ranking = await RankingStreamService._aggregate_final(
                all_rankings, top_n
            )

            # Send completion
            await manager.broadcast(
                stream_id,
                RankingStreamMessage(
                    type="complete",
                    data={
                        "aggregate_ranking": aggregate_ranking,
                        "total_players_ranked": len(all_rankings),
                        "courses": courses,
                    },
                    tournament_name=tournament_name,
                    message="Ranking computation complete",
                    progress_percent=100
                )
            )

        except Exception as e:
            logger.error(f"Error in ranking stream: {e}")
            await manager.broadcast(
                stream_id,
                RankingStreamMessage(
                    type="error",
                    message=f"Error during ranking computation: {str(e)}"
                )
            )
        finally:
            manager.set_streaming(stream_id, False)

    @staticmethod
    async def _rank_batch(player_ids: List[str], course_id: str) -> List[Dict]:
        """Rank a batch of players for a course."""
        rankings = []
        for rank, player_id in enumerate(player_ids, 1):
            fit_result = CourseFitService.predict_player_course_fit(player_id, course_id)
            if fit_result:
                rankings.append({
                    "rank": rank,
                    "player_id": player_id,
                    "fit_score": fit_result.get("fit_score", 0),
                    "confidence": fit_result.get("confidence", 0),
                })
        return rankings
    
    @staticmethod
    async def _aggregate_final(
        all_rankings: List[Dict],
        top_n: int
    ) -> List[Dict]:
        """Aggregate rankings and return top N."""
        player_scores = {}
        for ranking in all_rankings:
            player_id = ranking["player_id"]
            if player_id not in player_scores:
                player_scores[player_id] = []
            player_scores[player_id].append(ranking["fit_score"])

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
                "num_courses": len(scores),
            })

        return aggregate
    
class PredictionStreamService:
    """Service for streaming prediction computations."""
    
    @staticmethod
    async def stream_tournament_predictions(
        manager: ConnectionManager,
        stream_id: str,
        tournament_name: str,
        courses: List[str],
        top_n: int = 100,
    ):
        """
        Stream tournament predictions to all connected clients.
        
        Args:
            manager: ConnectionManager instance
            stream_id: Unique stream identifier
            tournament_name: Tournament name
            courses: List of course IDs
            top_n: Number of top players
        """
        manager.set_streaming(stream_id, True)
        
        try:
            # Get all player IDs
            player_ids = [f"player_{i}" for i in range(1, 201)]
            all_predictions = []
            
            total_batches = (len(player_ids) + BATCH_SIZE - 1) // BATCH_SIZE
            
            await manager.broadcast(
                stream_id,
                RankingStreamMessage(
                    type="progress",
                    tournament_name=tournament_name,
                    message=f"Starting prediction computation for {len(courses)} courses...",
                    progress_percent=0,
                )
            )
            
            # Process predictions in batches
            for batch_num, i in enumerate(range(0, len(player_ids), BATCH_SIZE)):
                if not manager.is_streaming(stream_id):
                    break
                
                batch_players = player_ids[i:i + BATCH_SIZE]
                batch_predictions = []
                
                for player_id in batch_players:
                    for course_id in courses:
                        prediction = await PredictionStreamService._predict_player(
                            player_id, course_id
                        )
                        if prediction:
                            batch_predictions.append(prediction)
                
                all_predictions.extend(batch_predictions)
                
                # Send batch update
                progress = int((batch_num / total_batches) * 100)
                await manager.broadcast(
                    stream_id,
                    RankingStreamMessage(
                        type="update",
                        data={
                            "predictions": [
                                PredictionUpdate(**p).model_dump(mode="json")
                                for p in batch_predictions
                            ]
                        },
                        batch_number=batch_num + 1,
                        total_batches=total_batches,
                        tournament_name=tournament_name,
                        progress_percent=progress,
                    )
                )
                
                await asyncio.sleep(0.1)
            
            # Send completion
            await manager.broadcast(
                stream_id,
                RankingStreamMessage(
                    type="complete",
                    data={
                        "total_predictions": len(all_predictions),
                        "courses": courses,
                        "sample_predictions": [
                            PredictionUpdate(**p).model_dump(mode="json")
                            for p in all_predictions[:10]
                        ],
                    },
                    tournament_name=tournament_name,
                    message="Prediction computation complete",
                    progress_percent=100,
                )
            )
            
        except Exception as e:
            logger.error(f"Error in prediction stream: {e}")
            await manager.broadcast(
                stream_id,
                RankingStreamMessage(
                    type="error",
                    message=f"Error during prediction computation: {str(e)}",
                )
            )
        finally:
            manager.set_streaming(stream_id, False)
    
    @staticmethod
    async def _predict_player(player_id: str, course_id: str) -> Optional[Dict]:
        """Predict outcomes for a player at a course."""
        try:
            fit_result = CourseFitService.predict_player_course_fit(player_id, course_id)
            outcome_result = OutcomeService.predict_player_outcomes(player_id, course_id)
            
            if not outcome_result:
                return None
            
            return {
                "player_id": player_id,
                "course_id": course_id,
                "fit_score": fit_result.get("fit_score") if fit_result else None,
                "make_cut_prob": outcome_result.get("make_cut_probability", 0),
                "top_10_prob": outcome_result.get("top_10_probability", 0),
                "win_prob": outcome_result.get("win_probability", 0),
                "confidence": outcome_result.get("overall_confidence", 0),
            }
        except Exception as e:
            logger.error(f"Error predicting for {player_id}/{course_id}: {e}")
            return None


# Global manager instance
manager = ConnectionManager()