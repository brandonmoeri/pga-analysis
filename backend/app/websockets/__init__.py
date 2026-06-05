"""WebSocket routers for streaming computations."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import logging
import asyncio
from datetime import datetime

from backend.app.websockets.ranking_stream import (
    manager,
    RankingStreamService,
    PredictionStreamService,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["websocket"])

@router.websocket("/ws/rankings/tournament")
async def websocket_tournament_rankings(
    websocket: WebSocket,
    tournament_name: str = Query(..., description="Tournament name"),
    courses: str = Query(..., description="Comma-separated course IDs"),
    top_n: int = Query(100, ge=1, le=500, description="Top N players"),
):
    """
    WebSocket endpoint for streaming tournament predictions.
    
    Usage:
        ws://localhost:8000/ws/predictions/tournament?tournament_name=masters_2024&courses=augusta_national&top_n=100
    
    Messages received:
        - {type: "progress", progress_percent: int, message: str}
        - {type: "update", batch_number: int, total_batches: int, data: {predictions: [...]}}
        - {type: "complete", data: {total_predictions: int, sample_predictions: [...]}}
        - {type: "error", message: str}
    """
    stream_id = f"predictions_{tournament_name}_{datetime.now().timestamp()}"

    try:
        await manager.connect(websocket, stream_id)

        # Parse courses
        course_list = [c.strip() for c in courses.split(",")]

        # Start streaming task
        stream_task = asyncio.create_task(
            PredictionStreamService.stream_tournament_predictions(
                manager=manager,
                stream_id=stream_id,
                tournament_name=tournament_name,
                courses=course_list,
                top_n=top_n
            )
        )

        # Keep connection open
        try:
            while True:
                data = await websocket.receive_text()
                if data == "cancel":
                    stream_task.cancel()
                    break
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from stream: {stream_id}")
            stream_task.cancel()

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, stream_id)