"""
End-to-end tests for /api/data/courses and /api/predictions/player-outcome endpoints.
Run with: pytest tests/test_endpoints.py -v
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

# Valid IDs that exist in the processed CSV files
VALID_PLAYER = "A. Ancer"
VALID_COURSE = "Muirfield Village Golf Club - Dublin, OH"

MOCK_OUTCOME = {
    "player_id": VALID_PLAYER,
    "course_id": VALID_COURSE,
    "make_cut_probability": 0.72,
    "top_10_probability": 0.18,
    "win_probability": 0.04,
    "overall_confidence": 0.65,
}

MOCK_FIT = {
    "player_id": VALID_PLAYER,
    "course_id": VALID_COURSE,
    "fit_score": 0.45,
    "confidence": 0.72,
    "interpretation": "Higher is better",
}


# ---------------------------------------------------------------------------
# /api/data/courses
# ---------------------------------------------------------------------------

def test_get_courses_returns_200():
    response = client.get("/api/data/courses")
    assert response.status_code == 200


def test_get_courses_shape():
    response = client.get("/api/data/courses")
    data = response.json()
    assert "courses" in data
    assert "total_courses" in data
    assert isinstance(data["courses"], list)
    assert data["total_courses"] == len(data["courses"])


def test_get_courses_nonempty():
    response = client.get("/api/data/courses")
    data = response.json()
    assert len(data["courses"]) > 0


def test_get_courses_entry_fields():
    response = client.get("/api/data/courses")
    first = response.json()["courses"][0]
    assert "id" in first
    assert "name" in first


# ---------------------------------------------------------------------------
# /api/predictions/player-outcome — validation
# ---------------------------------------------------------------------------

def test_predict_missing_course_field_returns_422():
    response = client.post("/api/predictions/player-outcome", json={"player_id": VALID_PLAYER})
    assert response.status_code == 422


def test_predict_missing_player_field_returns_422():
    response = client.post("/api/predictions/player-outcome", json={"course_id": VALID_COURSE})
    assert response.status_code == 422


def test_predict_unknown_player_returns_404():
    response = client.post(
        "/api/predictions/player-outcome",
        json={"player_id": "ghost_player_xyz", "course_id": VALID_COURSE},
    )
    assert response.status_code == 404
    # main.py's custom exception handler returns {"error": ..., "timestamp": ...}
    assert "not found" in response.json()["error"].lower()


def test_predict_unknown_course_returns_404():
    response = client.post(
        "/api/predictions/player-outcome",
        json={"player_id": VALID_PLAYER, "course_id": "fake_course_xyz"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["error"].lower()


# ---------------------------------------------------------------------------
# /api/predictions/player-outcome — success path (services mocked)
# ---------------------------------------------------------------------------

def test_predict_valid_inputs_returns_200():
    with patch(
        "backend.app.routers.predictions.OutcomeService.predict_player_outcomes",
        return_value=MOCK_OUTCOME,
    ), patch(
        "backend.app.routers.predictions.CourseFitService.predict_player_course_fit",
        return_value=MOCK_FIT,
    ):
        response = client.post(
            "/api/predictions/player-outcome",
            json={"player_id": VALID_PLAYER, "course_id": VALID_COURSE},
        )
    assert response.status_code == 200


def test_predict_response_contains_probabilities():
    with patch(
        "backend.app.routers.predictions.OutcomeService.predict_player_outcomes",
        return_value=MOCK_OUTCOME,
    ), patch(
        "backend.app.routers.predictions.CourseFitService.predict_player_course_fit",
        return_value=MOCK_FIT,
    ):
        data = client.post(
            "/api/predictions/player-outcome",
            json={"player_id": VALID_PLAYER, "course_id": VALID_COURSE},
        ).json()

    probs = data["outcome_probabilities"]
    assert 0 <= probs["make_cut"] <= 1
    assert 0 <= probs["top_10"] <= 1
    assert 0 <= probs["win"] <= 1


def test_predict_response_echoes_ids():
    with patch(
        "backend.app.routers.predictions.OutcomeService.predict_player_outcomes",
        return_value=MOCK_OUTCOME,
    ), patch(
        "backend.app.routers.predictions.CourseFitService.predict_player_course_fit",
        return_value=MOCK_FIT,
    ):
        data = client.post(
            "/api/predictions/player-outcome",
            json={"player_id": VALID_PLAYER, "course_id": VALID_COURSE},
        ).json()

    assert data["player_id"] == VALID_PLAYER
    assert data["course_id"] == VALID_COURSE
