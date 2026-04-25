from fastapi.testclient import TestClient

from social_engineer_arena.models import ArenaState
from social_engineer_arena.server.app import app
from social_engineer_arena.server.app import env


def test_canonical_and_api_endpoints_are_available():
    client = TestClient(app)

    reset = client.post("/reset")
    assert reset.status_code == 200
    assert "scenario_id" in reset.json()

    api_reset = client.post("/api/reset")
    assert api_reset.status_code == 200
    assert "scenario_id" in api_reset.json()

    action = {
        "verdict": "unknown",
        "explanation": "Need more evidence.",
        "cues_found": [],
        "response": "",
        "safety_boundary": "Defensive analysis only.",
    }
    step = client.post("/step", json=action)
    assert step.status_code == 200
    assert "observation" in step.json()
    assert "reward" in step.json()
    assert "done" in step.json()

    api_step = client.post("/api/step", json=action)
    assert api_step.status_code == 200
    assert "observation" in api_step.json()

    state = client.get("/state")
    assert state.status_code == 200
    assert "scenario_id" in state.json()

    api_state = client.get("/api/state")
    assert api_state.status_code == 200
    assert "scenario_id" in api_state.json()


def test_step_requires_reset_first():
    client = TestClient(app)
    env._state = ArenaState()
    action = {
        "verdict": "unknown",
        "explanation": "Need more evidence.",
        "cues_found": [],
        "response": "",
        "safety_boundary": "Defensive analysis only.",
    }
    response = client.post("/step", json=action)
    assert response.status_code == 409
    assert "Call reset() before step()." in response.text
