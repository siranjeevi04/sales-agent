"""
Integration tests for the Sales Assistant Agent API.
Set ANTHROPIC_API_KEY in environment before running.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_creates_session():
    r = client.post("/chat", json={"message": "Hi, what plans do you offer?"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0
    assert len(data["response"]) > 0


def test_chat_persists_session():
    # First turn
    r1 = client.post("/chat", json={"message": "What is the Pro plan price?"})
    session_id = r1.json()["session_id"]

    # Second turn with same session
    r2 = client.post("/chat", json={"session_id": session_id, "message": "Tell me more about it."})
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id


def test_history_endpoint():
    # Create a session
    r = client.post("/chat", json={"message": "Show me your enterprise plan."})
    session_id = r.json()["session_id"]

    # Get history
    h = client.get(f"/history/{session_id}")
    assert h.status_code == 200
    data = h.json()
    assert data["session_id"] == session_id
    assert len(data["messages"]) >= 2  # user + assistant


def test_history_not_found():
    r = client.get("/history/nonexistent-session-id")
    assert r.status_code == 404


def test_tool_use_add_to_cart():
    r = client.post("/chat", json={"message": "Add the Pro plan to my cart."})
    assert r.status_code == 200
    data = r.json()
    assert "add_to_cart" in data.get("tool_calls_made", [])


def test_tool_use_search():
    r = client.post("/chat", json={"message": "Do you have any storage add-ons?"})
    assert r.status_code == 200
    data = r.json()
    assert any(t in data.get("tool_calls_made", []) for t in ["search_catalog", "get_recommendations"])


def test_tool_use_order_status():
    r = client.post("/chat", json={"message": "Can you check order ORD-001?"})
    assert r.status_code == 200
    data = r.json()
    assert "check_order_status" in data.get("tool_calls_made", [])


def test_evaluation_returned():
    r = client.post("/chat", json={"message": "What plan is best for a 10-person team?"})
    assert r.status_code == 200
    data = r.json()
    if data.get("evaluation"):
        ev = data["evaluation"]
        assert 1 <= ev["helpfulness"] <= 5
        assert 1 <= ev["accuracy"] <= 5
        assert 1 <= ev["tone"] <= 5
        assert 1 <= ev["overall"] <= 5


def test_sessions_list():
    r = client.get("/sessions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
