"""Tests for customer calls API endpoints with mocked LLM calls."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.memory.database import Base, Hypothesis, User, get_db


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    session.add_all([
        User(
            id="founder",
            name="Founder",
            role="Founder",
            title="Co-founder",
        ),
        User(
            id="cofounder",
            name="Cofounder",
            role="Co-founder",
            title="Co-founder",
        ),
    ])
    session.commit()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


AUTH = {"Authorization": "Bearer founder"}


# ---------- CRUD ----------


def test_create_call(client):
    resp = client.post(
        "/api/calls/",
        headers=AUTH,
        json={
            "call_date": "2024-06-01T10:00:00",
            "customer_name": "Jane Doe",
            "customer_company": "Acme Corp",
            "customer_role": "VP Operations",
            "call_type": "discovery",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["customer_name"] == "Jane Doe"
    assert data["status"] == "scheduled"
    assert data["created_by_id"] == "founder"


def test_list_calls(client):
    client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-01T10:00:00", "customer_name": "Jane"},
    )
    client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-02T10:00:00", "customer_name": "Bob"},
    )
    resp = client.get("/api/calls/", headers=AUTH)
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_list_calls_filter_status(client):
    create = client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-01T10:00:00", "customer_name": "Jane"},
    )
    call_id = create.json()["id"]
    client.put(
        f"/api/calls/{call_id}",
        headers=AUTH,
        json={"status": "completed"},
    )
    client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-02T10:00:00", "customer_name": "Bob"},
    )

    resp = client.get("/api/calls/?status=scheduled", headers=AUTH)
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["customer_name"] == "Bob"


def test_get_call(client):
    create = client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-01T10:00:00", "customer_name": "Jane"},
    )
    call_id = create.json()["id"]
    resp = client.get(f"/api/calls/{call_id}", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["customer_name"] == "Jane"


def test_get_call_not_found(client):
    resp = client.get("/api/calls/999", headers=AUTH)
    assert resp.status_code == 404


def test_update_call(client):
    create = client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-01T10:00:00", "customer_name": "Jane"},
    )
    call_id = create.json()["id"]
    resp = client.put(
        f"/api/calls/{call_id}",
        headers=AUTH,
        json={
            "transcript": "Q: Tell me about your workflow...\nA: We spend hours on reports.",
            "status": "completed",
            "duration_minutes": 30,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["duration_minutes"] == 30
    assert "workflow" in data["transcript"]


def test_update_call_not_found(client):
    resp = client.put(
        "/api/calls/999",
        headers=AUTH,
        json={"status": "completed"},
    )
    assert resp.status_code == 404


# ---------- Pre-call questions (mocked) ----------


MOCK_QUESTIONS = [
    {
        "question": "Walk me through how you handled reporting last week?",
        "rationale": "Grounds the conversation in recent, specific behavior",
        "targeted_hypothesis_id": None,
    },
    {
        "question": "What was the most frustrating part of that process?",
        "rationale": "Uncovers pain points without leading",
        "targeted_hypothesis_id": None,
    },
]


@patch("src.api.routes.calls.generate_pre_call_questions", return_value=MOCK_QUESTIONS)
def test_generate_questions(mock_gen, client):
    create = client.post(
        "/api/calls/",
        headers=AUTH,
        json={
            "call_date": "2024-06-01T10:00:00",
            "customer_name": "Jane Doe",
            "customer_company": "Acme Corp",
        },
    )
    call_id = create.json()["id"]

    resp = client.post(f"/api/calls/{call_id}/questions", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["call_id"] == call_id
    assert len(data["questions"]) == 2
    assert "reporting" in data["questions"][0]["question"]

    mock_gen.assert_called_once()

    # Verify questions are saved to the call record
    call_resp = client.get(f"/api/calls/{call_id}", headers=AUTH)
    assert len(call_resp.json()["pre_call_questions"]) == 2


@patch("src.api.routes.calls.generate_pre_call_questions", return_value=MOCK_QUESTIONS)
def test_generate_questions_call_not_found(mock_gen, client):
    resp = client.post("/api/calls/999/questions", headers=AUTH)
    assert resp.status_code == 404
    mock_gen.assert_not_called()


# ---------- Transcript analysis (mocked) ----------


MOCK_ANALYSIS = {
    "key_insights": [
        {"insight": "Spends 6 hours weekly on manual reports", "quote": "I spend about 6 hours pulling numbers", "importance": "high"},
    ],
    "pain_points": [
        {"pain_point": "Manual reporting", "severity": "high", "quote": "it's the worst part of my week"},
    ],
    "sentiment": "negative",
    "wtp_signals": [
        {"signal": "Would pay $50/mo to automate", "strength": "strong", "quote": "I'd happily pay fifty bucks a month"},
    ],
    "next_steps": [
        {"action": "Send demo link", "owner": "us", "deadline": "this week"},
    ],
    "new_hypotheses": [
        {"hypothesis": "Mid-market ops managers spend >5hrs/week on manual reports", "category": "problem", "evidence": "6 hours pulling numbers"},
    ],
    "evidence_for_existing": [],
}


@patch("src.api.routes.calls.analyze_transcript", return_value=MOCK_ANALYSIS)
def test_analyze_call(mock_analyze, client):
    # Create and add transcript
    create = client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-01T10:00:00", "customer_name": "Jane Doe"},
    )
    call_id = create.json()["id"]
    client.put(
        f"/api/calls/{call_id}",
        headers=AUTH,
        json={"transcript": "Q: Tell me about reporting...\nA: I spend about 6 hours pulling numbers.", "status": "completed"},
    )

    resp = client.post(f"/api/calls/{call_id}/analyze", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["call_id"] == call_id
    assert data["sentiment"] == "negative"
    assert len(data["key_insights"]) == 1
    assert len(data["pain_points"]) == 1
    assert len(data["wtp_signals"]) == 1
    assert len(data["new_hypotheses"]) == 1
    assert data["new_hypotheses"][0]["hypothesis"] == "Mid-market ops managers spend >5hrs/week on manual reports"

    mock_analyze.assert_called_once()

    # Verify call record updated
    call_resp = client.get(f"/api/calls/{call_id}", headers=AUTH)
    call_data = call_resp.json()
    assert call_data["status"] == "analyzed"
    assert call_data["sentiment"] == "negative"


@patch("src.api.routes.calls.analyze_transcript", return_value=MOCK_ANALYSIS)
def test_analyze_call_no_transcript(mock_analyze, client):
    create = client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-01T10:00:00", "customer_name": "Jane"},
    )
    call_id = create.json()["id"]

    resp = client.post(f"/api/calls/{call_id}/analyze", headers=AUTH)
    assert resp.status_code == 400
    assert "no transcript" in resp.json()["detail"].lower()
    mock_analyze.assert_not_called()


@patch("src.api.routes.calls.analyze_transcript", return_value=MOCK_ANALYSIS)
def test_analyze_call_not_found(mock_analyze, client):
    resp = client.post("/api/calls/999/analyze", headers=AUTH)
    assert resp.status_code == 404
    mock_analyze.assert_not_called()


# ---------- Evidence mapping ----------


MOCK_ANALYSIS_WITH_EVIDENCE = {
    "key_insights": [],
    "pain_points": [],
    "sentiment": "neutral",
    "wtp_signals": [],
    "next_steps": [],
    "new_hypotheses": [],
    "evidence_for_existing": [
        {"hypothesis_id": None, "evidence_type": "for", "quote": "Confirmed!", "source": "call"},
    ],
}


@patch("src.api.routes.calls.analyze_transcript")
def test_analyze_maps_evidence_to_existing_hypothesis(mock_analyze, client, db_session):
    # Create a hypothesis directly in the DB
    from datetime import datetime
    hyp = Hypothesis(
        hypothesis="Users hate manual reports",
        owner_id="founder",
        start_date=datetime.utcnow(),
    )
    db_session.add(hyp)
    db_session.commit()
    db_session.refresh(hyp)

    # Set up mock with the real hypothesis ID
    analysis = {
        **MOCK_ANALYSIS_WITH_EVIDENCE,
        "evidence_for_existing": [
            {"hypothesis_id": hyp.id, "evidence_type": "for", "quote": "Confirmed!", "source": "call"},
        ],
    }
    mock_analyze.return_value = analysis

    # Create call with transcript
    create = client.post(
        "/api/calls/",
        headers=AUTH,
        json={"call_date": "2024-06-01T10:00:00", "customer_name": "Jane"},
    )
    call_id = create.json()["id"]
    client.put(
        f"/api/calls/{call_id}",
        headers=AUTH,
        json={"transcript": "Some transcript...", "status": "completed"},
    )

    resp = client.post(f"/api/calls/{call_id}/analyze", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["evidence_mapped"]) == 1

    # Verify hypothesis was updated with evidence
    db_session.refresh(hyp)
    assert len(hyp.evidence_for) == 1
    assert hyp.evidence_for[0]["quote"] == "Confirmed!"
