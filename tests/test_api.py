"""Tests for API endpoints using FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.memory.database import Base, User, get_db


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

    # Seed users
    session.add_all([
        User(
            id="founder",
            name="Founder",
            role="Founder",
            title="Co-founder",
            expertise_areas=["grocery operations"],
            communication_style={"style": "direct"},
        ),
        User(
            id="cofounder",
            name="Cofounder",
            role="Co-founder",
            title="Co-founder",
            expertise_areas=["forecasting"],
            communication_style={"style": "concise"},
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


# ---------- Health ----------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------- Auth ----------


def test_missing_auth(client):
    resp = client.get("/api/users/founder")
    assert resp.status_code == 422


def test_invalid_user(client):
    resp = client.get("/api/users/nobody", headers={"Authorization": "Bearer nobody"})
    assert resp.status_code == 401


# ---------- Users ----------


def test_list_users(client):
    resp = client.get("/api/users/", headers=AUTH)
    assert resp.status_code == 200
    users = resp.json()
    assert len(users) == 2


def test_get_user(client):
    resp = client.get("/api/users/founder", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "founder"
    assert data["title"] == "Co-founder"


def test_update_user(client):
    resp = client.put(
        "/api/users/founder",
        headers=AUTH,
        json={"title": "Co-founder and Director"},
    )
    assert resp.status_code == 200
    assert resp.json()["title"] == "Co-founder and Director"


def test_get_user_not_found(client):
    resp = client.get("/api/users/nobody", headers=AUTH)
    assert resp.status_code == 404


# ---------- Timeline ----------


def test_create_timeline_event(client):
    resp = client.post(
        "/api/company/timeline",
        headers=AUTH,
        json={
            "title": "Founded company",
            "event_type": "milestone",
            "date": "2024-01-01T00:00:00",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Founded company"
    assert data["created_by_id"] == "founder"


def test_list_timeline_events(client):
    client.post(
        "/api/company/timeline",
        headers=AUTH,
        json={"title": "Event 1", "date": "2024-01-01T00:00:00"},
    )
    client.post(
        "/api/company/timeline",
        headers=AUTH,
        json={"title": "Event 2", "date": "2024-06-01T00:00:00"},
    )
    resp = client.get("/api/company/timeline", headers=AUTH)
    assert resp.status_code == 200
    assert len(resp.json()) >= 2


def test_get_timeline_event(client):
    create = client.post(
        "/api/company/timeline",
        headers=AUTH,
        json={"title": "Get me", "date": "2024-01-01T00:00:00"},
    )
    event_id = create.json()["id"]
    resp = client.get(f"/api/company/timeline/{event_id}", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["title"] == "Get me"


def test_update_timeline_event(client):
    create = client.post(
        "/api/company/timeline",
        headers=AUTH,
        json={"title": "Update me", "date": "2024-01-01T00:00:00"},
    )
    event_id = create.json()["id"]
    resp = client.put(
        f"/api/company/timeline/{event_id}",
        headers=AUTH,
        json={"outcome": "Great success"},
    )
    assert resp.status_code == 200
    assert resp.json()["outcome"] == "Great success"


# ---------- Decisions ----------


def test_create_decision(client):
    resp = client.post(
        "/api/company/decisions",
        headers=AUTH,
        json={
            "decision": "Use FAISS for vector store",
            "category": "product",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["proposed_by_id"] == "founder"


def test_list_decisions(client):
    client.post(
        "/api/company/decisions",
        headers=AUTH,
        json={"decision": "Decision 1", "category": "product"},
    )
    resp = client.get("/api/company/decisions", headers=AUTH)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_update_decision(client):
    create = client.post(
        "/api/company/decisions",
        headers=AUTH,
        json={"decision": "Update me"},
    )
    did = create.json()["id"]
    resp = client.put(
        f"/api/company/decisions/{did}",
        headers=AUTH,
        json={"outcome": "Worked great"},
    )
    assert resp.status_code == 200
    assert resp.json()["outcome"] == "Worked great"


# ---------- Hypotheses ----------


def test_create_hypothesis(client):
    resp = client.post(
        "/api/company/hypotheses",
        headers=AUTH,
        json={
            "hypothesis": "SMBs will pay for AI consulting",
            "category": "market",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["owner_id"] == "founder"


def test_list_hypotheses(client):
    client.post(
        "/api/company/hypotheses",
        headers=AUTH,
        json={"hypothesis": "Test hypothesis"},
    )
    resp = client.get("/api/company/hypotheses", headers=AUTH)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_update_hypothesis(client):
    create = client.post(
        "/api/company/hypotheses",
        headers=AUTH,
        json={"hypothesis": "Update me"},
    )
    hid = create.json()["id"]
    resp = client.put(
        f"/api/company/hypotheses/{hid}",
        headers=AUTH,
        json={"result": "validated", "learnings": "Confirmed"},
    )
    assert resp.status_code == 200
    assert resp.json()["result"] == "validated"


# ---------- Metrics ----------


def test_create_metric(client):
    resp = client.post(
        "/api/company/metrics",
        headers=AUTH,
        json={"date": "2024-06-01", "mrr": 1000.0, "active_users": 50},
    )
    assert resp.status_code == 201
    assert resp.json()["mrr"] == 1000.0


def test_list_metrics(client):
    client.post(
        "/api/company/metrics",
        headers=AUTH,
        json={"date": "2024-06-01", "mrr": 1000.0},
    )
    resp = client.get("/api/company/metrics", headers=AUTH)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


# ---------- Problem Statements ----------


def test_create_problem_statement(client):
    resp = client.post(
        "/api/validation/problem",
        headers=AUTH,
        json={
            "problem": "SMBs waste 10hrs/week on manual reporting",
            "solution": "AI-powered automated reports",
            "target_customer": "SMB ops managers",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["problem"] == "SMBs waste 10hrs/week on manual reporting"
    assert data["version"] == 1
    assert data["is_active"] is True
    assert data["created_by_id"] == "founder"


def test_get_active_problem_statement(client):
    client.post(
        "/api/validation/problem",
        headers=AUTH,
        json={
            "problem": "Test problem",
            "solution": "Test solution",
            "target_customer": "Test customer",
        },
    )
    resp = client.get("/api/validation/problem", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["problem"] == "Test problem"


def test_problem_statement_evolution(client):
    client.post(
        "/api/validation/problem",
        headers=AUTH,
        json={
            "problem": "Problem v1",
            "solution": "Solution v1",
            "target_customer": "Customer v1",
        },
    )
    resp2 = client.post(
        "/api/validation/problem",
        headers=AUTH,
        json={
            "problem": "Problem v2",
            "solution": "Solution v2",
            "target_customer": "Customer v2",
        },
    )
    assert resp2.status_code == 201
    data = resp2.json()
    assert data["version"] == 2
    assert data["is_active"] is True

    # Active should be v2
    active = client.get("/api/validation/problem", headers=AUTH)
    assert active.json()["version"] == 2


def test_get_problem_statement_history(client):
    client.post(
        "/api/validation/problem",
        headers=AUTH,
        json={
            "problem": "P1",
            "solution": "S1",
            "target_customer": "C1",
        },
    )
    client.post(
        "/api/validation/problem",
        headers=AUTH,
        json={
            "problem": "P2",
            "solution": "S2",
            "target_customer": "C2",
        },
    )
    resp = client.get("/api/validation/problem/history", headers=AUTH)
    assert resp.status_code == 200
    history = resp.json()
    assert len(history) == 2
    assert history[0]["version"] == 2  # desc order


# ---------- Validation Hypotheses ----------


def test_create_hypothesis_manual(client):
    resp = client.post(
        "/api/validation/hypotheses",
        headers=AUTH,
        json={
            "hypothesis": "Ops managers spend >5hrs/week on manual reports",
            "category": "problem",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["hypothesis"] == "Ops managers spend >5hrs/week on manual reports"
    assert data["auto_generated"] is False
    assert data["owner_id"] == "founder"


def test_add_evidence_to_hypothesis(client):
    create = client.post(
        "/api/validation/hypotheses",
        headers=AUTH,
        json={"hypothesis": "Evidence test", "category": "problem"},
    )
    hid = create.json()["id"]

    resp = client.post(
        f"/api/validation/hypotheses/{hid}/evidence",
        headers=AUTH,
        json={
            "evidence_type": "for",
            "evidence": {"quote": "I spend 6 hours pulling numbers", "source": "Interview #1"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["evidence_for"]) == 1
    assert data["evidence_for"][0]["quote"] == "I spend 6 hours pulling numbers"

    # Add counter-evidence
    resp2 = client.post(
        f"/api/validation/hypotheses/{hid}/evidence",
        headers=AUTH,
        json={
            "evidence_type": "against",
            "evidence": {"quote": "We already automated this", "source": "Interview #2"},
        },
    )
    assert resp2.status_code == 200
    assert len(resp2.json()["evidence_against"]) == 1


def test_hypothesis_with_evidence_fields(client):
    resp = client.post(
        "/api/validation/hypotheses",
        headers=AUTH,
        json={
            "hypothesis": "With evidence",
            "category": "market",
            "evidence_for": [{"quote": "Yes", "source": "Call 1"}],
            "evidence_against": [],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert len(data["evidence_for"]) == 1
    assert data["evidence_against"] == []
