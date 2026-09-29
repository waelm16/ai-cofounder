"""Tests for database models and memory managers."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.memory.database import (
    Base,
    Decision,
    Hypothesis,
    Metric,
    ProblemStatement,
    TimelineEvent,
    User,
)
from src.memory.company_timeline import TimelineManager
from src.memory.decisions_tracker import DecisionTracker
from src.memory.hypotheses_tracker import HypothesesTracker
from src.memory.problem_statement import ProblemStatementManager
from src.memory.user_profile import UserProfileManager


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_users(db):
    founder = User(
        id="founder",
        name="Founder",
        role="Founder",
        title="Co-founder",
        expertise_areas=["grocery operations", "purchasing"],
        communication_style={"style": "direct"},
    )
    cofounder = User(
        id="cofounder",
        name="Cofounder",
        role="Co-founder",
        title="Co-founder",
        expertise_areas=["forecasting", "product design"],
        communication_style={"style": "concise"},
    )
    db.add_all([founder, cofounder])
    db.commit()
    return founder, cofounder


# ---------- DB Creation ----------


def test_create_tables(db):
    """All tables should be created without error."""
    tables = Base.metadata.tables.keys()
    assert "users" in tables
    assert "timeline" in tables
    assert "decisions" in tables
    assert "hypotheses" in tables
    assert "metrics" in tables
    assert "conversations" in tables
    assert "disagreements" in tables
    assert "collaboration_patterns" in tables
    assert "knowledge_transfers" in tables
    assert "reflections" in tables
    assert "tasks" in tables
    assert "problem_statements" in tables
    assert "customer_calls" in tables


# ---------- User CRUD ----------


def test_create_user(db, sample_users):
    founder, _ = sample_users
    fetched = db.query(User).filter(User.id == "founder").first()
    assert fetched is not None
    assert fetched.name == "Founder"
    assert fetched.role == "Founder"


def test_user_profile_manager_get(db, sample_users):
    manager = UserProfileManager(db)
    user = manager.get_user("founder")
    assert user is not None
    assert user.title == "Co-founder"


def test_user_profile_manager_get_cofounder(db, sample_users):
    manager = UserProfileManager(db)
    cofounder = manager.get_cofounder("founder")
    assert cofounder is not None
    assert cofounder.id == "cofounder"


def test_user_profile_manager_update(db, sample_users):
    manager = UserProfileManager(db)
    updated = manager.update_user("founder", {"title": "Co-founder and Director"})
    assert updated.title == "Co-founder and Director"


def test_user_profile_manager_enrich(db, sample_users):
    manager = UserProfileManager(db)
    enriched = manager.enrich_profile("founder", {
        "expertise_areas": ["supplier negotiation"],
        "communication_style": {"prefers_examples": True},
    })
    assert "supplier negotiation" in enriched.expertise_areas
    assert "grocery operations" in enriched.expertise_areas
    assert enriched.communication_style["prefers_examples"] is True
    assert enriched.communication_style["style"] == "direct"


# ---------- Timeline ----------


def test_timeline_add_and_get(db, sample_users):
    manager = TimelineManager(db)
    event = manager.add_event(
        title="Founded company",
        created_by="founder",
        date=datetime(2024, 1, 1),
        event_type="milestone",
        participants=["founder", "cofounder"],
    )
    assert event.id is not None
    assert event.title == "Founded company"

    fetched = manager.get_event(event.id)
    assert fetched.event_type == "milestone"


def test_timeline_filter_by_type(db, sample_users):
    manager = TimelineManager(db)
    manager.add_event(title="Launch", created_by="founder", date=datetime(2024, 6, 1), event_type="launch")
    manager.add_event(title="Pivot", created_by="founder", date=datetime(2024, 3, 1), event_type="pivot")

    launches = manager.get_events(event_type="launch")
    assert len(launches) == 1
    assert launches[0].title == "Launch"


def test_timeline_recent(db, sample_users):
    manager = TimelineManager(db)
    manager.add_event(title="E1", created_by="founder", date=datetime(2024, 1, 1))
    manager.add_event(title="E2", created_by="founder", date=datetime(2024, 6, 1))

    recent = manager.get_recent_events(limit=1)
    assert len(recent) == 1
    assert recent[0].title == "E2"


# ---------- Decisions ----------


def test_decision_add_and_get(db, sample_users):
    tracker = DecisionTracker(db)
    decision = tracker.add_decision(
        decision="Use FAISS for vector store",
        proposed_by="founder",
        category="product",
        rationales={"founder": "Free and fast", "cofounder": "Keeps costs low"},
        consensus=True,
        consensus_type="unanimous",
    )
    assert decision.id is not None

    decisions = tracker.get_decisions(category="product")
    assert len(decisions) == 1


def test_decision_update_outcome(db, sample_users):
    tracker = DecisionTracker(db)
    d = tracker.add_decision(decision="Price at $29/mo", proposed_by="cofounder", category="pricing")
    updated = tracker.update_decision(d.id, {"outcome": "Worked well, good conversion"})
    assert updated.outcome == "Worked well, good conversion"


# ---------- Hypotheses ----------


def test_hypothesis_add_and_get(db, sample_users):
    tracker = HypothesesTracker(db)
    h = tracker.add_hypothesis(
        hypothesis="SMBs will pay for AI consulting",
        owner_id="cofounder",
        category="market",
        validation_criteria=[{"metric": "signups", "target": 10}],
    )
    assert h.id is not None

    active = tracker.get_active_hypotheses()
    assert len(active) == 1


def test_hypothesis_validate(db, sample_users):
    tracker = HypothesesTracker(db)
    h = tracker.add_hypothesis(hypothesis="Test hyp", owner_id="founder", category="product")
    updated = tracker.update_hypothesis(h.id, {
        "result": "validated",
        "learnings": "Confirmed by 5 customer interviews",
        "end_date": datetime(2024, 3, 1),
    })
    assert updated.result == "validated"

    active = tracker.get_active_hypotheses()
    assert len(active) == 0


def test_hypothesis_evidence_fields(db, sample_users):
    tracker = HypothesesTracker(db)
    h = tracker.add_hypothesis(
        hypothesis="Users need automated reports",
        owner_id="founder",
        category="problem",
        evidence_for=[{"quote": "I hate manual reports", "source": "Interview #1"}],
        evidence_against=[],
        validation_score=0.6,
        auto_generated=True,
        source_call_id=42,
    )
    assert h.auto_generated is True
    assert h.source_call_id == 42
    assert len(h.evidence_for) == 1
    assert h.evidence_for[0]["quote"] == "I hate manual reports"
    assert h.evidence_against == []
    assert h.validation_score == 0.6


def test_hypothesis_add_evidence(db, sample_users):
    tracker = HypothesesTracker(db)
    h = tracker.add_hypothesis(hypothesis="Test evidence", owner_id="founder")

    updated = tracker.add_evidence(h.id, "for", {"quote": "Yes!", "source": "Call 1"})
    assert len(updated.evidence_for) == 1
    assert updated.evidence_for[0]["quote"] == "Yes!"

    updated = tracker.add_evidence(h.id, "against", {"quote": "No", "source": "Call 2"})
    assert len(updated.evidence_against) == 1

    # Add another supporting evidence
    updated = tracker.add_evidence(h.id, "for", {"quote": "Definitely", "source": "Call 3"})
    assert len(updated.evidence_for) == 2


# ---------- Problem Statements ----------


def test_problem_statement_create_and_get(db, sample_users):
    manager = ProblemStatementManager(db)
    ps = manager.create_problem_statement(
        problem="SMBs waste 10hrs/week on manual reporting",
        solution="AI-powered automated reports",
        target_customer="SMB ops managers",
        created_by_id="founder",
    )
    assert ps.id is not None
    assert ps.version == 1
    assert ps.is_active is True

    active = manager.get_active()
    assert active.id == ps.id


def test_problem_statement_versioning(db, sample_users):
    manager = ProblemStatementManager(db)
    ps1 = manager.create_problem_statement(
        problem="Problem v1",
        solution="Solution v1",
        target_customer="Customer v1",
        created_by_id="founder",
    )
    ps2 = manager.create_problem_statement(
        problem="Problem v2",
        solution="Solution v2",
        target_customer="Customer v2",
        created_by_id="cofounder",
    )

    # Only latest is active
    active = manager.get_active()
    assert active.id == ps2.id
    assert active.version == 2

    # First is deactivated
    db.refresh(ps1)
    assert ps1.is_active is False

    # All versions returned
    all_versions = manager.get_all_versions()
    assert len(all_versions) == 2
    assert all_versions[0].version == 2  # desc order
