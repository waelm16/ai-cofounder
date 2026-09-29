"""Tests for CustomerCall model and CustomerCallManager."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.memory.database import Base, CustomerCall, User
from src.memory.customer_calls import CustomerCallManager


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
    founder = User(id="founder", name="Founder", role="Founder")
    cofounder = User(id="cofounder", name="Cofounder", role="Co-founder")
    db.add_all([founder, cofounder])
    db.commit()
    return founder, cofounder


# ---------- Model ----------


def test_customer_call_table_exists(db):
    tables = Base.metadata.tables.keys()
    assert "customer_calls" in tables


def test_create_customer_call_record(db, sample_users):
    call = CustomerCall(
        call_date=datetime(2024, 6, 1),
        customer_name="Jane Doe",
        customer_company="Acme Corp",
        customer_role="VP Operations",
        call_type="discovery",
        status="scheduled",
        created_by_id="founder",
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    assert call.id is not None
    assert call.customer_name == "Jane Doe"
    assert call.status == "scheduled"


# ---------- Manager CRUD ----------


def test_create_call(db, sample_users):
    manager = CustomerCallManager(db)
    call = manager.create_call(
        call_date=datetime(2024, 6, 1),
        customer_name="Jane Doe",
        created_by_id="founder",
        customer_company="Acme Corp",
        customer_role="VP Operations",
        call_type="discovery",
    )
    assert call.id is not None
    assert call.customer_name == "Jane Doe"
    assert call.status == "scheduled"
    assert call.created_by_id == "founder"


def test_get_call(db, sample_users):
    manager = CustomerCallManager(db)
    created = manager.create_call(
        call_date=datetime(2024, 6, 1),
        customer_name="Jane Doe",
        created_by_id="founder",
    )
    fetched = manager.get_call(created.id)
    assert fetched is not None
    assert fetched.customer_name == "Jane Doe"


def test_get_call_not_found(db, sample_users):
    manager = CustomerCallManager(db)
    assert manager.get_call(999) is None


def test_get_calls_filter_status(db, sample_users):
    manager = CustomerCallManager(db)
    manager.create_call(
        call_date=datetime(2024, 6, 1),
        customer_name="Jane",
        created_by_id="founder",
    )
    manager.create_call(
        call_date=datetime(2024, 6, 2),
        customer_name="Bob",
        created_by_id="founder",
    )
    # Update one to completed
    calls = manager.get_calls()
    manager.update_call(calls[0].id, {"status": "completed"})

    scheduled = manager.get_calls(status="scheduled")
    assert len(scheduled) == 1

    completed = manager.get_calls(status="completed")
    assert len(completed) == 1


def test_get_calls_filter_type(db, sample_users):
    manager = CustomerCallManager(db)
    manager.create_call(
        call_date=datetime(2024, 6, 1),
        customer_name="Jane",
        created_by_id="founder",
        call_type="discovery",
    )
    manager.create_call(
        call_date=datetime(2024, 6, 2),
        customer_name="Bob",
        created_by_id="founder",
        call_type="follow_up",
    )

    discovery = manager.get_calls(call_type="discovery")
    assert len(discovery) == 1
    assert discovery[0].customer_name == "Jane"


def test_get_calls_filter_user(db, sample_users):
    manager = CustomerCallManager(db)
    manager.create_call(
        call_date=datetime(2024, 6, 1),
        customer_name="Jane",
        created_by_id="founder",
    )
    manager.create_call(
        call_date=datetime(2024, 6, 2),
        customer_name="Bob",
        created_by_id="cofounder",
    )

    founder_calls = manager.get_calls(created_by_id="founder")
    assert len(founder_calls) == 1
    assert founder_calls[0].customer_name == "Jane"


def test_update_call(db, sample_users):
    manager = CustomerCallManager(db)
    call = manager.create_call(
        call_date=datetime(2024, 6, 1),
        customer_name="Jane",
        created_by_id="founder",
    )
    updated = manager.update_call(call.id, {
        "transcript": "Interviewer: Tell me about your workflow...",
        "status": "completed",
        "duration_minutes": 30,
    })
    assert updated.transcript == "Interviewer: Tell me about your workflow..."
    assert updated.status == "completed"
    assert updated.duration_minutes == 30


def test_update_call_not_found(db, sample_users):
    manager = CustomerCallManager(db)
    assert manager.update_call(999, {"status": "completed"}) is None


def test_get_recent_calls(db, sample_users):
    manager = CustomerCallManager(db)
    manager.create_call(
        call_date=datetime(2024, 1, 1),
        customer_name="Old Call",
        created_by_id="founder",
    )
    manager.create_call(
        call_date=datetime(2024, 6, 1),
        customer_name="New Call",
        created_by_id="founder",
    )

    recent = manager.get_recent_calls(limit=1)
    assert len(recent) == 1
    assert recent[0].customer_name == "New Call"


def test_update_call_with_analysis_fields(db, sample_users):
    manager = CustomerCallManager(db)
    call = manager.create_call(
        call_date=datetime(2024, 6, 1),
        customer_name="Jane",
        created_by_id="founder",
    )
    updated = manager.update_call(call.id, {
        "key_insights": [{"insight": "Hates manual reports", "importance": "high"}],
        "pain_points": [{"pain_point": "Manual reporting", "severity": "high"}],
        "sentiment": "negative",
        "wtp_signals": [{"signal": "Would pay $50/mo", "strength": "strong"}],
        "status": "analyzed",
    })
    assert updated.status == "analyzed"
    assert len(updated.key_insights) == 1
    assert updated.sentiment == "negative"
