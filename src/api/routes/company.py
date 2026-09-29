"""Shared company memory API routes.

Manages the four pillars of company-wide state that all cofounders share:
timeline events, decisions (with per-user rationales), hypotheses, and metrics.

Endpoints:
    POST /api/company/timeline               — add a timeline event
    GET  /api/company/timeline               — list events (filterable)
    GET  /api/company/timeline/{event_id}    — get a single event
    PUT  /api/company/timeline/{event_id}    — update an event

    POST /api/company/decisions              — record a decision
    GET  /api/company/decisions              — list decisions (filterable)
    PUT  /api/company/decisions/{id}         — update a decision

    POST /api/company/hypotheses             — create a hypothesis
    GET  /api/company/hypotheses             — list hypotheses (filterable)
    PUT  /api/company/hypotheses/{id}        — update a hypothesis

    POST /api/company/metrics                — record a metrics snapshot
    GET  /api/company/metrics                — list metrics (date-filterable)
"""

from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.auth import get_current_user
from src.api.schemas import (
    DecisionCreate,
    DecisionResponse,
    DecisionUpdate,
    HypothesisCreate,
    HypothesisResponse,
    HypothesisUpdate,
    MetricCreate,
    MetricResponse,
    TimelineEventCreate,
    TimelineEventResponse,
    TimelineEventUpdate,
)
from src.memory.company_timeline import TimelineManager
from src.memory.database import Metric, User, get_db
from src.memory.decisions_tracker import DecisionTracker
from src.memory.hypotheses_tracker import HypothesesTracker

router = APIRouter(prefix="/api/company", tags=["company"])


# ---------- Timeline ----------


@router.post("/timeline", response_model=TimelineEventResponse, status_code=201)
def create_timeline_event(
    event: TimelineEventCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a new event to the shared company timeline.

    Args:
        event: Event data including title, date, and optional participants.

    Returns:
        TimelineEventResponse: The newly created event.
    """
    manager = TimelineManager(db)
    result = manager.add_event(
        title=event.title,
        event_type=event.event_type,
        date=event.date,
        created_by=current_user.id,
        participants=event.participants,
        description=event.description,
        context=event.context,
    )
    return result


@router.get("/timeline", response_model=list[TimelineEventResponse])
def list_timeline_events(
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    event_type: str | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List timeline events with optional date-range and type filters.

    Args:
        start_date: Include events on or after this datetime.
        end_date: Include events on or before this datetime.
        event_type: Filter to a specific event type string.

    Returns:
        list[TimelineEventResponse]: Matching events.
    """
    manager = TimelineManager(db)
    return manager.get_events(start_date=start_date, end_date=end_date, event_type=event_type)


@router.get("/timeline/{event_id}", response_model=TimelineEventResponse)
def get_timeline_event(
    event_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch a single timeline event by its ID.

    Args:
        event_id: Primary key of the event.

    Returns:
        TimelineEventResponse: The matching event.

    Raises:
        HTTPException: 404 if the event does not exist.
    """
    manager = TimelineManager(db)
    event = manager.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Timeline event not found")
    return event


@router.put("/timeline/{event_id}", response_model=TimelineEventResponse)
def update_timeline_event(
    event_id: int,
    updates: TimelineEventUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update a timeline event.

    Args:
        event_id: Primary key of the event.
        updates: Fields to update.

    Returns:
        TimelineEventResponse: The updated event.

    Raises:
        HTTPException: 404 if the event does not exist.
    """
    manager = TimelineManager(db)
    update_data = updates.model_dump(exclude_unset=True)
    event = manager.update_event(event_id, update_data)
    if not event:
        raise HTTPException(status_code=404, detail="Timeline event not found")
    return event


# ---------- Decisions ----------


@router.post("/decisions", response_model=DecisionResponse, status_code=201)
def create_decision(
    decision: DecisionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record a new company decision.

    The authenticated user is automatically set as the proposer.

    Args:
        decision: Decision data including text, category, and rationales.

    Returns:
        DecisionResponse: The newly created decision record.
    """
    tracker = DecisionTracker(db)
    result = tracker.add_decision(
        decision=decision.decision,
        category=decision.category,
        proposed_by=current_user.id,
        date=decision.date,
        decision_makers=decision.decision_makers,
        rationales=decision.rationales,
        consensus=decision.consensus,
        consensus_type=decision.consensus_type,
    )
    return result


@router.get("/decisions", response_model=list[DecisionResponse])
def list_decisions(
    category: str | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List decisions, optionally filtered by category.

    Args:
        category: Optional category to filter by.

    Returns:
        list[DecisionResponse]: Matching decisions.
    """
    tracker = DecisionTracker(db)
    return tracker.get_decisions(category=category)


@router.put("/decisions/{decision_id}", response_model=DecisionResponse)
def update_decision(
    decision_id: int,
    updates: DecisionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update a decision record (e.g. add outcome or revisit date).

    Args:
        decision_id: Primary key of the decision.
        updates: Fields to update.

    Returns:
        DecisionResponse: The updated decision.

    Raises:
        HTTPException: 404 if the decision does not exist.
    """
    tracker = DecisionTracker(db)
    update_data = updates.model_dump(exclude_unset=True)
    result = tracker.update_decision(decision_id, update_data)
    if not result:
        raise HTTPException(status_code=404, detail="Decision not found")
    return result


# ---------- Hypotheses ----------


@router.post("/hypotheses", response_model=HypothesisResponse, status_code=201)
def create_hypothesis(
    hypothesis: HypothesisCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new hypothesis for validation.

    The authenticated user is set as the owner.

    Args:
        hypothesis: Hypothesis data including text, category, and criteria.

    Returns:
        HypothesisResponse: The newly created hypothesis.
    """
    tracker = HypothesesTracker(db)
    result = tracker.add_hypothesis(
        hypothesis=hypothesis.hypothesis,
        category=hypothesis.category,
        owner_id=current_user.id,
        validation_criteria=hypothesis.validation_criteria,
        start_date=hypothesis.start_date,
        collaborators=hypothesis.collaborators,
    )
    return result


@router.get("/hypotheses", response_model=list[HypothesisResponse])
def list_hypotheses(
    category: str | None = Query(None),
    status: str | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List hypotheses, optionally filtered by category and/or status.

    Args:
        category: Optional category filter.
        status: Optional status filter (e.g. ``"active"``, ``"validated"``).

    Returns:
        list[HypothesisResponse]: Matching hypotheses.
    """
    tracker = HypothesesTracker(db)
    return tracker.get_hypotheses(category=category, status=status)


@router.put("/hypotheses/{hypothesis_id}", response_model=HypothesisResponse)
def update_hypothesis(
    hypothesis_id: int,
    updates: HypothesisUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update a hypothesis (e.g. add result, learnings, or score).

    Args:
        hypothesis_id: Primary key of the hypothesis.
        updates: Fields to update.

    Returns:
        HypothesisResponse: The updated hypothesis.

    Raises:
        HTTPException: 404 if the hypothesis does not exist.
    """
    tracker = HypothesesTracker(db)
    update_data = updates.model_dump(exclude_unset=True)
    result = tracker.update_hypothesis(hypothesis_id, update_data)
    if not result:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return result


# ---------- Metrics ----------


@router.post("/metrics", response_model=MetricResponse, status_code=201)
def create_metric(
    metric: MetricCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Record a daily metrics snapshot (MRR, ARR, churn, runway, etc.).

    Args:
        metric: Metric values for a specific date.

    Returns:
        MetricResponse: The persisted metric record.
    """
    record = Metric(
        **metric.model_dump(exclude_unset=True),
        updated_by_id=current_user.id,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/metrics", response_model=list[MetricResponse])
def list_metrics(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List metrics snapshots, optionally filtered by date range.

    Results are ordered most-recent first.

    Args:
        start_date: Include metrics on or after this date.
        end_date: Include metrics on or before this date.

    Returns:
        list[MetricResponse]: Matching metric records.
    """
    query = db.query(Metric)
    if start_date:
        query = query.filter(Metric.date >= start_date)
    if end_date:
        query = query.filter(Metric.date <= end_date)
    return query.order_by(Metric.date.desc()).all()
