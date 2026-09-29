"""Hypothesis validation API routes.

Manages problem statements (versioned problem-solution-customer triples)
and hypothesis CRUD with evidence tracking — the core of the customer
discovery validation loop.

Endpoints:
    POST /api/validation/problem                        — create a problem statement
    GET  /api/validation/problem                        — get the active problem statement
    GET  /api/validation/problem/history                — list all versions
    PUT  /api/validation/problem/{problem_id}           — update a problem statement

    POST /api/validation/hypotheses                     — create a hypothesis
    GET  /api/validation/hypotheses/{hypothesis_id}     — get a hypothesis
    POST /api/validation/hypotheses/{id}/evidence       — append evidence
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.auth import get_current_user
from src.api.schemas import (
    EvidenceAdd,
    HypothesisCreate,
    HypothesisResponse,
    ProblemStatementCreate,
    ProblemStatementResponse,
    ProblemStatementUpdate,
)
from src.memory.database import User, get_db
from src.memory.hypotheses_tracker import HypothesesTracker
from src.memory.problem_statement import ProblemStatementManager

router = APIRouter(prefix="/api/validation", tags=["validation"])


# ---------- Problem Statements ----------


@router.post("/problem", response_model=ProblemStatementResponse, status_code=201)
def create_problem_statement(
    body: ProblemStatementCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new problem statement (problem / solution / target customer).

    Each new statement becomes the active version; previous statements
    are preserved in the version history.

    Args:
        body: The problem-solution-customer triple.

    Returns:
        ProblemStatementResponse: The newly created problem statement.
    """
    manager = ProblemStatementManager(db)
    result = manager.create_problem_statement(
        problem=body.problem,
        solution=body.solution,
        target_customer=body.target_customer,
        created_by_id=current_user.id,
    )
    return result


@router.get("/problem", response_model=ProblemStatementResponse)
def get_active_problem_statement(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the currently active problem statement.

    Returns:
        ProblemStatementResponse: The active problem statement.

    Raises:
        HTTPException: 404 if no active problem statement exists.
    """
    manager = ProblemStatementManager(db)
    result = manager.get_active()
    if not result:
        raise HTTPException(status_code=404, detail="No active problem statement")
    return result


@router.get("/problem/history", response_model=list[ProblemStatementResponse])
def get_problem_statement_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all historical versions of the problem statement.

    Returns:
        list[ProblemStatementResponse]: All versions, newest first.
    """
    manager = ProblemStatementManager(db)
    return manager.get_all_versions()


@router.put("/problem/{problem_id}", response_model=ProblemStatementResponse)
def update_problem_statement(
    problem_id: int,
    updates: ProblemStatementUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update a problem statement.

    Args:
        problem_id: Primary key of the problem statement.
        updates: Fields to update.

    Returns:
        ProblemStatementResponse: The updated problem statement.

    Raises:
        HTTPException: 404 if the problem statement does not exist.
    """
    manager = ProblemStatementManager(db)
    update_data = updates.model_dump(exclude_unset=True)
    result = manager.update_problem_statement(problem_id, update_data)
    if not result:
        raise HTTPException(status_code=404, detail="Problem statement not found")
    return result


# ---------- Hypotheses ----------


@router.post("/hypotheses", response_model=HypothesisResponse, status_code=201)
def create_hypothesis(
    body: HypothesisCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new hypothesis for validation.

    Hypotheses created through this endpoint are marked as manually
    created (``auto_generated=False``).  Initial evidence can be
    supplied via ``evidence_for`` / ``evidence_against``.

    Args:
        body: Hypothesis text, category, criteria, and optional evidence.

    Returns:
        HypothesisResponse: The newly created hypothesis.
    """
    tracker = HypothesesTracker(db)
    result = tracker.add_hypothesis(
        hypothesis=body.hypothesis,
        owner_id=current_user.id,
        category=body.category,
        validation_criteria=body.validation_criteria,
        start_date=body.start_date,
        collaborators=body.collaborators,
        auto_generated=False,
        evidence_for=body.evidence_for,
        evidence_against=body.evidence_against,
    )
    return result


@router.get("/hypotheses/{hypothesis_id}", response_model=HypothesisResponse)
def get_hypothesis(
    hypothesis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch a single hypothesis by ID.

    Args:
        hypothesis_id: Primary key of the hypothesis.

    Returns:
        HypothesisResponse: The matching hypothesis.

    Raises:
        HTTPException: 404 if the hypothesis does not exist.
    """
    tracker = HypothesesTracker(db)
    result = tracker.get_hypothesis(hypothesis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return result


@router.post("/hypotheses/{hypothesis_id}/evidence", response_model=HypothesisResponse)
def add_evidence(
    hypothesis_id: int,
    body: EvidenceAdd,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Append a piece of supporting or opposing evidence to a hypothesis.

    Args:
        hypothesis_id: Primary key of the hypothesis.
        body: Evidence dict and its type (``"for"`` or ``"against"``).

    Returns:
        HypothesisResponse: The hypothesis with updated evidence lists.

    Raises:
        HTTPException: 404 if the hypothesis does not exist.
    """
    tracker = HypothesesTracker(db)
    result = tracker.add_evidence(
        hypothesis_id=hypothesis_id,
        evidence_type=body.evidence_type,
        evidence=body.evidence,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Hypothesis not found")
    return result
