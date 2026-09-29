"""Customer call intelligence API routes.

CRUD for customer calls plus two LLM-powered endpoints:
- Pre-call question generation (Mom Test methodology via RAG + Sonnet)
- Post-call transcript analysis (Haiku extraction with auto-hypothesis
  generation and evidence mapping)

Endpoints:
    POST /api/calls/                       — schedule/record a call
    GET  /api/calls/                       — list calls (filterable)
    GET  /api/calls/{call_id}              — get a single call
    PUT  /api/calls/{call_id}              — update a call
    POST /api/calls/{call_id}/questions    — generate pre-call questions
    POST /api/calls/{call_id}/analyze      — analyse a call transcript
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.auth import get_current_user
from src.api.schemas import (
    CallAnalysisResponse,
    CustomerCallCreate,
    CustomerCallResponse,
    CustomerCallUpdate,
    PreCallQuestionsResponse,
)
from src.intelligence.question_generator import generate_pre_call_questions
from src.intelligence.transcript_analyzer import analyze_transcript
from src.memory.customer_calls import CustomerCallManager
from src.memory.database import User, get_db
from src.memory.hypotheses_tracker import HypothesesTracker

router = APIRouter(prefix="/api/calls", tags=["calls"])


@router.post("/", response_model=CustomerCallResponse, status_code=201)
def create_call(
    body: CustomerCallCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Schedule or record a new customer call.

    Args:
        body: Call metadata (date, customer info, type).

    Returns:
        CustomerCallResponse: The newly created call record.
    """
    manager = CustomerCallManager(db)
    result = manager.create_call(
        call_date=body.call_date,
        customer_name=body.customer_name,
        created_by_id=current_user.id,
        duration_minutes=body.duration_minutes,
        customer_company=body.customer_company,
        customer_role=body.customer_role,
        customer_context=body.customer_context,
        call_type=body.call_type,
    )
    return result


@router.get("/", response_model=list[CustomerCallResponse])
def list_calls(
    status: str | None = Query(None),
    call_type: str | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List customer calls, optionally filtered by status and/or call type.

    Args:
        status: Filter by call status (e.g. ``"scheduled"``, ``"analyzed"``).
        call_type: Filter by call type (e.g. ``"discovery"``, ``"demo"``).

    Returns:
        list[CustomerCallResponse]: Matching call records.
    """
    manager = CustomerCallManager(db)
    return manager.get_calls(status=status, call_type=call_type)


@router.get("/{call_id}", response_model=CustomerCallResponse)
def get_call(
    call_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch a single customer call by ID.

    Args:
        call_id: Primary key of the call.

    Returns:
        CustomerCallResponse: The matching call record.

    Raises:
        HTTPException: 404 if the call does not exist.
    """
    manager = CustomerCallManager(db)
    result = manager.get_call(call_id)
    if not result:
        raise HTTPException(status_code=404, detail="Call not found")
    return result


@router.put("/{call_id}", response_model=CustomerCallResponse)
def update_call(
    call_id: int,
    body: CustomerCallUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update a customer call (e.g. attach a transcript or set outcome).

    Args:
        call_id: Primary key of the call.
        body: Fields to update.

    Returns:
        CustomerCallResponse: The updated call record.

    Raises:
        HTTPException: 404 if the call does not exist.
    """
    manager = CustomerCallManager(db)
    update_data = body.model_dump(exclude_unset=True)
    result = manager.update_call(call_id, update_data)
    if not result:
        raise HTTPException(status_code=404, detail="Call not found")
    return result


@router.post("/{call_id}/questions", response_model=PreCallQuestionsResponse)
def generate_questions(
    call_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate Mom-Test-style pre-call questions for a scheduled call.

    Uses the active problem statement and hypotheses as context so the
    generated questions target current validation needs.  Questions are
    persisted on the call record for later reference.

    Args:
        call_id: Primary key of the call to generate questions for.

    Returns:
        PreCallQuestionsResponse: The call ID and generated questions list.

    Raises:
        HTTPException: 404 if the call does not exist.
    """
    call_manager = CustomerCallManager(db)
    call = call_manager.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    hyp_tracker = HypothesesTracker(db)
    active = hyp_tracker.get_active_hypotheses()
    active_hyps = [
        {"id": h.id, "hypothesis": h.hypothesis, "category": h.category}
        for h in active
    ]

    from src.memory.problem_statement import ProblemStatementManager

    ps_manager = ProblemStatementManager(db)
    ps = ps_manager.get_active()
    ps_dict = None
    if ps:
        ps_dict = {
            "problem": ps.problem,
            "solution": ps.solution,
            "target_customer": ps.target_customer,
        }

    questions = generate_pre_call_questions(
        customer_name=call.customer_name,
        customer_company=call.customer_company,
        customer_role=call.customer_role,
        customer_context=call.customer_context,
        active_hypotheses=active_hyps,
        problem_statement=ps_dict,
    )

    call_manager.update_call(call_id, {"pre_call_questions": questions})

    return PreCallQuestionsResponse(call_id=call_id, questions=questions)


@router.post("/{call_id}/analyze", response_model=CallAnalysisResponse)
def analyze_call(
    call_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyse a call transcript and extract structured intelligence.

    Performs three actions:
    1. Extracts insights, pain points, sentiment, WTP signals, and next steps.
    2. Auto-generates new hypotheses from the transcript.
    3. Maps supporting/opposing evidence to existing active hypotheses.

    All results are persisted to the call record and the hypotheses tracker.

    Args:
        call_id: Primary key of the call whose transcript to analyse.

    Returns:
        CallAnalysisResponse: Extracted insights, new hypotheses, and
            evidence mappings.

    Raises:
        HTTPException: 404 if the call does not exist, 400 if no transcript
            has been attached yet.
    """
    call_manager = CustomerCallManager(db)
    call = call_manager.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    if not call.transcript:
        raise HTTPException(status_code=400, detail="Call has no transcript to analyze")

    hyp_tracker = HypothesesTracker(db)
    active = hyp_tracker.get_active_hypotheses()
    active_hyps = [
        {"id": h.id, "hypothesis": h.hypothesis, "category": h.category}
        for h in active
    ]

    analysis = analyze_transcript(
        transcript=call.transcript,
        customer_name=call.customer_name,
        customer_company=call.customer_company,
        active_hypotheses=active_hyps,
    )

    call_manager.update_call(call_id, {
        "key_insights": analysis.get("key_insights", []),
        "pain_points": analysis.get("pain_points", []),
        "sentiment": analysis.get("sentiment", "neutral"),
        "wtp_signals": analysis.get("wtp_signals", []),
        "next_steps": analysis.get("next_steps", []),
        "status": "analyzed",
    })

    # Create a hypothesis for each new hypothesis the LLM identified
    new_hypotheses = []
    for hyp_data in analysis.get("new_hypotheses", []):
        h = hyp_tracker.add_hypothesis(
            hypothesis=hyp_data.get("hypothesis", ""),
            owner_id=current_user.id,
            category=hyp_data.get("category"),
            auto_generated=True,
            source_call_id=call_id,
            evidence_for=[{"quote": hyp_data.get("evidence", ""), "source": f"Call #{call_id} with {call.customer_name}"}],
        )
        new_hypotheses.append({"id": h.id, "hypothesis": h.hypothesis, "category": h.category})

    # Attach evidence from this call to existing hypotheses
    evidence_mapped = []
    for ev in analysis.get("evidence_for_existing", []):
        hyp_id = ev.get("hypothesis_id")
        ev_type = ev.get("evidence_type", "for")
        if hyp_id and ev_type in ("for", "against"):
            result = hyp_tracker.add_evidence(
                hypothesis_id=hyp_id,
                evidence_type=ev_type,
                evidence={
                    "quote": ev.get("quote", ""),
                    "source": f"Call #{call_id} with {call.customer_name}",
                },
            )
            if result:
                evidence_mapped.append(ev)

    return CallAnalysisResponse(
        call_id=call_id,
        key_insights=analysis.get("key_insights", []),
        pain_points=analysis.get("pain_points", []),
        sentiment=analysis.get("sentiment", "neutral"),
        wtp_signals=analysis.get("wtp_signals", []),
        next_steps=analysis.get("next_steps", []),
        new_hypotheses=new_hypotheses,
        evidence_mapped=evidence_mapped,
    )
