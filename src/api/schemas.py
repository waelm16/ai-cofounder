"""Pydantic request and response schemas for all API endpoints.

Organised by domain: Users, Timeline Events, Decisions, Hypotheses,
Problem Statements, Metrics, Customer Calls, Chat, Knowledge Search,
and a generic PaginatedResponse wrapper.
"""

from datetime import date, datetime
from typing import Any

from typing import Literal

from pydantic import BaseModel, Field


# ---------- Users ----------


class UserCreate(BaseModel):
    """Request body for creating a new user profile."""

    id: str
    name: str
    email: str | None = None
    role: str | None = None
    title: str | None = None
    communication_style: dict | None = None
    expertise_areas: list[str] | None = None
    interests: list[str] | None = None
    working_patterns: dict | None = None
    decision_style: dict | None = None


class UserUpdate(BaseModel):
    """Request body for partially updating an existing user profile."""

    name: str | None = None
    email: str | None = None
    role: str | None = None
    title: str | None = None
    communication_style: dict | None = None
    expertise_areas: list[str] | None = None
    interests: list[str] | None = None
    working_patterns: dict | None = None
    decision_style: dict | None = None


class UserResponse(BaseModel):
    """Response model for user profile data, including DB timestamps."""

    id: str
    name: str
    email: str | None = None
    role: str | None = None
    title: str | None = None
    communication_style: dict | None = None
    expertise_areas: list[str] | None = None
    interests: list[str] | None = None
    working_patterns: dict | None = None
    decision_style: dict | None = None
    created_at: datetime | None = None
    last_active: datetime | None = None

    model_config = {"from_attributes": True}


# ---------- Timeline Events ----------


class TimelineEventCreate(BaseModel):
    """Request body for adding a new event to the company timeline."""

    title: str
    event_type: str | None = None
    date: datetime
    description: str | None = None
    context: dict | None = None
    participants: list[str] | None = None


class TimelineEventUpdate(BaseModel):
    """Request body for partially updating an existing timeline event."""

    title: str | None = None
    event_type: str | None = None
    date: datetime | None = None
    description: str | None = None
    context: dict | None = None
    outcome: str | None = None
    participants: list[str] | None = None


class TimelineEventResponse(BaseModel):
    """Response model for a single timeline event with audit fields."""

    id: int
    title: str
    event_type: str | None = None
    date: datetime
    description: str | None = None
    context: dict | None = None
    outcome: str | None = None
    created_by_id: str | None = None
    participants: list[str] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# ---------- Decisions ----------


class DecisionCreate(BaseModel):
    """Request body for recording a new company decision.

    The ``rationales`` dict maps user IDs to their individual rationales,
    supporting multi-user decision tracking.
    """

    decision: str
    category: str | None = None
    date: datetime | None = None
    decision_makers: list[str] | None = None
    rationales: dict | None = None
    consensus: bool | None = None
    consensus_type: str | None = None


class DecisionUpdate(BaseModel):
    """Request body for partially updating a decision record."""

    decision: str | None = None
    category: str | None = None
    outcome: str | None = None
    outcome_date: datetime | None = None
    revisit_date: datetime | None = None
    rationales: dict | None = None
    consensus: bool | None = None
    consensus_type: str | None = None


class DecisionResponse(BaseModel):
    """Response model for a decision record with proposer and audit fields."""

    id: int
    decision: str
    category: str | None = None
    date: datetime
    proposed_by_id: str | None = None
    decision_makers: list[str] | None = None
    rationales: dict | None = None
    consensus: bool | None = None
    consensus_type: str | None = None
    outcome: str | None = None
    outcome_date: datetime | None = None
    revisit_date: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# ---------- Hypotheses ----------


class HypothesisCreate(BaseModel):
    """Request body for creating a new hypothesis under validation."""

    hypothesis: str
    category: str | None = None
    collaborators: list[str] | None = None
    start_date: datetime | None = None
    validation_criteria: list[dict] | None = None
    auto_generated: bool | None = None
    evidence_for: list[dict] | None = None
    evidence_against: list[dict] | None = None


class HypothesisUpdate(BaseModel):
    """Request body for partially updating a hypothesis."""

    hypothesis: str | None = None
    category: str | None = None
    end_date: datetime | None = None
    result: str | None = None
    result_data: dict | None = None
    learnings: str | None = None
    collaborators: list[str] | None = None
    validation_criteria: list[dict] | None = None
    evidence_for: list[dict] | None = None
    evidence_against: list[dict] | None = None
    validation_score: float | None = None
    auto_generated: bool | None = None


class HypothesisResponse(BaseModel):
    """Response model for a hypothesis with evidence tracking and validation score."""

    id: int
    hypothesis: str
    category: str | None = None
    owner_id: str | None = None
    collaborators: list[str] | None = None
    start_date: datetime
    end_date: datetime | None = None
    validation_criteria: list[dict] | None = None
    result: str | None = None
    result_data: dict | None = None
    learnings: str | None = None
    auto_generated: bool | None = None
    source_call_id: int | None = None
    evidence_for: list[dict] | None = None
    evidence_against: list[dict] | None = None
    validation_score: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class EvidenceAdd(BaseModel):
    """Request body for appending evidence to a hypothesis.

    ``evidence_type`` must be ``"for"`` or ``"against"``.
    """

    evidence_type: Literal["for", "against"]
    evidence: dict


# ---------- Problem Statements ----------


class ProblemStatementCreate(BaseModel):
    """Request body for defining a new problem-solution-target-customer triple."""

    problem: str
    solution: str
    target_customer: str


class ProblemStatementUpdate(BaseModel):
    """Request body for partially updating a problem statement."""

    problem: str | None = None
    solution: str | None = None
    target_customer: str | None = None


class ProblemStatementResponse(BaseModel):
    """Response model for a versioned problem statement."""

    id: int
    problem: str
    solution: str
    target_customer: str
    version: int
    is_active: bool
    created_by_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


# ---------- Metrics ----------


class MetricCreate(BaseModel):
    """Request body for recording a daily metrics snapshot (MRR, ARR, churn, etc.)."""

    date: date
    mrr: float | None = None
    arr: float | None = None
    burn_rate: float | None = None
    runway_months: float | None = None
    total_users: int | None = None
    active_users: int | None = None
    paying_customers: int | None = None
    churn_rate: float | None = None
    user_growth_rate: float | None = None
    revenue_growth_rate: float | None = None
    custom_metrics: dict | None = None


class MetricResponse(BaseModel):
    """Response model for a daily metrics snapshot."""

    id: int
    date: date
    mrr: float | None = None
    arr: float | None = None
    burn_rate: float | None = None
    runway_months: float | None = None
    total_users: int | None = None
    active_users: int | None = None
    paying_customers: int | None = None
    churn_rate: float | None = None
    user_growth_rate: float | None = None
    revenue_growth_rate: float | None = None
    custom_metrics: dict | None = None
    updated_by_id: str | None = None
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


# ---------- Customer Calls ----------


class CustomerCallCreate(BaseModel):
    """Request body for scheduling / recording a new customer call."""

    call_date: datetime
    duration_minutes: int | None = None
    customer_name: str
    customer_company: str | None = None
    customer_role: str | None = None
    customer_context: dict | None = None
    call_type: str | None = None


class CustomerCallUpdate(BaseModel):
    """Request body for partially updating a customer call (e.g. adding a transcript)."""

    call_date: datetime | None = None
    duration_minutes: int | None = None
    customer_name: str | None = None
    customer_company: str | None = None
    customer_role: str | None = None
    customer_context: dict | None = None
    call_type: str | None = None
    transcript: str | None = None
    outcome: str | None = None
    next_steps: list[dict] | None = None
    status: str | None = None


class CustomerCallResponse(BaseModel):
    """Full response model for a customer call including LLM-generated fields."""

    id: int
    call_date: datetime
    duration_minutes: int | None = None
    customer_name: str
    customer_company: str | None = None
    customer_role: str | None = None
    customer_context: dict | None = None
    call_type: str | None = None
    pre_call_questions: list[dict] | None = None
    transcript: str | None = None
    key_insights: list[dict] | None = None
    pain_points: list[dict] | None = None
    sentiment: str | None = None
    wtp_signals: list[dict] | None = None
    outcome: str | None = None
    next_steps: list[dict] | None = None
    status: str | None = None
    created_by_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class PreCallQuestionsResponse(BaseModel):
    """Response model for Mom-Test-style pre-call questions generated by the LLM."""

    call_id: int
    questions: list[dict]


class CallAnalysisResponse(BaseModel):
    """Response model for post-call transcript analysis.

    Includes extracted insights, auto-generated hypotheses, and evidence
    mapped back to existing hypotheses.
    """

    call_id: int
    key_insights: list[dict]
    pain_points: list[dict]
    sentiment: str
    wtp_signals: list[dict]
    next_steps: list[dict]
    new_hypotheses: list[dict]
    evidence_mapped: list[dict]


# ---------- Chat ----------


class ChatRequest(BaseModel):
    """Request body for the multi-agent chat endpoint.

    If ``session_id`` is omitted a new session is created automatically.
    """

    message: str = Field(..., min_length=1)
    session_id: str | None = None


class AgentOutputResponse(BaseModel):
    """Individual specialist agent's response within a chat turn."""

    agent_id: str
    agent_name: str
    response: str
    sources: list[dict] = []
    confidence: str | None = None


class ChatResponse(BaseModel):
    """Response from the multi-agent cofounder graph.

    Contains the synthesised answer, which agents were consulted, cited
    sources, individual agent outputs, and any learnings extracted.
    """

    response: str
    session_id: str
    agents_consulted: list[str] = []
    sources: list[dict] = []
    agent_outputs: list[AgentOutputResponse] = []
    learnings_extracted: dict | None = None


class AgentConsultationResponse(BaseModel):
    """Detail of a single specialist agent's contribution to a chat response."""

    agent_id: str
    agent_name: str
    sub_question: str | None = None
    response: str
    confidence: str | None = None
    sources: list[dict] = []
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    """Conversation history for a chat session."""

    session_id: str
    messages: list[dict]
    agents_consulted: list[str] = []


class ChatSessionSummary(BaseModel):
    """Summary of a chat session for the session list view."""

    session_id: str
    first_message: str
    message_count: int
    agents_consulted: list[str] = []
    created_at: str | None = None


# ---------- Knowledge Search ----------


class KnowledgeSearchResult(BaseModel):
    """A single RAG retrieval result with metadata for citation."""

    content: str
    source_type: str | None = None
    source_name: str | None = None
    score: float
    chunk_index: int | None = None
    page_number: int | None = None
    title: str | None = None
    author: str | None = None
    date: str | None = None
    chapter: str | None = None
    url: str | None = None
    customer_name: str | None = None
    customer_company: str | None = None


class KnowledgeSearchResponse(BaseModel):
    """Response model for a knowledge-base search, including formatted citations."""

    query: str
    results: list[KnowledgeSearchResult]
    formatted: str


# ---------- Pagination ----------


class PaginatedResponse(BaseModel):
    """Generic paginated list wrapper used across multiple endpoints."""

    items: list[Any]
    total: int
    page: int = 1
    page_size: int = 50
