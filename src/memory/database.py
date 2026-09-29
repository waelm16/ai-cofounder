"""SQLAlchemy models and database session configuration.

Defines all 15 ORM models that back the three memory layers (user-specific,
shared company, and collaboration), plus the chat/agent audit trail. Also
configures the engine, session factory, and provides ``init_db()`` /
``get_db()`` helpers.

Models:
    User, TimelineEvent, Decision, Hypothesis, ProblemStatement,
    CustomerCall, Metric, Conversation, DisagreementLog,
    CollaborationPattern, KnowledgeTransfer, Reflection, Task,
    ChatMessage, AgentConsultation.
"""

from datetime import datetime, date

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from src.config.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base class for all ORM models in the system."""

    pass


# Many-to-many link between timeline events and the decisions they relate to
timeline_decision_link = Table(
    "timeline_decision_link",
    Base.metadata,
    Column("timeline_event_id", Integer, ForeignKey("timeline.id"), primary_key=True),
    Column("decision_id", Integer, ForeignKey("decisions.id"), primary_key=True),
)


# ---------- User Management ----------


class User(Base):
    """A cofounder or team member in the system.

    Stores identity, role, and auto-enriched behavioural attributes (JSON
    columns) that the system learns over time from conversations.

    Columns:
        id: Unique string identifier (e.g. "founder").
        name: Display name.
        email: Unique email address.
        role / title: Organisational role and job title.
        communication_style: JSON — learned communication preferences.
        expertise_areas: JSON — domains of expertise.
        interests: JSON — topics of interest.
        working_patterns: JSON — scheduling / work-style signals.
        decision_style: JSON — how the user approaches decisions.
        created_at / last_active: Timestamps for account creation and
            most-recent activity.
    """

    __tablename__ = "users"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True)

    role = Column(String)
    title = Column(String)

    communication_style = Column(JSON)
    expertise_areas = Column(JSON)
    interests = Column(JSON)
    working_patterns = Column(JSON)
    decision_style = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, onupdate=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="user")
    tasks = relationship("Task", foreign_keys="Task.assigned_to_id", back_populates="assignee")


# ---------- Company Memory ----------


class TimelineEvent(Base):
    """A dated event in the shared company timeline.

    Records milestones, meetings, launches, pivots, and any other notable
    events. Supports multi-user attribution via ``participants`` (JSON list
    of user IDs) and links to related decisions through a many-to-many table.

    Columns:
        event_type: Free-form category (e.g. "milestone", "meeting", "pivot").
        context: JSON — arbitrary structured metadata about the event.
        participants: JSON list of user IDs who were involved.
    """

    __tablename__ = "timeline"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    event_type = Column(String)

    title = Column(String, nullable=False)
    description = Column(Text)
    context = Column(JSON)
    outcome = Column(Text)

    created_by_id = Column(String, ForeignKey("users.id"))
    participants = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    created_by = relationship("User")
    related_decisions = relationship("Decision", secondary=timeline_decision_link)


class Decision(Base):
    """A recorded business decision with per-user rationales.

    Tracks who proposed the decision, who participated, whether consensus
    was reached, and an optional revisit date for follow-up.

    Columns:
        decision_makers: JSON list of user IDs who participated.
        rationales: JSON dict mapping user IDs to their reasoning.
        consensus: Whether the team reached agreement.
        consensus_type: How consensus was reached (e.g. "unanimous", "compromise").
        revisit_date: Optional future date to re-evaluate the decision.
    """

    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)

    decision = Column(Text, nullable=False)
    category = Column(String)

    proposed_by_id = Column(String, ForeignKey("users.id"))
    decision_makers = Column(JSON)

    rationales = Column(JSON)

    consensus = Column(Boolean)
    consensus_type = Column(String)

    outcome = Column(Text)
    outcome_date = Column(DateTime)
    revisit_date = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    proposed_by = relationship("User")


class Hypothesis(Base):
    """A testable hypothesis with evidence tracking and validation scoring.

    Hypotheses can be created manually or auto-generated from customer call
    transcripts (``auto_generated=True``, ``source_call_id`` set). Evidence
    is accumulated as JSON lists of dicts (``evidence_for`` / ``evidence_against``)
    and a composite ``validation_score`` is computed downstream.

    Columns:
        validation_criteria: JSON list of dicts describing how to validate.
        result: Final status string (e.g. "validated", "invalidated") — None while active.
        auto_generated: True if the hypothesis was created by the transcript analyser.
        source_call_id: FK-like reference to the CustomerCall that generated it.
        evidence_for / evidence_against: JSON lists of evidence dicts.
        validation_score: Float 0-1 composite score; build signal at >= 0.70.
    """

    __tablename__ = "hypotheses"

    id = Column(Integer, primary_key=True)

    hypothesis = Column(Text, nullable=False)
    category = Column(String)

    owner_id = Column(String, ForeignKey("users.id"))
    collaborators = Column(JSON)

    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime)

    validation_criteria = Column(JSON)
    result = Column(String)
    result_data = Column(JSON)
    learnings = Column(Text)

    auto_generated = Column(Boolean, default=False)
    source_call_id = Column(Integer)
    evidence_for = Column(JSON)
    evidence_against = Column(JSON)
    validation_score = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    owner = relationship("User")


# ---------- Validation ----------


class ProblemStatement(Base):
    """A versioned problem-solution-customer statement.

    Only one statement is active (``is_active=True``) at any time. Creating
    a new version automatically deactivates the previous one. Version numbers
    are auto-incremented based on existing record count.

    Columns:
        problem: The customer problem being addressed.
        solution: The proposed solution.
        target_customer: Description of the ideal customer profile.
        version: Monotonically increasing version number.
        is_active: True for the current active statement; False for historical.
    """

    __tablename__ = "problem_statements"

    id = Column(Integer, primary_key=True)
    problem = Column(Text, nullable=False)
    solution = Column(Text, nullable=False)
    target_customer = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)
    created_by_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    created_by = relationship("User")


class CustomerCall(Base):
    """A customer discovery or follow-up call record.

    Tracks the full lifecycle: scheduling, Mom Test question generation,
    transcript ingestion, and LLM-powered analysis (insights, pain points,
    sentiment, willingness-to-pay signals). Analysis may also auto-generate
    hypotheses linked via ``source_call_id`` on the Hypothesis model.

    Columns:
        call_type: Category string — "discovery", "follow_up", "demo", etc.
        pre_call_questions: JSON list of Mom Test questions generated pre-call.
        transcript: Raw call transcript text (ingested post-call).
        key_insights / pain_points / wtp_signals: JSON — LLM-extracted analysis.
        sentiment: Overall call sentiment string.
        status: Lifecycle state — "scheduled", "completed", or "analyzed".
    """

    __tablename__ = "customer_calls"

    id = Column(Integer, primary_key=True)
    call_date = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer)

    customer_name = Column(String, nullable=False)
    customer_company = Column(String)
    customer_role = Column(String)
    customer_context = Column(JSON)

    call_type = Column(String)  # discovery, follow_up, demo, etc.
    pre_call_questions = Column(JSON)
    transcript = Column(Text)

    key_insights = Column(JSON)
    pain_points = Column(JSON)
    sentiment = Column(String)
    wtp_signals = Column(JSON)

    outcome = Column(Text)
    next_steps = Column(JSON)

    status = Column(String, default="scheduled")  # scheduled, completed, analyzed

    created_by_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    created_by = relationship("User")


class Metric(Base):
    """A dated snapshot of key business metrics.

    Each row captures financial, user, and growth KPIs for a given date.
    Additional ad-hoc metrics can be stored in the ``custom_metrics`` JSON
    column.

    Columns:
        mrr / arr: Monthly and Annual Recurring Revenue.
        burn_rate / runway_months: Cash-burn rate and remaining runway.
        total_users / active_users / paying_customers / churn_rate: User KPIs.
        user_growth_rate / revenue_growth_rate: Period-over-period growth rates.
        custom_metrics: JSON dict for any additional KPIs.
    """

    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)

    mrr = Column(Float)
    arr = Column(Float)
    burn_rate = Column(Float)
    runway_months = Column(Float)

    total_users = Column(Integer)
    active_users = Column(Integer)
    paying_customers = Column(Integer)
    churn_rate = Column(Float)

    user_growth_rate = Column(Float)
    revenue_growth_rate = Column(Float)

    custom_metrics = Column(JSON)

    updated_by_id = Column(String, ForeignKey("users.id"))

    created_at = Column(DateTime, default=datetime.utcnow)

    updated_by = relationship("User")


# ---------- Conversation Memory ----------


class Conversation(Base):
    """An archived conversation session between a user and the AI system.

    Stores the full message history as JSON along with extracted metadata
    (topics, insights, sentiment). The ``embedding`` column holds a binary
    vector for semantic search over past conversations via per-user FAISS
    indices.

    Columns:
        session_id: Unique identifier tying messages to a single session.
        messages: JSON list of message dicts (role + content).
        embedding: LargeBinary — serialised embedding vector for similarity search.
        mentioned_cofounder: True if the other cofounder was referenced.
        related_timeline_events / related_decisions: JSON lists of linked IDs.
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, nullable=False)

    messages = Column(JSON, nullable=False)

    topics_discussed = Column(JSON)
    key_insights = Column(JSON)
    sentiment = Column(String)

    mentioned_cofounder = Column(Boolean, default=False)
    related_timeline_events = Column(JSON)
    related_decisions = Column(JSON)

    embedding = Column(LargeBinary)

    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="conversations")


# ---------- Collaboration Tracking ----------


class DisagreementLog(Base):
    """A logged disagreement between cofounders on a specific topic.

    Captures each party's position and tracks resolution status so the
    team can learn from past conflicts.

    Columns:
        positions: JSON dict mapping user IDs to their stated position.
        resolution_status: e.g. "open", "resolved", "deferred".
        resolution_type: How it was resolved (e.g. "compromise", "data-driven").
    """

    __tablename__ = "disagreements"

    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    category = Column(String)

    date = Column(DateTime, nullable=False)
    description = Column(Text)

    positions = Column(JSON)

    resolution_status = Column(String)
    resolution_type = Column(String)
    resolution_description = Column(Text)
    resolution_date = Column(DateTime)

    outcome = Column(Text)
    learnings = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


class CollaborationPattern(Base):
    """An observed pattern in how cofounders collaborate.

    Used by the reflection system to surface recurring collaboration dynamics
    (positive or negative) and their effectiveness over time.

    Columns:
        pattern_type: Category of the pattern (e.g. "delegation", "pair-work").
        examples: JSON list of concrete instances where the pattern occurred.
        frequency / effectiveness: Qualitative assessments.
        first_observed / last_observed: When the pattern was first and last seen.
    """

    __tablename__ = "collaboration_patterns"

    id = Column(Integer, primary_key=True)

    pattern_type = Column(String)

    description = Column(Text)
    examples = Column(JSON)

    frequency = Column(String)
    effectiveness = Column(String)

    first_observed = Column(DateTime)
    last_observed = Column(DateTime)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


class KnowledgeTransfer(Base):
    """A record of knowledge shared from one cofounder to another.

    Tracks the topic, summary, what triggered the transfer, and the
    source conversation — enabling the system to surface what each
    cofounder has taught the other.

    Columns:
        trigger: What prompted the transfer (e.g. "question", "observation").
        source_conversation_id: FK to the Conversation where the transfer occurred.
        status: Lifecycle state (e.g. "pending", "acknowledged").
    """

    __tablename__ = "knowledge_transfers"

    id = Column(Integer, primary_key=True)

    from_user_id = Column(String, ForeignKey("users.id"))
    to_user_id = Column(String, ForeignKey("users.id"))

    topic = Column(String, nullable=False)
    summary = Column(Text, nullable=False)

    trigger = Column(String)
    source_conversation_id = Column(Integer, ForeignKey("conversations.id"))

    status = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)

    from_user = relationship("User", foreign_keys=[from_user_id])
    to_user = relationship("User", foreign_keys=[to_user_id])


# ---------- Reflection System ----------


class Reflection(Base):
    """A periodic retrospective generated by the reflection system.

    Covers a defined time period and summarises insights, patterns,
    highlights, challenges, and actionable suggestions for the team
    or a specific user.

    Columns:
        reflection_type: Scope — e.g. "individual", "team".
        period_type: Granularity — e.g. "weekly", "monthly".
        period_start / period_end: The date range covered.
        collaboration_analysis / individual_contributions: JSON breakdowns.
        suggestions / focus_areas: JSON lists of recommended actions.
    """

    __tablename__ = "reflections"

    id = Column(Integer, primary_key=True)

    reflection_type = Column(String)
    user_id = Column(String, ForeignKey("users.id"))

    period_type = Column(String)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)

    insights = Column(JSON)
    patterns = Column(JSON)
    highlights = Column(JSON)
    challenges = Column(JSON)

    collaboration_analysis = Column(JSON)
    individual_contributions = Column(JSON)

    suggestions = Column(JSON)
    focus_areas = Column(JSON)

    generated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


# ---------- Task Management ----------


class Task(Base):
    """A task assigned to a cofounder, optionally linked to a conversation.

    Supports priority, time estimates, and completion tracking. Tasks can
    be created manually or auto-generated from conversations (``source``
    and ``source_conversation_id``).

    Columns:
        status: Lifecycle state — e.g. "todo", "in_progress", "done".
        priority: Urgency level — e.g. "high", "medium", "low".
        source: How the task was created — e.g. "conversation", "manual".
        source_conversation_id: FK to the Conversation that spawned the task.
    """

    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)

    title = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String)

    assigned_to_id = Column(String, ForeignKey("users.id"))
    created_by_id = Column(String, ForeignKey("users.id"))

    due_date = Column(DateTime)
    estimated_hours = Column(Float)
    actual_hours = Column(Float)

    status = Column(String)
    priority = Column(String)

    completion_date = Column(DateTime)
    completion_notes = Column(Text)

    source = Column(String)
    source_conversation_id = Column(Integer, ForeignKey("conversations.id"))

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    assignee = relationship("User", foreign_keys=[assigned_to_id], back_populates="tasks")
    creator = relationship("User", foreign_keys=[created_by_id])


# ---------- Chat & Agent Consultations ----------


class ChatMessage(Base):
    """A single message in a multi-agent chat session.

    Provides an audit trail of every user query and assistant response,
    including which specialist agents were consulted and what knowledge
    sources were cited.

    Columns:
        session_id: Groups messages belonging to the same chat session (indexed).
        role: "user" or "assistant".
        agents_consulted: JSON list of agent IDs that contributed to the response.
        sources_cited: JSON list of knowledge-source references.
    """

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    agents_consulted = Column(JSON)
    sources_cited = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


class AgentConsultation(Base):
    """A record of a single specialist agent's contribution to a chat response.

    Each time the Cofounder orchestrator routes a sub-question to a specialist
    agent, the agent's response is stored here — linking back to the parent
    ChatMessage for full audit traceability.

    Columns:
        agent_id / agent_name: Which specialist agent responded.
        sub_question: The targeted question the classifier generated for this agent.
        confidence: Self-reported confidence level of the response.
        sources: JSON list of knowledge sources the agent cited.
    """

    __tablename__ = "agent_consultations"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    chat_message_id = Column(Integer, ForeignKey("chat_messages.id"))
    agent_id = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)
    sub_question = Column(Text)
    response = Column(Text, nullable=False)
    confidence = Column(String)
    sources = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    chat_message = relationship("ChatMessage")


# ---------- Engine & Session ----------

settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables defined by Base.metadata if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a SQLAlchemy session and ensure it is closed after use.

    Designed for use as a FastAPI dependency via ``Depends(get_db)``.

    Yields:
        Session: An active SQLAlchemy session bound to the configured engine.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
