"""Context loader node -- first node in the graph, hydrates state with DB context.

Loads the authenticated user's profile, active hypotheses, current problem statement,
recent decisions, timeline events, and latest metrics from the database. Combines
the static ``COMPANY_CONTEXT`` with live business data to produce the
``company_context`` string that all downstream nodes consume.

This node manages its own DB session when none is injected (production path),
and accepts an optional ``db`` parameter for testing.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from src.agent.prompts.company_context import COMPANY_CONTEXT
from src.agent.state import CofounderState
from src.memory.database import Metric, get_db


def _serialize(obj) -> dict:
    """Serialize a SQLAlchemy model instance to a plain dict.

    Args:
        obj: A SQLAlchemy ORM model instance.

    Returns:
        Dict with column names as keys; datetime values converted to ISO strings.
    """
    result = {}
    for c in obj.__table__.columns:
        val = getattr(obj, c.name)
        if isinstance(val, datetime):
            val = val.isoformat()
        result[c.name] = val
    return result


def context_loader(state: CofounderState, *, db: Session | None = None) -> dict:
    """Load user profile, hypotheses, decisions, timeline, and metrics from DB.

    Args:
        state: Current graph state; must contain ``user_id``.
        db: Optional SQLAlchemy session (injected in tests; auto-created otherwise).

    Returns:
        Partial state dict with ``user_profile``, ``active_hypotheses``,
        ``problem_statement``, ``recent_metrics``, ``recent_decisions``,
        ``recent_timeline``, and ``company_context``.
    """
    from src.memory.company_timeline import TimelineManager
    from src.memory.decisions_tracker import DecisionTracker
    from src.memory.hypotheses_tracker import HypothesesTracker
    from src.memory.problem_statement import ProblemStatementManager
    from src.memory.user_profile import UserProfileManager

    own_session = db is None
    if own_session:
        db = next(get_db())
    try:
        user_id = state["user_id"]

        # User profile
        profile_mgr = UserProfileManager(db)
        user = profile_mgr.get_user(user_id)
        user_profile = _serialize(user) if user else {"id": user_id}

        # Active hypotheses
        hyp_tracker = HypothesesTracker(db)
        active_hyps = hyp_tracker.get_active_hypotheses()
        active_hypotheses = [_serialize(h) for h in active_hyps]

        # Problem statement
        ps_mgr = ProblemStatementManager(db)
        active_ps = ps_mgr.get_active()
        problem_statement = _serialize(active_ps) if active_ps else None

        # Recent decisions (last 10)
        dec_tracker = DecisionTracker(db)
        decisions = dec_tracker.get_decisions()[:10]
        recent_decisions = [_serialize(d) for d in decisions]

        # Recent timeline (last 10)
        tl_mgr = TimelineManager(db)
        timeline = tl_mgr.get_recent_events(limit=10)
        recent_timeline = [_serialize(e) for e in timeline]

        # Latest metrics
        latest_metric = db.query(Metric).order_by(Metric.date.desc()).first()
        recent_metrics = _serialize(latest_metric) if latest_metric else None

        # Build enriched company context
        dynamic_parts = []
        if problem_statement:
            dynamic_parts.append(
                f"\n## Current Problem Statement\n"
                f"Problem: {problem_statement.get('problem', 'N/A')}\n"
                f"Solution: {problem_statement.get('solution', 'N/A')}\n"
                f"Target Customer: {problem_statement.get('target_customer', 'N/A')}"
            )
        if active_hypotheses:
            hyp_lines = [f"- {h.get('hypothesis', '')}" for h in active_hypotheses[:5]]
            dynamic_parts.append(
                f"\n## Active Hypotheses ({len(active_hypotheses)} total)\n"
                + "\n".join(hyp_lines)
            )
        if recent_decisions:
            dec_lines = [f"- {d.get('decision', '')}" for d in recent_decisions[:5]]
            dynamic_parts.append(
                f"\n## Recent Decisions\n" + "\n".join(dec_lines)
            )

        company_context = COMPANY_CONTEXT
        if dynamic_parts:
            company_context += "\n\n## Live Business Context" + "\n".join(dynamic_parts)

        return {
            "user_profile": user_profile,
            "active_hypotheses": active_hypotheses,
            "problem_statement": problem_statement,
            "recent_metrics": recent_metrics,
            "recent_decisions": recent_decisions,
            "recent_timeline": recent_timeline,
            "company_context": company_context,
        }
    finally:
        if own_session:
            db.close()
