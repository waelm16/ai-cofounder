"""Persistence layer for the multi-agent AI cofounder system.

Provides three memory layers:
    - **User-specific**: Per-user profiles, conversation history, tasks, preferences
      (UserProfileManager).
    - **Shared company**: Timeline events, decisions, metrics, hypotheses, versioned
      problem statements, and customer calls — all with multi-user attribution
      (TimelineManager, DecisionTracker, HypothesesTracker, ProblemStatementManager,
      CustomerCallManager).
    - **Collaboration**: Disagreement logs, collaboration patterns, knowledge transfers,
      and reflections (models in database.py).

All SQLAlchemy models and the database session factory live in ``database.py``.
Manager classes in sibling modules provide CRUD logic over those models.
"""
