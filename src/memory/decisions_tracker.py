"""Decision tracking with per-user rationales and timeline linking.

Provides CRUD operations for Decision records. Decisions capture who
proposed an idea, who participated, per-user rationales (JSON dict),
consensus status, and optional revisit dates for follow-up.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from src.memory.database import Decision


class DecisionTracker:
    """CRUD manager for business decisions.

    Args:
        db: An active SQLAlchemy session.
    """

    def __init__(self, db: Session):
        self.db = db

    def add_decision(
        self,
        decision: str,
        proposed_by: str,
        category: str | None = None,
        date: datetime | None = None,
        decision_makers: list[str] | None = None,
        rationales: dict | None = None,
        consensus: bool | None = None,
        consensus_type: str | None = None,
    ) -> Decision:
        """Record a new business decision.

        Args:
            decision: Text describing the decision.
            proposed_by: User ID of the proposer.
            category: Free-form category (e.g. "product", "hiring").
            date: When the decision was made; defaults to now.
            decision_makers: List of user IDs who participated; defaults
                to [proposed_by].
            rationales: Dict mapping user IDs to their reasoning.
            consensus: Whether the team reached agreement.
            consensus_type: How consensus was reached (e.g. "unanimous").

        Returns:
            The newly created Decision.
        """
        record = Decision(
            decision=decision,
            category=category,
            date=date or datetime.utcnow(),
            proposed_by_id=proposed_by,
            decision_makers=decision_makers or [proposed_by],
            rationales=rationales,
            consensus=consensus,
            consensus_type=consensus_type,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_decisions(
        self,
        category: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[Decision]:
        """Query decisions with optional filters.

        Args:
            category: Filter by decision category.
            start_date: Only decisions on or after this datetime.
            end_date: Only decisions on or before this datetime.

        Returns:
            List of matching Decision records, newest first.
        """
        query = self.db.query(Decision)

        if category:
            query = query.filter(Decision.category == category)
        if start_date:
            query = query.filter(Decision.date >= start_date)
        if end_date:
            query = query.filter(Decision.date <= end_date)

        return query.order_by(Decision.date.desc()).all()

    def get_decision(self, decision_id: int) -> Decision | None:
        """Fetch a single decision by primary key.

        Args:
            decision_id: The decision's integer ID.

        Returns:
            The Decision, or None if not found.
        """
        return self.db.query(Decision).filter(Decision.id == decision_id).first()

    def update_decision(self, decision_id: int, updates: dict) -> Decision | None:
        """Apply column updates to an existing decision.

        Args:
            decision_id: The decision to update.
            updates: Dict of column-name to new-value pairs. The "id" key
                is silently skipped.

        Returns:
            The refreshed Decision, or None if not found.
        """
        record = self.get_decision(decision_id)
        if not record:
            return None
        for key, value in updates.items():
            if hasattr(record, key) and key != "id":
                setattr(record, key, value)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_pending_revisits(self) -> list[Decision]:
        """Return decisions past their revisit_date that have no recorded outcome.

        Returns:
            List of Decision records ordered by revisit_date ascending.
        """
        now = datetime.utcnow()
        return (
            self.db.query(Decision)
            .filter(Decision.revisit_date <= now, Decision.outcome.is_(None))
            .order_by(Decision.revisit_date.asc())
            .all()
        )
