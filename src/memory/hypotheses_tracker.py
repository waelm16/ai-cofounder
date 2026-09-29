"""Hypothesis tracking with evidence accumulation and validation scoring.

Provides CRUD operations for Hypothesis records. Hypotheses support
incremental evidence tracking (for/against) and can be auto-generated
from customer call transcripts.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from src.memory.database import Hypothesis


class HypothesesTracker:
    """CRUD manager for business hypotheses with evidence tracking.

    Args:
        db: An active SQLAlchemy session.
    """

    def __init__(self, db: Session):
        self.db = db

    def add_hypothesis(
        self,
        hypothesis: str,
        owner_id: str,
        category: str | None = None,
        validation_criteria: list[dict] | None = None,
        start_date: datetime | None = None,
        collaborators: list[str] | None = None,
        auto_generated: bool = False,
        source_call_id: int | None = None,
        evidence_for: list[dict] | None = None,
        evidence_against: list[dict] | None = None,
        validation_score: float | None = None,
    ) -> Hypothesis:
        """Create and persist a new hypothesis.

        Args:
            hypothesis: The testable statement.
            owner_id: User ID responsible for validation.
            category: Domain category (e.g. "pricing", "distribution").
            validation_criteria: List of dicts describing success criteria.
            start_date: When testing began; defaults to now.
            collaborators: List of additional user IDs involved.
            auto_generated: True if created by the transcript analyser.
            source_call_id: ID of the CustomerCall that generated this hypothesis.
            evidence_for: Initial supporting evidence (list of dicts).
            evidence_against: Initial contradicting evidence (list of dicts).
            validation_score: Initial composite score (0-1).

        Returns:
            The newly created Hypothesis.
        """
        record = Hypothesis(
            hypothesis=hypothesis,
            category=category,
            owner_id=owner_id,
            collaborators=collaborators,
            start_date=start_date or datetime.utcnow(),
            validation_criteria=validation_criteria,
            auto_generated=auto_generated,
            source_call_id=source_call_id,
            evidence_for=evidence_for,
            evidence_against=evidence_against,
            validation_score=validation_score,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def add_evidence(
        self, hypothesis_id: int, evidence_type: str, evidence: dict
    ) -> Hypothesis | None:
        """Append a single piece of evidence to a hypothesis.

        Args:
            hypothesis_id: The hypothesis to update.
            evidence_type: Either "for" (supporting) or "against" (contradicting).
            evidence: A dict describing the evidence (e.g. source, summary, date).

        Returns:
            The refreshed Hypothesis, or None if not found or invalid type.
        """
        record = self.get_hypothesis(hypothesis_id)
        if not record:
            return None

        if evidence_type == "for":
            current = list(record.evidence_for or [])
            current.append(evidence)
            record.evidence_for = current
        elif evidence_type == "against":
            current = list(record.evidence_against or [])
            current.append(evidence)
            record.evidence_against = current
        else:
            return None

        self.db.commit()
        self.db.refresh(record)
        return record

    def get_hypotheses(
        self,
        category: str | None = None,
        status: str | None = None,
        owner_id: str | None = None,
    ) -> list[Hypothesis]:
        """Query hypotheses with optional filters.

        Args:
            category: Filter by domain category.
            status: "active" returns hypotheses with no result; any other
                string filters by exact ``result`` value.
            owner_id: Filter by owning user ID.

        Returns:
            List of matching Hypothesis records, newest first.
        """
        query = self.db.query(Hypothesis)

        if category:
            query = query.filter(Hypothesis.category == category)
        if owner_id:
            query = query.filter(Hypothesis.owner_id == owner_id)
        if status == "active":
            query = query.filter(Hypothesis.result.is_(None))
        elif status:
            query = query.filter(Hypothesis.result == status)

        return query.order_by(Hypothesis.start_date.desc()).all()

    def get_hypothesis(self, hypothesis_id: int) -> Hypothesis | None:
        """Fetch a single hypothesis by primary key.

        Args:
            hypothesis_id: The hypothesis's integer ID.

        Returns:
            The Hypothesis, or None if not found.
        """
        return self.db.query(Hypothesis).filter(Hypothesis.id == hypothesis_id).first()

    def update_hypothesis(self, hypothesis_id: int, updates: dict) -> Hypothesis | None:
        """Apply column updates to an existing hypothesis.

        Args:
            hypothesis_id: The hypothesis to update.
            updates: Dict of column-name to new-value pairs. The "id" key
                is silently skipped.

        Returns:
            The refreshed Hypothesis, or None if not found.
        """
        record = self.get_hypothesis(hypothesis_id)
        if not record:
            return None
        for key, value in updates.items():
            if hasattr(record, key) and key != "id":
                setattr(record, key, value)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_active_hypotheses(self) -> list[Hypothesis]:
        """Return hypotheses that have no end_date and no result.

        Returns:
            List of still-active Hypothesis records, newest first.
        """
        return (
            self.db.query(Hypothesis)
            .filter(Hypothesis.end_date.is_(None), Hypothesis.result.is_(None))
            .order_by(Hypothesis.start_date.desc())
            .all()
        )
