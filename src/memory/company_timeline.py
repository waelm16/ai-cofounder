"""Shared company timeline management.

Provides CRUD operations for TimelineEvent records — the chronological
backbone of the shared company memory layer. Events track milestones,
meetings, pivots, and other notable occurrences with multi-user attribution.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from src.memory.database import TimelineEvent


class TimelineManager:
    """CRUD manager for the shared company timeline.

    Args:
        db: An active SQLAlchemy session.
    """

    def __init__(self, db: Session):
        self.db = db

    def add_event(
        self,
        title: str,
        created_by: str,
        date: datetime,
        event_type: str | None = None,
        participants: list[str] | None = None,
        description: str | None = None,
        context: dict | None = None,
    ) -> TimelineEvent:
        """Create and persist a new timeline event.

        Args:
            title: Short summary of the event.
            created_by: User ID of the event creator.
            date: When the event occurred.
            event_type: Category string (e.g. "milestone", "meeting").
            participants: List of user IDs involved; defaults to [created_by].
            description: Longer narrative about the event.
            context: Arbitrary structured metadata (stored as JSON).

        Returns:
            The newly created TimelineEvent.
        """
        event = TimelineEvent(
            title=title,
            event_type=event_type,
            date=date,
            description=description,
            context=context,
            created_by_id=created_by,
            participants=participants or [created_by],
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def get_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_type: str | None = None,
        participant: str | None = None,
    ) -> list[TimelineEvent]:
        """Query timeline events with optional filters.

        Args:
            start_date: Only events on or after this datetime.
            end_date: Only events on or before this datetime.
            event_type: Filter by category string.
            participant: Filter to events where this user ID appears in
                the participants list (applied in Python after the DB query
                because participants is a JSON column).

        Returns:
            List of matching TimelineEvent records, newest first.
        """
        query = self.db.query(TimelineEvent)

        if start_date:
            query = query.filter(TimelineEvent.date >= start_date)
        if end_date:
            query = query.filter(TimelineEvent.date <= end_date)
        if event_type:
            query = query.filter(TimelineEvent.event_type == event_type)

        events = query.order_by(TimelineEvent.date.desc()).all()

        # Participant filtering done in Python — JSON column not SQL-filterable
        if participant:
            events = [e for e in events if e.participants and participant in e.participants]

        return events

    def get_event(self, event_id: int) -> TimelineEvent | None:
        """Fetch a single timeline event by primary key.

        Args:
            event_id: The event's integer ID.

        Returns:
            The TimelineEvent, or None if not found.
        """
        return self.db.query(TimelineEvent).filter(TimelineEvent.id == event_id).first()

    def update_event(self, event_id: int, updates: dict) -> TimelineEvent | None:
        """Apply column updates to an existing timeline event.

        Args:
            event_id: The event to update.
            updates: Dict of column-name to new-value pairs. The "id" key
                is silently skipped to prevent accidental PK mutation.

        Returns:
            The refreshed TimelineEvent, or None if not found.
        """
        event = self.get_event(event_id)
        if not event:
            return None
        for key, value in updates.items():
            if hasattr(event, key) and key != "id":
                setattr(event, key, value)
        self.db.commit()
        self.db.refresh(event)
        return event

    def get_recent_events(self, limit: int = 10) -> list[TimelineEvent]:
        """Return the most recent timeline events.

        Args:
            limit: Maximum number of events to return (default 10).

        Returns:
            List of TimelineEvent records, newest first.
        """
        return (
            self.db.query(TimelineEvent)
            .order_by(TimelineEvent.date.desc())
            .limit(limit)
            .all()
        )
