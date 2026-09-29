"""CRUD manager for customer calls.

Manages the full lifecycle of customer discovery and follow-up calls:
scheduling, transcript ingestion, and retrieval. Analysis (Mom Test
question generation, transcript extraction, auto-hypothesis creation)
lives in ``src/intelligence/`` and updates call records through this
manager's ``update_call`` method.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from src.memory.database import CustomerCall


class CustomerCallManager:
    """CRUD manager for CustomerCall records.

    Args:
        db: An active SQLAlchemy session.
    """

    def __init__(self, db: Session):
        self.db = db

    def create_call(
        self,
        call_date: datetime,
        customer_name: str,
        created_by_id: str,
        duration_minutes: int | None = None,
        customer_company: str | None = None,
        customer_role: str | None = None,
        customer_context: dict | None = None,
        call_type: str | None = None,
    ) -> CustomerCall:
        """Schedule a new customer call.

        The call is created with ``status="scheduled"``. After the call
        takes place, use ``update_call`` to add the transcript and change
        the status to "completed" or "analyzed".

        Args:
            call_date: Scheduled date/time of the call.
            customer_name: Name of the customer contact.
            created_by_id: User ID who scheduled the call.
            duration_minutes: Expected or actual duration.
            customer_company: Company the customer belongs to.
            customer_role: Customer's job title or role.
            customer_context: JSON-serialisable dict with background info.
            call_type: Category — "discovery", "follow_up", "demo", etc.

        Returns:
            The newly created CustomerCall.
        """
        record = CustomerCall(
            call_date=call_date,
            customer_name=customer_name,
            created_by_id=created_by_id,
            duration_minutes=duration_minutes,
            customer_company=customer_company,
            customer_role=customer_role,
            customer_context=customer_context,
            call_type=call_type,
            status="scheduled",
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_call(self, call_id: int) -> CustomerCall | None:
        """Fetch a single customer call by primary key.

        Args:
            call_id: The call's integer ID.

        Returns:
            The CustomerCall, or None if not found.
        """
        return self.db.query(CustomerCall).filter(CustomerCall.id == call_id).first()

    def get_calls(
        self,
        status: str | None = None,
        call_type: str | None = None,
        created_by_id: str | None = None,
    ) -> list[CustomerCall]:
        """Query customer calls with optional filters.

        Args:
            status: Filter by lifecycle status ("scheduled", "completed",
                "analyzed").
            call_type: Filter by call category.
            created_by_id: Filter by the user who scheduled the call.

        Returns:
            List of matching CustomerCall records, newest first.
        """
        query = self.db.query(CustomerCall)
        if status:
            query = query.filter(CustomerCall.status == status)
        if call_type:
            query = query.filter(CustomerCall.call_type == call_type)
        if created_by_id:
            query = query.filter(CustomerCall.created_by_id == created_by_id)
        return query.order_by(CustomerCall.call_date.desc()).all()

    def update_call(self, call_id: int, updates: dict) -> CustomerCall | None:
        """Apply column updates to an existing customer call.

        Args:
            call_id: The call to update.
            updates: Dict of column-name to new-value pairs. The "id" key
                is silently skipped.

        Returns:
            The refreshed CustomerCall, or None if not found.
        """
        record = self.get_call(call_id)
        if not record:
            return None
        for key, value in updates.items():
            if hasattr(record, key) and key != "id":
                setattr(record, key, value)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_recent_calls(self, limit: int = 10) -> list[CustomerCall]:
        """Return the most recent customer calls.

        Args:
            limit: Maximum number of calls to return (default 10).

        Returns:
            List of CustomerCall records, newest first.
        """
        return (
            self.db.query(CustomerCall)
            .order_by(CustomerCall.call_date.desc())
            .limit(limit)
            .all()
        )
