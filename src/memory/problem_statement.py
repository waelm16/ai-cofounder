"""Versioned problem-solution-customer statement management.

Enforces the invariant that only one ProblemStatement is active at any
time. Creating a new version automatically deactivates the previous one
and increments the version counter.
"""

from sqlalchemy.orm import Session

from src.memory.database import ProblemStatement


class ProblemStatementManager:
    """CRUD manager for versioned problem statements.

    Args:
        db: An active SQLAlchemy session.
    """

    def __init__(self, db: Session):
        self.db = db

    def create_problem_statement(
        self,
        problem: str,
        solution: str,
        target_customer: str,
        created_by_id: str,
    ) -> ProblemStatement:
        """Create a new problem statement, deactivating any prior active one.

        The version number is auto-incremented based on total record count.

        Args:
            problem: The customer problem being addressed.
            solution: The proposed solution.
            target_customer: Description of the ideal customer profile.
            created_by_id: User ID of the creator.

        Returns:
            The newly created (and now active) ProblemStatement.
        """
        # Deactivate any existing active statement
        self.db.query(ProblemStatement).filter(
            ProblemStatement.is_active.is_(True)
        ).update({"is_active": False})

        # Auto-increment version based on existing records
        count = self.db.query(ProblemStatement).count()
        version = count + 1

        record = ProblemStatement(
            problem=problem,
            solution=solution,
            target_customer=target_customer,
            version=version,
            is_active=True,
            created_by_id=created_by_id,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_active(self) -> ProblemStatement | None:
        """Return the currently active problem statement.

        Returns:
            The active ProblemStatement, or None if none exists.
        """
        return (
            self.db.query(ProblemStatement)
            .filter(ProblemStatement.is_active.is_(True))
            .first()
        )

    def get_all_versions(self) -> list[ProblemStatement]:
        """Return all problem statement versions, newest first.

        Returns:
            List of all ProblemStatement records ordered by descending version.
        """
        return (
            self.db.query(ProblemStatement)
            .order_by(ProblemStatement.version.desc())
            .all()
        )

    def update_problem_statement(
        self, problem_id: int, updates: dict
    ) -> ProblemStatement | None:
        """Apply column updates to an existing problem statement.

        Args:
            problem_id: The problem statement's integer ID.
            updates: Dict of column-name to new-value pairs. The "id" key
                is silently skipped.

        Returns:
            The refreshed ProblemStatement, or None if not found.
        """
        record = (
            self.db.query(ProblemStatement)
            .filter(ProblemStatement.id == problem_id)
            .first()
        )
        if not record:
            return None
        for key, value in updates.items():
            if hasattr(record, key) and key != "id":
                setattr(record, key, value)
        self.db.commit()
        self.db.refresh(record)
        return record

    def evolve(
        self,
        problem: str,
        solution: str,
        target_customer: str,
        created_by_id: str,
    ) -> ProblemStatement:
        """Deactivate the current active statement and create a new version.

        This is the primary method for evolving the problem statement as
        customer discovery progresses. Delegates to ``create_problem_statement``
        which handles deactivation and version incrementing.

        Args:
            problem: The updated customer problem.
            solution: The updated solution.
            target_customer: The updated ideal customer profile.
            created_by_id: User ID of the creator.

        Returns:
            The newly created ProblemStatement (now active).
        """
        return self.create_problem_statement(
            problem=problem,
            solution=solution,
            target_customer=target_customer,
            created_by_id=created_by_id,
        )
