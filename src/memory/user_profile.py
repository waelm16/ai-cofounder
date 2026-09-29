"""User profile management with auto-enrichment.

Provides CRUD operations for User records and an ``enrich_profile`` method
that incrementally merges learned behavioural attributes (communication style,
expertise, interests, etc.) into the user's JSON columns — enabling the system
to tailor responses to each cofounder over time.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from src.memory.database import User


class UserProfileManager:
    """Manages user profiles stored in the ``users`` table.

    Args:
        db: An active SQLAlchemy session.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_user(self, user_id: str) -> User | None:
        """Fetch a single user by their unique string ID.

        Args:
            user_id: The user's primary-key identifier.

        Returns:
            The User record, or None if not found.
        """
        return self.db.query(User).filter(User.id == user_id).first()

    def update_user(self, user_id: str, updates: dict) -> User | None:
        """Apply arbitrary column updates to a user record.

        Args:
            user_id: The user to update.
            updates: Dict mapping column names to new values. Keys that do
                not correspond to a User column attribute are silently ignored.

        Returns:
            The refreshed User, or None if the user was not found.
        """
        user = self.get_user(user_id)
        if not user:
            return None
        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)
        user.last_active = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_cofounder(self, user_id: str) -> User | None:
        """Get the other cofounder's profile.

        Args:
            user_id: The current user's ID — the returned user will be
                a different user.

        Returns:
            The first User whose ID differs from *user_id*, or None.
        """
        return self.db.query(User).filter(User.id != user_id).first()

    def enrich_profile(self, user_id: str, learned_data: dict) -> User | None:
        """Merge newly learned attributes into existing JSON fields.

        For each recognised JSON field present in *learned_data*:
        - **dict + dict**: shallow-merge (new keys override).
        - **list + list**: union (deduplicated via ``set``).
        - **None + any**: set the field to the new value outright.

        Args:
            user_id: The user whose profile to enrich.
            learned_data: Dict whose keys are JSON column names
                (e.g. "expertise_areas") and values are the new data to merge.

        Returns:
            The refreshed User, or None if the user was not found.
        """
        user = self.get_user(user_id)
        if not user:
            return None

        json_fields = [
            "communication_style",
            "expertise_areas",
            "interests",
            "working_patterns",
            "decision_style",
        ]

        for field in json_fields:
            if field not in learned_data:
                continue

            current = getattr(user, field)
            new_value = learned_data[field]

            if isinstance(current, dict) and isinstance(new_value, dict):
                merged = {**current, **new_value}
                setattr(user, field, merged)
            elif isinstance(current, list) and isinstance(new_value, list):
                merged = list(set(current + new_value))
                setattr(user, field, merged)
            elif current is None:
                setattr(user, field, new_value)

        user.last_active = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        return user
