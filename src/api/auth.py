"""Authentication dependency for FastAPI routes.

Provides a simple dev-mode auth scheme where the ``Authorization`` header
carries a Bearer token whose value is the user ID (e.g. ``Bearer founder``).
The user is looked up in the database and returned as the current user.
"""

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from src.memory.database import User, get_db


def get_current_user(
    authorization: str = Header(...),
    db: Session = Depends(get_db),
) -> User:
    """Extract user_id from the Bearer token and validate against the DB.

    Dev-mode auth: the token value is the literal user ID.

    Args:
        authorization: ``Authorization`` header value (``Bearer <user_id>``).
        db: SQLAlchemy session injected by FastAPI.

    Returns:
        The authenticated ``User`` ORM instance.

    Raises:
        HTTPException: 401 if the header is malformed, empty, or the user
            does not exist.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    user_id = authorization.removeprefix("Bearer ").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user ID in token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail=f"User '{user_id}' not found")

    return user
