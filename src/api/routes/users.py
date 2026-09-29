"""User management API routes.

Endpoints:
    GET  /api/users/          — list all users
    GET  /api/users/{user_id} — get a single user profile
    PUT  /api/users/{user_id} — partially update a user profile
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.auth import get_current_user
from src.api.schemas import UserResponse, UserUpdate
from src.memory.database import User, get_db
from src.memory.user_profile import UserProfileManager

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/", response_model=list[UserResponse])
def list_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all registered user profiles.

    Returns:
        list[UserResponse]: Every user in the database.
    """
    users = db.query(User).all()
    return users


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Fetch a single user profile by ID.

    Args:
        user_id: The unique user identifier.

    Returns:
        UserResponse: The matching user profile.

    Raises:
        HTTPException: 404 if the user does not exist.
    """
    manager = UserProfileManager(db)
    user = manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: str,
    updates: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Partially update a user profile.

    Only fields present in the request body are modified (``exclude_unset``).

    Args:
        user_id: The unique user identifier.
        updates: Fields to update.

    Returns:
        UserResponse: The updated user profile.

    Raises:
        HTTPException: 404 if the user does not exist.
    """
    manager = UserProfileManager(db)
    update_data = updates.model_dump(exclude_unset=True)
    user = manager.update_user(user_id, update_data)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
