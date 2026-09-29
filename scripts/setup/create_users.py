"""Create seed user profiles for the two sample users (Founder and Co-founder).

Inserts the initial ``User`` records that the multi-user memory system
expects to exist.  Existing users (matched by ``id``) are left untouched.

Usage::

    python scripts/setup/create_users.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.memory.database import SessionLocal, User, init_db

#: Sample user profiles for the fictional company. Replace with your own.
USERS = [
    {
        "id": "founder",
        "name": "Founder",
        "role": "Founder",
        "title": "Co-founder",
        "communication_style": {"style": "plain", "prefers_examples": True, "detail_level": "medium"},
        "expertise_areas": ["grocery operations", "purchasing", "store rollout"],
        "interests": ["customer interviews", "pilot design", "pricing"],
        "working_patterns": {"timezone": "UTC"},
        "decision_style": {"type": "evidence-first", "speed": "steady"},
    },
    {
        "id": "cofounder",
        "name": "Co-founder",
        "role": "Co-founder",
        "title": "Co-founder",
        "communication_style": {"style": "plain", "prefers_examples": True, "detail_level": "high"},
        "expertise_areas": ["demand forecasting", "data pipelines", "product design"],
        "interests": ["forecast accuracy", "store onboarding", "reporting"],
        "working_patterns": {"timezone": "UTC"},
        "decision_style": {"type": "evidence-first", "speed": "steady"},
    },
]


def main():
    """Insert seed user profiles, skipping any that already exist."""
    init_db()
    db = SessionLocal()
    try:
        for user_data in USERS:
            existing = db.query(User).filter(User.id == user_data["id"]).first()
            if existing:
                print(f"User '{user_data['id']}' already exists, skipping.")
                continue
            user = User(**user_data)
            db.add(user)
            print(f"Created user: {user_data['id']} ({user_data['role']})")
        db.commit()
        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
