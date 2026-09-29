"""Initialize the SQLAlchemy database, creating all tables defined in the ORM models.

Calls :func:`src.memory.database.init_db` which issues ``CREATE TABLE IF NOT EXISTS``
for every model registered on the declarative base. Safe to run multiple times.

Usage::

    python scripts/setup/init_database.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.memory.database import init_db


def main():
    """Create all database tables (idempotent)."""
    print("Initializing database...")
    init_db()
    print("Database initialized successfully.")


if __name__ == "__main__":
    main()
