"""Database connection layer — PostgreSQL only."""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from app.config import get_secrets


def connect(path: str | None = None):
    """Open a PostgreSQL connection with dict-like rows.

    ``connect_timeout`` bounds the TCP/connection phase so a down database
    fails fast instead of hanging the worker or the test suite.
    """
    url = get_secrets().database_url
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set it to the PostgreSQL service from docker-compose.yml."
        )
    return psycopg.connect(url, row_factory=dict_row, autocommit=True, connect_timeout=5)


def q(sql: str) -> str:
    """Rewrite `?` placeholders to PostgreSQL `%s`."""
    return sql.replace("?", "%s")


def is_unique_violation(exc: Exception) -> bool:
    """True for a PostgreSQL UNIQUE constraint violation."""
    return type(exc).__name__ == "UniqueViolation"