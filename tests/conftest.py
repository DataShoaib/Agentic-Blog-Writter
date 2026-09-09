"""Shared pytest fixtures.

The production stack is PostgreSQL-only (DATABASE_URL). Tests that exercise the
real JobStore/UserStore need a reachable Postgres; in environments without one
(local dev without docker-compose) they are skipped instead of erroring.
"""
from __future__ import annotations

import pytest

from app.config import get_secrets

_available: bool | None = None


def _postgres_reachable() -> bool:
    global _available
    if _available is None:
        url = (get_secrets().database_url or "").strip()
        if not url:
            _available = False
        else:
            try:
                import psycopg

                conn = psycopg.connect(url, connect_timeout=3)
                conn.close()
                _available = True
            except Exception:
                _available = False
    return _available


@pytest.fixture
def requires_db():
    """Skip unless a live PostgreSQL (DATABASE_URL) is reachable."""
    if not _postgres_reachable():
        pytest.skip(
            "PostgreSQL not reachable (DATABASE_URL missing or server down); "
            "start it with: docker-compose up -d postgres"
        )
