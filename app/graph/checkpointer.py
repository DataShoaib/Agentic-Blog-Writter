"""Process-wide shared LangGraph checkpointer — PostgreSQL only."""

from __future__ import annotations

from langgraph.checkpoint.postgres import PostgresSaver

from app.config import get_secrets

_checkpointer = None


def get_default_checkpointer():
    """Return the cached process-wide Postgres checkpointer instance."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    url = get_secrets().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not configured.")

    _saver_cm = PostgresSaver.from_conn_string(url)
    saver = _saver_cm.__enter__()
    saver.setup()  # creates checkpoint tables on first run
    _checkpointer = saver
    return _checkpointer
