"""Process-wide shared LangGraph checkpointer — PostgreSQL only."""

from __future__ import annotations

from langgraph.checkpoint.postgres import PostgresSaver

from app.config import get_secrets

_checkpointer = None
_saver_cm = None


def get_default_checkpointer():
    """Return the cached process-wide Postgres checkpointer instance."""
    global _checkpointer, _saver_cm
    if _checkpointer is not None:
        return _checkpointer

    url = get_secrets().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not configured.")

    # The context manager MUST stay referenced for the lifetime of the
    # process: if it is garbage collected, its __exit__ closes the shared
    # Postgres connection and every later checkpoint call fails with
    # "the connection is closed".
    _saver_cm = PostgresSaver.from_conn_string(url)
    _checkpointer = _saver_cm.__enter__()
    _checkpointer.setup()  # creates checkpoint tables on first run
    return _checkpointer

