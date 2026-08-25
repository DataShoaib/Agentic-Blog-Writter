"""Process-wide LangSmith tracing setup driven by configured secrets.

Tracing used to be enabled lazily inside the LLM gateway, so the API and worker
processes only picked it up after their first model call, and graph runs had no
guaranteed root trace. This module switches tracing on explicitly, once per
process, straight from ``app.config`` so a local ``.env`` file is enough.
"""

from __future__ import annotations

import logging
import os
import sys

from app.config import get_secrets

logger = logging.getLogger(__name__)

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
DEFAULT_PROJECT = "agentic-content-orchestrator"

_configured = False


def configure_langsmith() -> bool:
    """Enable LangSmith tracing for this process when an API key is present.

    Safe to call multiple times; only the first call per process mutates the
    environment. Returns True when tracing ends up enabled. Under pytest the
    switch stays off unless ``LANGSMITH_TRACING=force`` is set explicitly, so
    the test suite never talks to the LangSmith API.
    """
    global _configured
    if _configured:
        return tracing_enabled()
    _configured = True

    forced = os.getenv("LANGSMITH_TRACING", "").strip().lower() == "force"
    if "pytest" in sys.modules and not forced:
        logger.debug("LangSmith tracing left disabled under pytest.")
        return False

    secrets = get_secrets()
    if not secrets.langsmith_api_key:
        logger.info("LANGSMITH_API_KEY is not set; LangSmith tracing stays disabled.")
        return False

    endpoint = os.getenv("LANGSMITH_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT)
    project = os.getenv("LANGSMITH_PROJECT", DEFAULT_PROJECT)

    # Modern names plus the legacy LANGCHAIN_* aliases that langchain-core and
    # older integrations still honor, so every layer picks the switch up.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = secrets.langsmith_api_key
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_API_KEY"] = secrets.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = project

    logger.info("LangSmith tracing enabled (project=%s endpoint=%s).", project, endpoint)

    # Fail loudly on a bad key instead of dropping traces silently.
    state, _detail = _verify_langsmith_auth()
    if state != "ok":
        # Tracing cannot deliver without valid auth; switch it off rather than
        # pretending to trace while every upload dies in a background thread.
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return False
    return True


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"


def active_project() -> str | None:
    """Configured LangSmith project name when tracing is on, otherwise None."""
    if not tracing_enabled():
        return None
    return os.getenv("LANGSMITH_PROJECT", DEFAULT_PROJECT)


_auth_state: tuple[str, str] = ("not_checked", "")


def _verify_langsmith_auth(timeout_seconds: float = 6.0) -> tuple[str, str]:
    """Ping the LangSmith API once so a bad key surfaces at startup.

    The tracing client uploads in a background thread and logs auth failures
    quietly, which made broken keys invisible. This check makes the failure
    loud: it is reported via ``auth_state`` and the /health endpoint.
    """
    global _auth_state
    try:
        from langsmith import Client

        Client(timeout_ms=int(timeout_seconds * 1000), retry_config=None).list_projects(limit=1)
        _auth_state = ("ok", "")
        logger.info("LangSmith authentication verified.")
    except Exception as exc:  # noqa: BLE001 - report any auth/network failure
        detail = f"{type(exc).__name__}: {str(exc)[:180]}"
        _auth_state = ("failed", detail)
        logger.error(
            "LangSmith authentication FAILED - traces are NOT being uploaded. "
            "Fix LANGSMITH_API_KEY in .env. Detail: %s",
            detail,
        )
    return _auth_state


def auth_state() -> tuple[str, str]:
    """(state, detail) of the last auth verification: ok/failed/not_checked."""
    return _auth_state