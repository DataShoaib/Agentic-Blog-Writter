from __future__ import annotations

import logging
import os
import sys

from app.config import get_secrets

logger = logging.getLogger(__name__)

DEFAULT_LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
DEFAULT_PROJECT = "agentic-content-orchestrator"

_configured = False
_auth_state: tuple[str, str] = ("not_checked", "")


def configure_langsmith() -> bool:
    global _configured

    if _configured:
        return tracing_enabled()

    _configured = True
    forced = os.getenv("LANGSMITH_TRACING", "").strip().lower() == "force"

    if "pytest" in sys.modules and not forced:
        logger.debug("LangSmith tracing disabled under pytest.")
        return False

    api_key = get_secrets().langsmith_api_key

    if not api_key:
        logger.info("LANGSMITH_API_KEY is not set; tracing disabled.")
        return False

    endpoint = os.getenv("LANGSMITH_ENDPOINT", DEFAULT_LANGSMITH_ENDPOINT)
    project = os.getenv("LANGSMITH_PROJECT", DEFAULT_PROJECT)

    os.environ.update(
        {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": api_key,
            "LANGSMITH_ENDPOINT": endpoint,
            "LANGSMITH_PROJECT": project,
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_ENDPOINT": endpoint,
            "LANGCHAIN_API_KEY": api_key,
            "LANGCHAIN_PROJECT": project,
        }
    )

    logger.info(
        "LangSmith tracing enabled (project=%s endpoint=%s).",
        project,
        endpoint,
    )

    state, _ = _verify_langsmith_auth()

    if state != "ok":
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        return False

    return True


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"


def active_project() -> str | None:
    if not tracing_enabled():
        return None
    return os.getenv("LANGSMITH_PROJECT", DEFAULT_PROJECT)


def _verify_langsmith_auth(timeout_seconds: float = 6.0) -> tuple[str, str]:
    global _auth_state

    try:
        from langsmith import Client

        Client(
            timeout_ms=int(timeout_seconds * 1000),
            retry_config=None,
        ).list_projects(limit=1)

        _auth_state = ("ok", "")
        logger.info("LangSmith authentication verified.")

    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {str(exc)[:180]}"
        _auth_state = ("failed", detail)

        logger.error(
            "LangSmith authentication failed; traces are not being uploaded. "
            "Fix LANGSMITH_API_KEY in .env. Detail: %s",
            detail,
        )

    return _auth_state


def auth_state() -> tuple[str, str]:
    return _auth_state

