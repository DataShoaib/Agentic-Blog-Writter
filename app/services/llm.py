from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import APP_CONFIG, get_secrets
from app.observability.metrics import (
    LLM_CALLS,
    LLM_FAILURES,
    LLM_MODEL_FALLBACKS,
    LLM_RETRIES,
)
from app.observability.tracing import configure_langsmith

logger = logging.getLogger(__name__)

T = TypeVar("T")

RATE_LIMIT_RETRY_SECONDS = 12
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
# Desired output sizes handed to the TPM-aware budget (input+output must fit
# inside one free-tier window; see _output_budget).
TEXT_OUTPUT_TOKENS = 1300
STRUCTURED_OUTPUT_TOKENS = 3400


class LLMGatewayError(RuntimeError):
    """Raised when every configured model and retry attempt has been exhausted."""


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    return status_code


def _is_rate_limit_error(exc: Exception) -> bool:
    return _status_code(exc) == 429 or "rate_limit" in str(exc).lower()


def _is_json_generation_error(exc: Exception) -> bool:
    """Provider-side failures where the model emitted invalid JSON."""
    text = str(exc).lower()
    return (
        "json_validate_failed" in text
        or "failed to generate json" in text
        or "failed to validate json" in text
        or "does not support response format" in text
    )


def _is_transient_error(exc: Exception) -> bool:
    """Mirror the transient classification used by the Tavily search service."""
    if _is_rate_limit_error(exc) or _status_code(exc) in RETRYABLE_STATUS_CODES:
        return True
    # Invalid-JSON generations are stochastic: a retry (or fallback model)
    # frequently succeeds, so they must not be treated as permanent faults.
    if _status_code(exc) == 400 and _is_json_generation_error(exc):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _retry_delay(exc: Exception) -> float:
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", str(exc), re.IGNORECASE)
    return min(max(float(match.group(1)) if match else RATE_LIMIT_RETRY_SECONDS, 1), 30)


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, capped by configuration."""
    delay = APP_CONFIG.llm_retry_backoff_seconds * (2 ** max(attempt - 1, 0))
    return min(delay * random.uniform(0.8, 1.2), APP_CONFIG.llm_retry_backoff_cap_seconds)


def _content_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts)
    return ""


def get_llm(model: str | None = None, *, max_tokens: int | None = None) -> ChatGroq:
    secrets = get_secrets()
    if not secrets.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    resolved = model or secrets.groq_model or APP_CONFIG.groq_model
    options: dict[str, Any] = {
        "model": resolved,
        "groq_api_key": secrets.groq_api_key,
        "temperature": 0,
        # Per-attempt timeout; the gateway below owns the retry policy.
        "timeout": APP_CONFIG.request_timeout_seconds,
        "max_retries": 0,
    }
    if resolved.startswith("openai/gpt-oss"):
        options["reasoning_format"] = "hidden"
        options["reasoning_effort"] = "low"
        options["max_tokens"] = max_tokens or APP_CONFIG.llm_max_output_tokens
    return ChatGroq(**options)


def _estimate_input_tokens(messages: list[BaseMessage] | None) -> int:
    """Cheap chars/3.5 heuristic; good enough to stay under a TPM budget."""
    if not messages:
        return 0
    chars = 0
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            chars += len(content)
        else:
            chars += len(str(content))
    return int(chars / 3.5) + 64


def _output_budget(messages: list[BaseMessage], desired_output_tokens: int) -> int:
    """Fit input+output inside one free-tier TPM window (Groq counts both).

    Requesting ``max_tokens`` larger than the remaining TPM makes even tiny
    prompts fail with 413 "Request too large", so the output cap shrinks to
    whatever room is actually left.
    """
    room = APP_CONFIG.llm_request_token_budget - _estimate_input_tokens(messages)
    return max(512, min(desired_output_tokens, room))


def get_llm_for_messages(
    model: str,
    messages: list[BaseMessage],
    desired_output_tokens: int,
):
    """get_llm variant whose output cap adapts to the message size."""
    client = get_llm(model)
    try:
        client.max_tokens = _output_budget(messages, desired_output_tokens)
    except Exception:  # pragma: no cover - best-effort attribute tweak
        pass
    return client


def model_candidates() -> list[str]:
    """Ordered chat models: primary first, then configured fallbacks."""
    secrets = get_secrets()
    primary = secrets.groq_model or APP_CONFIG.groq_model
    override = [
        name.strip()
        for name in getattr(secrets, "groq_fallback_models", "").split(",")
        if name.strip()
    ]
    candidates = [primary]
    for name in override or list(APP_CONFIG.llm_fallback_models):
        if name != primary and name not in candidates:
            candidates.append(name)
    return candidates


def _ordered_candidates(preferred_model: str | None) -> list[str]:
    """Fallback order starting from ``preferred_model`` when it is a candidate.

    Keeps the same overall fallback chain but lets callers (e.g. parallel
    section workers) start on different models so bursts spread across
    separate provider rate-limit buckets instead of hammering one pool.
    """
    base = model_candidates()
    if preferred_model and preferred_model in base:
        index = base.index(preferred_model)
        return base[index:] + base[:index]
    return base


def _call_with_gateway(
    build_call: Callable[[ChatGroq], Callable[[], Any]],
    *,
    operation: str,
    require_non_empty: bool = False,
    preferred_model: str | None = None,
    messages: list[BaseMessage] | None = None,
    desired_output_tokens: int | None = None,
) -> Any:
    """Run one logical LLM operation with bounded retries and model fallbacks.

    For each candidate model (primary first, then configured fallbacks) the call
    is attempted up to ``APP_CONFIG.llm_max_attempts_per_model`` times. Transient
    faults (timeouts, connection errors, HTTP 408/429/5xx) back off exponentially;
    permanent faults switch to the next fallback model immediately. Rate-limit
    responses honor Groq's "try again in Xs" hint. Raises LLMGatewayError when
    every option has been exhausted.
    """
    candidates = _ordered_candidates(preferred_model)
    max_attempts = max(1, APP_CONFIG.llm_max_attempts_per_model)
    last_exc: Exception | None = None

    for index, model in enumerate(candidates):
        if messages is not None and desired_output_tokens:
            client = get_llm_for_messages(model, messages, desired_output_tokens)
        else:
            client = get_llm(model)
        call = build_call(client)
        for attempt in range(1, max_attempts + 1):
            try:
                response = call()
            except Exception as exc:
                last_exc = exc
                if not _is_transient_error(exc):
                    logger.warning(
                        "llm operation=%s model=%s non-retryable %s: %s",
                        operation, model, type(exc).__name__, exc,
                    )
                    break  # stop burning retries on this model; try the next one
                if attempt < max_attempts:
                    delay = (
                        _retry_delay(exc)
                        if _is_rate_limit_error(exc)
                        else _backoff_delay(attempt)
                    )
                    LLM_RETRIES.labels(operation=operation).inc()
                    logger.warning(
                        "llm operation=%s model=%s transient %s on attempt %d/%d; "
                        "retrying in %.2fs",
                        operation, model, type(exc).__name__, attempt, max_attempts, delay,
                    )
                    time.sleep(delay)
                continue

            if not require_non_empty or _content_text(response):
                logger.info(
                    "llm operation=%s succeeded with model=%s on attempt %d",
                    operation, model, attempt,
                )
                return response

            last_exc = RuntimeError(f"LLM returned empty content for operation '{operation}'.")
            logger.warning(
                "llm operation=%s model=%s returned empty content on attempt %d/%d",
                operation, model, attempt, max_attempts,
            )
            if attempt < max_attempts:
                LLM_RETRIES.labels(operation=operation).inc()
                time.sleep(_backoff_delay(attempt))

        if index < len(candidates) - 1:
            LLM_MODEL_FALLBACKS.labels(operation=operation).inc()
            logger.warning(
                "llm operation=%s exhausted model=%s; falling back to %s",
                operation, model, candidates[index + 1],
            )

    raise LLMGatewayError(
        f"LLM operation '{operation}' failed after trying models {candidates}."
    ) from last_exc


def invoke_text(
    messages: list[BaseMessage],
    *,
    operation: str,
    preferred_model: str | None = None,
) -> str:
    LLM_CALLS.labels(operation=operation).inc()
    try:
        response = _call_with_gateway(
            lambda client: lambda: client.invoke(messages),
            operation=operation,
            require_non_empty=True,
            preferred_model=preferred_model,
            messages=messages,
            desired_output_tokens=TEXT_OUTPUT_TOKENS,
        )
        return _content_text(response)
    except Exception as exc:
        LLM_FAILURES.labels(operation=operation).inc()
        logger.error("llm operation=%s failed permanently: %s", operation, exc)
        raise


def invoke_structured(
    schema: type[T],
    messages: list[BaseMessage],
    *,
    operation: str,
    preferred_model: str | None = None,
) -> T:
    LLM_CALLS.labels(operation=operation).inc()
    structured_messages = [
        SystemMessage(content="Return only valid JSON matching the requested schema."),
        *messages,
    ]
    try:
        return _call_with_gateway(
            lambda client: lambda: client.with_structured_output(schema, method="json_schema").invoke(structured_messages),
            operation=operation,
            preferred_model=preferred_model,
            messages=structured_messages,
            desired_output_tokens=STRUCTURED_OUTPUT_TOKENS,
        )
    except Exception as exc:
        LLM_FAILURES.labels(operation=operation).inc()
        logger.error("llm operation=%s failed permanently: %s", operation, exc)
        raise
