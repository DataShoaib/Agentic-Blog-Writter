"""Centralized LLM gateway built on LiteLLM.

All chat traffic flows through :func:`invoke_text` and :func:`invoke_structured`.
LiteLLM normalizes the provider API surface, so the gateway only adds what the
application actually needs: model fallback, transient retries, token-usage
accounting, and Prometheus metrics.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, TypeVar

from litellm import completion as _litellm_completion
from pydantic import BaseModel

from app.config import APP_CONFIG, get_secrets
from app.observability.metrics import (
    LLM_CALLS,
    LLM_FAILURES,
    LLM_MODEL_FALLBACKS,
    LLM_RETRIES,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

RATE_LIMIT_RETRY_SECONDS = 12
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
TEXT_OUTPUT_TOKENS = 1300
STRUCTURED_OUTPUT_TOKENS = 3400

# Module-level seam so tests can stub the transport without patching litellm.
_completion = _litellm_completion

# Per-call token usage, reported by the evaluation runner.
_TOKEN_USAGE: list[dict] = []


class LLMGatewayError(RuntimeError):
    """Raised when every configured model and retry attempt has been exhausted."""


class _EmptyContentError(RuntimeError):
    """The model answered but produced no usable text; retry/fallback applies."""


def model_candidates() -> list[str]:
    """Ordered litellm model names: primary first, then configured fallbacks.

    Configured names (e.g. "openai/gpt-oss-20b", "qwen/qwen3.6-27b") are Groq
    model ids, not litellm provider routes — they are all routed through the
    ``groq/`` provider so litellm talks to the Groq API with GROQ_API_KEY.
    """
    secrets = get_secrets()
    primary = secrets.groq_model or APP_CONFIG.groq_model
    overrides = [
        name.strip()
        for name in getattr(secrets, "groq_fallback_models", "").split(",")
        if name.strip()
    ]
    names = [primary, *(overrides or list(APP_CONFIG.llm_fallback_models))]

    candidates: list[str] = []
    for name in names:
        if not name:
            continue
        routed = name if name.startswith("groq/") else f"groq/{name}"
        if routed not in candidates:
            candidates.append(routed)

    if not candidates:
        raise RuntimeError("No LLM model is configured.")
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


def _to_litellm_messages(messages: list[Any]) -> list[dict[str, str]]:
    """Accept langchain BaseMessage objects or plain {role, content} dicts."""
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    converted: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, dict):
            converted.append({"role": message.get("role", "user"), "content": message.get("content", "")})
            continue
        role = getattr(message, "type", None) or "user"
        converted.append({"role": role_map.get(role, role), "content": str(getattr(message, "content", ""))})
    return converted

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
    # Empty generations are stochastic: a retry (or fallback model) frequently
    # succeeds, so they must not be treated as permanent faults.
    if isinstance(exc, _EmptyContentError):
        return True
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


def _record_token_usage(response: Any) -> None:
    usage = getattr(response, "usage", None)
    if not usage:
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    if not prompt and not completion:
        return
    _TOKEN_USAGE.append({
        "model": getattr(response, "model", "") or "",
        "prompt_tokens": int(prompt),
        "completion_tokens": int(completion),
    })


def get_recorded_token_usage() -> list[dict]:
    """Token usage per call, consumed by the evaluation runner's cost report."""
    return [dict(entry) for entry in _TOKEN_USAGE]


def _estimate_input_tokens(messages: list[dict[str, str]]) -> int:
    """Count input tokens, falling back to a chars/4 estimate if the
    tokenizer is unavailable (keeps the gateway dependency-light)."""
    try:
        from litellm import token_counter
        count = token_counter(messages=messages)
        if isinstance(count, (int, float)) and count > 0:
            return int(count)
    except Exception:
        pass
    return sum(len(m.get("content", "")) for m in messages) // 4


def _output_budget(messages: list[dict[str, str]], desired_output_tokens: int) -> int:
    """Shrink the output cap so input+output fits one free-tier TPM window.

    Groq returns 413 when a single request exceeds the TPM window, which on
    the free tier is small. Keep the request under ``llm_request_token_budget``
    but never shrink below a workable minimum.
    """
    remaining = APP_CONFIG.llm_request_token_budget - _estimate_input_tokens(messages)
    return max(256, min(desired_output_tokens, remaining, APP_CONFIG.llm_max_output_tokens))


def _call(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    response_format: type[BaseModel] | None = None,
) -> Any:
    """Single litellm.completion attempt."""
    api_key = get_secrets().groq_api_key
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "api_key": api_key,
        "num_retries": 0,  # the gateway owns the retry policy
        "timeout": APP_CONFIG.request_timeout_seconds,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    return _completion(**kwargs)


def _response_text(response: Any) -> str:
    """Extract plain text from a litellm completion response."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError):
        return ""
    return content.strip() if isinstance(content, str) else ""


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object/array from model text."""
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text
    return text[min(starts):]


def _invoke(
    messages: list[Any],
    *,
    operation: str,
    preferred_model: str | None,
    max_tokens: int,
    response_format: type[BaseModel] | None = None,
) -> str:
    """Retry/fallback core shared by invoke_text and invoke_structured."""
    secrets = get_secrets()
    if not secrets.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    converted = _to_litellm_messages(messages)
    max_tokens = _output_budget(converted, max_tokens)
    candidates = _ordered_candidates(preferred_model)
    max_attempts = max(1, APP_CONFIG.llm_max_attempts_per_model)
    last_exc: Exception | None = None

    for index, model in enumerate(candidates):
        for attempt in range(1, max_attempts + 1):
            try:
                response = _call(
                    model, converted, max_tokens=max_tokens, response_format=response_format
                )
                _record_token_usage(response)
                text = _response_text(response)
                if not text:
                    raise _EmptyContentError(
                        f"LLM returned empty content for operation '{operation}'."
                    )
                logger.info(
                    "llm operation=%s succeeded with model=%s on attempt %d",
                    operation, model, attempt,
                )
                return text
            except Exception as exc:
                last_exc = exc
                if not _is_transient_error(exc):
                    logger.warning(
                        "llm operation=%s model=%s permanent %s; moving to next model",
                        operation, model, type(exc).__name__,
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
    messages: list[Any],
    *,
    operation: str,
    preferred_model: str | None = None,
) -> str:
    """Generate free-form text; retries transient errors, falls back models."""
    LLM_CALLS.labels(operation=operation).inc()
    try:
        return _invoke(
            messages,
            operation=operation,
            preferred_model=preferred_model,
            max_tokens=TEXT_OUTPUT_TOKENS,
        )
    except Exception as exc:
        LLM_FAILURES.labels(operation=operation).inc()
        logger.error("llm operation=%s failed permanently: %s", operation, exc)
        raise


def invoke_structured(
    schema: type[T],
    messages: list[Any],
    *,
    operation: str,
    preferred_model: str | None = None,
) -> T:
    """Generate a Pydantic ``schema`` instance; same retry/fallback policy."""
    LLM_CALLS.labels(operation=operation).inc()
    structured_messages = [
        {"role": "system", "content": "Return only valid JSON matching the requested schema."},
        *_to_litellm_messages(messages),
    ]
    try:
        text = _invoke(
            structured_messages,
            operation=operation,
            preferred_model=preferred_model,
            max_tokens=STRUCTURED_OUTPUT_TOKENS,
            response_format=(
                schema
                if isinstance(schema, type) and issubclass(schema, BaseModel)
                else None
            ),
        )
        return _parse_structured(schema, text)
    except Exception as exc:
        LLM_FAILURES.labels(operation=operation).inc()
        logger.error("llm operation=%s failed permanently: %s", operation, exc)
        raise


def _parse_structured(schema: type[T], text: str) -> T:
    """Validate model output against ``schema``, tolerating prose-wrapped JSON."""
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            return schema.model_validate_json(text)
        except Exception:
            return schema.model_validate_json(_extract_json(text))
    return json.loads(_extract_json(text))  # type: ignore[return-value]
