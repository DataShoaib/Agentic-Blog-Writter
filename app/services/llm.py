from __future__ import annotations
import logging
import re
import time
from typing import Any

from litellm import Router
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.config import APP_CONFIG, get_secrets

logger = logging.getLogger(__name__)

TEXT_OUTPUT_TOKENS, STRUCTURED_OUTPUT_TOKENS = 3000, 3400

# Full-article revision must re-emit EVERY planned section (often 2500-4500
# words). Capping it at TEXT_OUTPUT_TOKENS (~1300 tokens) truncates the
# article mid-sentence and silently drops trailing sections — exactly the
# "plan has 7 sections but final blog has 3" bug. This budget is only used
# for whole-article rewrites, never for single-section workers.
REVISION_OUTPUT_TOKENS = 12000

# Structured output is the only sanctioned way to get data out of the LLM:
# every caller passes a pydantic schema (or plain dict) and receives a
# validated instance back. No other module parses raw LLM JSON.
StructuredSchema = type[BaseModel] | type[dict]

# Defensive recovery for providers that ignore `response_format` and wrap the
# JSON payload in prose or markdown fences. The normal path is native
# structured output; this only rescues contract-violating completions.
_JSON_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")


class LLMGatewayError(RuntimeError):
    """Raised when the LLM Router cannot complete a request."""


_QUOTA_HINTS = ("quota", "rate limit", "429", "tpm", "rpm", "requests per day", "resource exhausted")


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _QUOTA_HINTS)


def _friendly_gateway_error(operation: str, candidates: list[str], exc: Exception) -> LLMGatewayError:
    """Translate a raw LLM failure into a user-actionable message."""
    if _is_quota_error(exc):
        return LLMGatewayError(
            f"LLM quota/rate limit reached for '{operation}' on all tried models ({', '.join(candidates)}). "
            "The Gemini free tier allows only ~20 requests/day per model. Either wait for the daily reset, "
            "add/replace the GEMINI_API_KEY in the .env file, or upgrade the key to Pay-as-you-go in "
            "Google AI Studio."
        )
    return LLMGatewayError(
        f"LLM operation '{operation}' failed after trying {candidates}."
    )


def model_candidates() -> list[str]:
    secrets = get_secrets()
    primary = getattr(secrets, "llm_model", None) or APP_CONFIG.llm_model
    fallbacks = getattr(secrets, "llm_fallback_models", None) or APP_CONFIG.llm_fallback_models
    models = [model for model in dict.fromkeys([primary, *fallbacks]) if model]

    if not models:
        raise RuntimeError("No LLM model is configured.")
    return models


def _build_router() -> Router:
    secrets = get_secrets()
    return Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_key": secrets.gemini_api_key,
                    "timeout": APP_CONFIG.request_timeout_seconds,
                },
            }
            for model in model_candidates()
        ],
        num_retries=2,
        # Failed models cool down briefly so the router moves on to the next
        # candidate instead of hammering the same exhausted quota.
        cooldown_time=30,
        allowed_fails=3,
    )


_router: Router | None = None


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = _build_router()
    return _router


_TOKEN_USAGE: list[dict[str, Any]] = []


def _record_usage(response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage:
        _TOKEN_USAGE.append({"model": getattr(response, "model", "unknown"), "prompt_tokens": getattr(usage, "prompt_tokens", 0), "completion_tokens": getattr(usage, "completion_tokens", 0)})


def _to_messages(messages: list[Any]) -> list[dict[str, str]]:
    roles = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    return [{"role": str(message.get("role", "user")), "content": str(message.get("content", ""))} if isinstance(message, dict) else {"role": roles.get(getattr(message, "type", "user"), "user"), "content": str(message.content)} for message in messages]


def _complete(messages: list[dict[str, str]], *, operation: str, preferred_model: str | None = None, max_tokens: int, response_format: type[BaseModel] | None = None, temperature: float = 0) -> Any:
    candidates = model_candidates()

    # Ordered try-list: the user's preferred model first, then every other
    # candidate. This guarantees a real failover chain — requesting an exact
    # provider route from litellm's Router does NOT fall back on its own.
    if preferred_model and preferred_model in candidates:
        ordered = [preferred_model, *(model for model in candidates if model != preferred_model)]
    else:
        ordered = list(candidates)

    last_exc: Exception | None = None
    for index, model in enumerate(ordered):
        kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if response_format:
            kwargs["response_format"] = response_format
        try:
            response = _get_router().completion(**kwargs)
            _record_usage(response)
            return response
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "LLM model %s failed for operation=%s: %s", model, operation, str(exc)[:300]
            )
            # A candidate remains: pause briefly before handing over, honouring
            # the provider's suggested backoff for quota/rate-limit failures.
            if index < len(ordered) - 1:
                if _is_quota_error(exc):
                    wait = float(getattr(exc, "retry_delay_seconds", 0) or 2.0)
                    time.sleep(min(wait, 3.0))
                else:
                    time.sleep(1.0)

    assert last_exc is not None
    raise _friendly_gateway_error(operation, candidates, last_exc) from last_exc


def _content(response: Any) -> str:
    return (response.choices[0].message.content or "").strip()


def _json_payload(content: str) -> str:
    """Extract the embedded JSON object/array from a prose-wrapped completion.

    Used only when a provider violates the response_format contract (e.g.
    wraps the payload in "Here is the answer: ```json ... ```").
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = _JSON_FENCE_RE.sub("", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return text
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    return text[start : end + 1] if end > start else text

_A_STRING_OPENERS = frozenset("{[,:=(")


def _looks_like_string_start(text: str, i: int) -> bool:
    """Heuristic: is a single-quote at ``i`` a JSON string delimiter?

    True when, ignoring whitespace, the previous char is an opener of a key or
    value position (``{ [ , : =`` or start of input) AND the next non-whitespace
    char is actual string content rather than an immediate closing delimiter.
    """
    j = i - 1
    while j >= 0 and text[j] in " \t\r\n":
        j -= 1
    if j >= 0 and text[j] not in _A_STRING_OPENERS:
        return False
    k = i + 1
    while k < len(text) and text[k] in " \t\r\n":
        k += 1
    if k >= len(text) or text[k] in "},]":
        return False
    return True


def _repair_json(text: str) -> str:
    """Best-effort repair of LLM-emitted JSON.

    Handles the most common ways LLMs violate the JSON contract:
      * literal newlines/tabs/carriage returns inside string values,
      * stray unescaped double-quotes inside double-quoted string values,
      * single-quoted strings (converted to double-quoted delimiters),
      * missing or mismatched closing brackets (e.g. an unclosed array).

    Pydantic validation still runs after this, so anything still invalid is
    rejected before use; this only ever returns a *better-approximating* string.
    """
    out = []
    i = 0
    n = len(text)
    in_str = False
    delim = ""
    stack = []
    while i < n:
        ch = text[i]
        if in_str:
            nxt = text[i + 1] if i + 1 < n else ""
            if ch == "\\" and nxt:
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            if ch == '"' and delim == '"':
                # Terminator or stray quote? Peek ahead: structural -> closer.
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j < n and text[j] in ":,]}":
                    out.append('"')
                    in_str = False
                    delim = ""
                else:
                    out.append('\\"')
                i += 1
                continue
            if ch == delim:
                out.append('"')
                in_str = False
                delim = ""
                i += 1
                continue
            if ch in ("\r", "\n", "\t"):
                out.append({"\r": "\\r", "\n": "\\n", "\t": "\\t"}[ch])
                i += 1
                continue
            if ch == '"' and delim == "'":
                out.append('\\"')
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        if ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            want = "{" if ch == "}" else "["
            if stack and stack[-1] != want:
                # Mismatched closer (e.g. "}" while an array is still open):
                # close the open bracket first so the current one lands right.
                opened = stack.pop()
                out.append("]" if opened == "[" else "}")
            if stack and stack[-1] == want:
                stack.pop()
            out.append(ch)
        elif ch == '"':
            in_str = True
            delim = '"'
            out.append('"')
        elif ch == "'" and _looks_like_string_start(text, i):
            in_str = True
            delim = "'"
            out.append('"')
        else:
            out.append(ch)
        i += 1
    if in_str:
        out.append('"')
    while stack:
        opened = stack.pop()
        out.append("]" if opened == "[" else "}")
    return "".join(out)


def invoke_text(messages: list[Any], *, operation: str, preferred_model: str | None = None, max_tokens: int | None = None, temperature: float = 0.7) -> str:
    return _content(_complete(_to_messages(messages), operation=operation, preferred_model=preferred_model, max_tokens=max_tokens if max_tokens is not None else TEXT_OUTPUT_TOKENS, temperature=temperature))


def invoke_structured(
    schema: StructuredSchema,
    messages: list[Any],
    *,
    operation: str,
    preferred_model: str | None = None,
) -> Any:
    """Structured output through the gateway.

    Primary contract: native provider structured output (`response_format`),
    schema-validated on our side with a pydantic TypeAdapter. If a provider
    violates the contract and returns prose-wrapped JSON, the payload is
    recovered defensively before the same validation runs.
    """
    response_format = schema if isinstance(schema, type) and issubclass(schema, BaseModel) else None
    response = _complete(
        _to_messages(messages),
        operation=operation,
        preferred_model=preferred_model,
        max_tokens=STRUCTURED_OUTPUT_TOKENS,
        response_format=response_format,
    )
    content = _content(response)
    adapter = TypeAdapter(schema)
    payload = _json_payload(content)
    # Try the raw completion, then the fence-stripped payload. If a provider
    # violates its response_format contract and emits malformed JSON, raise
    # LLMGatewayError so callers degrade gracefully instead of crashing the
    # whole job on a single flaky completion.
    for candidate in (content, payload, _repair_json(payload), _repair_json(content)):
        try:
            return adapter.validate_json(candidate)
        except ValidationError:
            continue
        except Exception:
            continue
    schema_name = getattr(schema, '__name__', 'schema')
    msg = (
        f"LLM operation {operation!r} returned unparseable JSON for schema"
        f" {schema_name} ({len(content)} chars); the model ignored its"
        f" response_format contract and JSON repair could not recover it."
    )
    raise LLMGatewayError(msg)


def get_recorded_token_usage() -> list[dict[str, Any]]:
    return [dict(usage) for usage in _TOKEN_USAGE]

