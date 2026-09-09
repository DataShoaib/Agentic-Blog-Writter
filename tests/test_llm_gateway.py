"""Hermetic unit tests for the LiteLLM Router-based gateway."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.config import AppConfig
from app.services import llm


def _resp(content, model="gemini/primary", usage=None):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


@pytest.fixture
def gateway(monkeypatch):
    def install(models, handler):
        monkeypatch.setattr(llm, "model_candidates", lambda: list(models))
        monkeypatch.setattr(llm, "_get_router", lambda: SimpleNamespace(completion=handler))
        monkeypatch.setattr(llm, "get_secrets", lambda: SimpleNamespace(gemini_api_key="test-key", llm_model=""))
    return install


def test_model_candidates_are_full_litellm_routes(monkeypatch):
    monkeypatch.setattr(llm, "get_secrets", lambda: SimpleNamespace(
        gemini_api_key="test-key",
        llm_model="gemini/gemini-2.5-flash",
    ))
    monkeypatch.setattr(llm, "APP_CONFIG", AppConfig(
        llm_model="gemini/gemini-2.5-flash",
        llm_fallback_models=("gemini/gemini-2.5-flash-lite", "gemini/gemini-2.0-flash"),
    ))

    assert llm.model_candidates() == [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash-lite",
        "gemini/gemini-2.0-flash",
    ]


def test_transient_error_is_delegated_to_router(gateway):
    calls = {"count": 0}

    def handler(**kwargs):
        calls["count"] += 1
        return _resp("section text")

    gateway(["gemini/primary"], handler)

    result = llm.invoke_text([SimpleNamespace(content="hi")], operation="t-retry")

    assert result == "section text"
    assert calls["count"] == 1


def test_router_failure_raises_gateway_error(gateway):
    def handler(**kwargs):
        raise RuntimeError("quota exceeded")

    gateway(["gemini/primary"], handler)

    with pytest.raises(llm.LLMGatewayError):
        llm.invoke_text([SimpleNamespace(content="hi")], operation="t-dead")


def test_structured_output_uses_selected_model(gateway):
    class Judge(BaseModel):
        value: int

    def handler(**kwargs):
        return _resp(json.dumps({"value": 42}), "gemini/primary")

    gateway(["gemini/primary"], handler)

    result = llm.invoke_structured(Judge, [SimpleNamespace(content="hi")], operation="t-struct")
    assert result.value == 42


def test_structured_output_parses_prose_wrapped_json(gateway):
    class Judge(BaseModel):
        value: int

    def handler(**kwargs):
        return _resp('Here is the answer:\n```json\n{"value": 7}\n```', "gemini/primary")

    gateway(["gemini/primary"], handler)

    result = llm.invoke_structured(Judge, [SimpleNamespace(content="hi")], operation="t-prose")
    assert result.value == 7


def test_token_usage_is_recorded(gateway):
    def handler(**kwargs):
        return _resp("ok", "gemini/primary", SimpleNamespace(prompt_tokens=10, completion_tokens=5))

    gateway(["gemini/primary"], handler)
    llm._TOKEN_USAGE.clear()

    llm.invoke_text([SimpleNamespace(content="hi")], operation="t-usage")

    assert llm.get_recorded_token_usage() == [
        {"model": "gemini/primary", "prompt_tokens": 10, "completion_tokens": 5}
    ]


def test_invoke_structured_plain_schema(gateway):
    def handler(**kwargs):
        return _resp('{"ok": true}', "gemini/primary")

    gateway(["gemini/primary"], handler)

    result = llm.invoke_structured(dict, [SimpleNamespace(content="hi")], operation="t-dict")
    assert result == {"ok": True}


def test_structured_output_malformed_json_raises_gateway_error(gateway):
    """A contract-violating, unparseable completion must not crash callers.

    Regression: previously a malformed quality-gate verdict raised an opaque
    ValidationError and aborted the whole job. It must surface as the graceful
    LLMGatewayError that quality_gate already knows how to degrade on.
    """
    from pydantic import BaseModel

    class Judge(BaseModel):
        value: int

    def handler(**kwargs):
        # Missing closing brace -> unparseable JSON.
        return _resp('{"passed": false, "issues": ["bad"]', "gemini/primary")

    gateway(["gemini/primary"], handler)

    with pytest.raises(llm.LLMGatewayError):
        llm.invoke_structured(Judge, [SimpleNamespace(content="hi")], operation="t-malformed")
