"""Hermetic unit tests for the LiteLLM-based retry/fallback gateway."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY
from pydantic import BaseModel

from app.config import AppConfig
from app.services import llm


def _sample(name: str, labels: dict) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


def _resp(content: str, model: str = "test-model"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
    )


@pytest.fixture
def gateway(monkeypatch):
    """Install a deterministic model chain and a scripted fake completion."""

    def install(models, script_factory, attempts=2, backoff=0.0, cap=0.0):
        requested: list[str] = []

        def fake_completion(**kwargs):
            model = kwargs["model"]
            if model not in requested:  # record routes once per model, like clients
                requested.append(model)
            return script_factory(model)(kwargs["messages"])

        def fake_secrets():
            return SimpleNamespace(
                groq_api_key="test-key",
                langsmith_api_key="",
                groq_model=models[0],
                groq_fallback_models="",
            )

        cfg = AppConfig(
            groq_model=models[0],
            llm_max_attempts_per_model=attempts,
            llm_retry_backoff_seconds=backoff,
            llm_retry_backoff_cap_seconds=cap,
            llm_fallback_models=tuple(models[1:]),
        )
        monkeypatch.setattr(llm, "_completion", fake_completion)
        monkeypatch.setattr(llm, "get_secrets", fake_secrets)
        monkeypatch.setattr(llm, "APP_CONFIG", cfg)

        sleeps: list[float] = []
        monkeypatch.setattr(llm, "time", SimpleNamespace(sleep=sleeps.append))
        return requested, sleeps

    return install


def test_model_candidates_route_through_groq_provider(monkeypatch):
    names = ["openai/gpt-oss-20b", "qwen/qwen3.6-27b", "groq/llama-3.3-70b"]
    monkeypatch.setattr(
        llm,
        "get_secrets",
        lambda: SimpleNamespace(groq_model=names[0], groq_fallback_models=""),
    )
    monkeypatch.setattr(
        llm,
        "APP_CONFIG",
        AppConfig(groq_model=names[0], llm_fallback_models=(names[1], names[2])),
    )

    routed = llm.model_candidates()

    assert routed == [
        "groq/openai/gpt-oss-20b",
        "groq/qwen/qwen3.6-27b",
        "groq/llama-3.3-70b",
    ]


def test_retries_transient_error_on_same_model(gateway):
    calls = {"count": 0}

    def script_factory(model):
        def handler(messages):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("upstream timed out")
            return _resp("section text", model)

        return handler

    requested, sleeps = gateway(("primary-model", "fallback-model"), script_factory)

    result = llm.invoke_text([SimpleNamespace(content="hi")], operation="t-retry")

    assert result == "section text"
    assert requested == ["groq/primary-model"]  # never fell back
    assert len(sleeps) == 1  # exactly one backoff sleep
    assert _sample("agentic_llm_retries_total", {"operation": "t-retry"}) == 1.0


def test_rate_limit_honours_try_again_hint(gateway):
    def script_factory(model):
        def handler(messages):
            raise RuntimeError("rate_limit_exceeded: Rate limit reached. Try again in 7s")

        return handler

    _, sleeps = gateway(("only-model",), script_factory, attempts=2)

    with pytest.raises(llm.LLMGatewayError):
        llm.invoke_text([SimpleNamespace(content="hi")], operation="t-hint")

    assert sleeps == [7.0]


def test_permanent_error_switches_to_fallback_immediately(gateway):
    def script_factory(model):
        def handler(messages):
            if model == "groq/primary-model":
                raise ValueError("invalid request shape")
            return _resp("fallback text", model)

        return handler

    requested, sleeps = gateway(("primary-model", "fallback-model"), script_factory)

    result = llm.invoke_text([SimpleNamespace(content="hi")], operation="t-fb")

    assert result == "fallback text"
    assert requested[0] == "groq/primary-model"
    assert "groq/fallback-model" in requested
    assert sleeps == []  # no retries burned on the broken primary
    assert _sample("agentic_llm_model_fallbacks_total", {"operation": "t-fb"}) == 1.0


def test_all_models_exhausted_raises_gateway_error(gateway):
    calls = {"count": 0}

    def script_factory(model):
        def handler(messages):
            calls["count"] += 1
            raise ConnectionError("provider unreachable")

        return handler

    requested, sleeps = gateway(("primary", "fallback-a"), script_factory, attempts=2)

    with pytest.raises(llm.LLMGatewayError):
        llm.invoke_text([SimpleNamespace(content="hi")], operation="t-dead")

    assert requested == ["groq/primary", "groq/fallback-a"]  # one route per model
    assert calls["count"] == 4  # 2 models x 2 attempts
    assert len(sleeps) == 2  # one backoff per model's first failure
    assert _sample("agentic_llm_calls_total", {"operation": "t-dead"}) == 1.0
    assert _sample("agentic_llm_failures_total", {"operation": "t-dead"}) == 1.0


def test_empty_content_is_retried_then_succeeds(gateway):
    responses = iter([_resp("   "), _resp("real text")])

    def script_factory(model):
        def handler(messages):
            return next(responses)

        return handler

    requested, sleeps = gateway(("primary-model",), script_factory)

    result = llm.invoke_text([SimpleNamespace(content="hi")], operation="t-empty")

    assert result == "real text"
    assert requested == ["groq/primary-model"]
    assert len(sleeps) == 1


class Judge(BaseModel):
    value: int


def test_structured_output_uses_fallback_chain(gateway):
    def script_factory(model):
        def handler(messages):
            if model == "groq/primary":
                raise ValueError("schema rejected")
            return _resp(json.dumps({"value": 42}), model)

        return handler

    requested, _ = gateway(("primary", "secondary"), script_factory)

    result = llm.invoke_structured(
        Judge, [SimpleNamespace(content="hi")], operation="t-struct"
    )

    assert result.value == 42
    assert requested == ["groq/primary", "groq/secondary"]


def test_structured_output_parses_prose_wrapped_json(gateway):
    def script_factory(model):
        def handler(messages):
            return _resp("Here is the answer:\n```json\n{\"value\": 7}\n```", model)

        return handler

    gateway(("primary",), script_factory)

    result = llm.invoke_structured(
        Judge, [SimpleNamespace(content="hi")], operation="t-prose"
    )

    assert result.value == 7


def test_token_usage_is_recorded(gateway):
    def script_factory(model):
        def handler(messages):
            return SimpleNamespace(
                model=model,
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        return handler

    gateway(("primary",), script_factory)
    llm._TOKEN_USAGE.clear()

    llm.invoke_text([SimpleNamespace(content="hi")], operation="t-usage")

    assert llm.get_recorded_token_usage() == [
        {"model": "groq/primary", "prompt_tokens": 10, "completion_tokens": 5}
    ]


def test_output_budget_stays_under_tpm_window(monkeypatch):
    monkeypatch.setattr(
        llm, "_estimate_input_tokens", lambda messages: 7000
    )  # big research context

    budget = llm._output_budget([], TEXT_DESIRED := 1300)

    assert budget == 600  # 7600 - 7000 remaining, floored at the minimum


def test_gateway_config_defaults_are_sane():
    cfg = AppConfig()

    assert cfg.llm_max_attempts_per_model >= 1
    assert 0 < cfg.llm_retry_backoff_seconds <= cfg.llm_retry_backoff_cap_seconds
    assert cfg.llm_fallback_models
    assert len(set(cfg.llm_fallback_models)) == len(cfg.llm_fallback_models)
