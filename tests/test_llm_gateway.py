"""Hermetic unit tests for the centralized LLM retry/fallback gateway."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY
from pydantic import BaseModel

from app.config import AppConfig
from app.services import llm


def _sample(name: str, labels: dict) -> float | None:
    return REGISTRY.get_sample_value(name, labels)


class FakeLLM:
    """Minimal ChatGroq stand-in whose behaviour is scripted per model name."""

    def __init__(self, script_factory):
        self._script_factory = script_factory
        self.model = ""

    def invoke(self, messages):
        return self._script_factory(self.model)(messages)

    def with_structured_output(self, schema, method=None):
        outer = self

        class _Structured:
            def invoke(self, messages):
                return outer._script_factory(outer.model)(messages)

        return _Structured()


@pytest.fixture
def gateway(monkeypatch):
    """Install a deterministic model chain and stub client creation."""

    def install(models, script_factory, attempts=2, backoff=0.0, cap=0.0):
        requested: list[str] = []

        def fake_get_llm(model=None):
            client = FakeLLM(script_factory)
            client.model = model or models[0]
            requested.append(client.model)
            return client

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
        monkeypatch.setattr(llm, "get_llm", fake_get_llm)
        monkeypatch.setattr(llm, "get_secrets", fake_secrets)
        monkeypatch.setattr(llm, "APP_CONFIG", cfg)

        sleeps: list[float] = []
        monkeypatch.setattr(llm, "time", SimpleNamespace(sleep=sleeps.append))
        return requested, sleeps

    return install


def test_retries_transient_error_on_same_model(gateway):
    calls = {"count": 0}

    def script_factory(model):
        def handler(messages):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("upstream timed out")
            return SimpleNamespace(content="section text")

        return handler

    requested, sleeps = gateway(("primary-model", "fallback-model"), script_factory)

    result = llm.invoke_text([SimpleNamespace(content="hi")], operation="t-retry")

    assert result == "section text"
    assert requested == ["primary-model"]  # never fell back
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
            if model == "primary-model":
                raise ValueError("invalid request shape")
            return SimpleNamespace(content="fallback text")

        return handler

    requested, sleeps = gateway(("primary-model", "fallback-model"), script_factory)

    result = llm.invoke_text([SimpleNamespace(content="hi")], operation="t-fb")

    assert result == "fallback text"
    assert requested[0] == "primary-model"
    assert "fallback-model" in requested
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

    assert requested == ["primary", "fallback-a"]  # client built once per model
    assert calls["count"] == 4  # 2 models x 2 attempts
    assert len(sleeps) == 2  # one backoff per model's first failure
    assert _sample("agentic_llm_calls_total", {"operation": "t-dead"}) == 1.0
    assert _sample("agentic_llm_failures_total", {"operation": "t-dead"}) == 1.0


def test_empty_content_is_retried_then_succeeds(gateway):
    responses = iter(
        [SimpleNamespace(content="   "), SimpleNamespace(content="real text")]
    )

    def script_factory(model):
        def handler(messages):
            return next(responses)

        return handler

    requested, sleeps = gateway(("primary-model",), script_factory)

    result = llm.invoke_text([SimpleNamespace(content="hi")], operation="t-empty")

    assert result == "real text"
    assert requested == ["primary-model"]
    assert len(sleeps) == 1


class Judge(BaseModel):
    value: int


def test_structured_output_uses_fallback_chain(gateway):
    def script_factory(model):
        def handler(messages):
            if model == "primary":
                raise ValueError("schema rejected")
            return Judge(value=42)

        return handler

    requested, _ = gateway(("primary", "secondary"), script_factory)

    result = llm.invoke_structured(
        Judge, [SimpleNamespace(content="hi")], operation="t-struct"
    )

    assert result.value == 42
    assert requested == ["primary", "secondary"]


def test_gateway_config_defaults_are_sane():
    cfg = AppConfig()

    assert cfg.llm_max_attempts_per_model >= 1
    assert 0 < cfg.llm_retry_backoff_seconds <= cfg.llm_retry_backoff_cap_seconds
    assert cfg.llm_fallback_models
    assert len(set(cfg.llm_fallback_models)) == len(cfg.llm_fallback_models)