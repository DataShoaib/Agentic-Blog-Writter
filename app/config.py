"""Clear separation between private environment values and committed defaults."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Only private or deployment-specific values loaded from .env."""

    groq_api_key: str = ""
    tavily_api_key: str = ""
    pollinations_api_key: str = ""
    langsmith_api_key: str = ""
    jwt_secret_key: str = ""
    redis_url: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    # Comma-separated fallback chat models. Empty means use the committed
    # defaults from AppConfig.llm_fallback_models.
    groq_fallback_models: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


@dataclass(frozen=True)
class AppConfig:
    """Committed application behaviour; change here, not in .env."""

    groq_model: str = "openai/gpt-oss-20b"
    image_model: str = "gemini"
    max_workers: int = 4
    max_revision_attempts: int = 2
    max_research_results: int = 3
    max_research_queries: int = 4
    request_timeout_seconds: int = 60
    job_timeout_seconds: int = 900
    search_timeout_seconds: int = 20
    image_timeout_seconds: int = 90
    cache_ttl_seconds: int = 900
    rate_limit_per_minute: int = 10
    llm_max_attempts_per_model: int = 3
    llm_retry_backoff_seconds: float = 1.0
    llm_retry_backoff_cap_seconds: float = 20.0
    # Free-tier TPM counts input+max_tokens together (Groq returns 413 when a
    # single request exceeds the window), so keep the default output cap small
    # and let the gateway shrink it further for large inputs.
    llm_max_output_tokens: int = 3072
    llm_request_token_budget: int = 7600
    worker_start_delay_seconds: float = 0.6
    llm_fallback_models: tuple[str, ...] = (
        # Verified against the Groq /models endpoint; older llama-3.x ids now
        # return 404 model_not_found and could not rescue rate-limited runs.
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
    )


APP_CONFIG = AppConfig()


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()
