"""Clear separation between private environment values and committed defaults."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Only private or deployment-specific values loaded from .env."""

    gemini_api_key: str = ""
    tavily_api_key: str = ""
    google_api_key: str = ""
    langsmith_api_key: str = ""
    jwt_secret_key: str = ""
    redis_url: str = ""
    database_url: str = ""
    llm_model: str = ""
    llm_fallback_models: tuple[str, ...] = ()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


@dataclass(frozen=True)
class AppConfig:
    """Committed application behaviour; change here, not in .env."""

    # All chat models are full litellm routes. Gemini is primary; the
    # fallbacks are additional Gemini variants so the same API key serves
    # every candidate. Each model carries its OWN free-tier daily quota
    # (~20 req/day), so a longer candidate list multiplies the free budget.
    # The list is verified against the live Gemini ListModels endpoint:
    # legacy ids (2.0-flash, 1.5-flash, 2.5-pro) have been decommissioned.
    llm_model: str = "gemini/gemini-2.5-flash"
    llm_fallback_models: tuple[str, ...] = (
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-flash-latest",
    )
    image_model: str = "gemini-2.5-flash-image"
    # Live-verified (Gemini ListModels, 2026-09) image-capable models. Each
    # carries its own free-tier quota, so a dead primary can still yield an
    # image via the retry loop in app/services/images.py.
    image_fallback_models: tuple[str, ...] = (
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-image-preview",
        "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image",
    )
    max_workers: int = 4
    max_revision_attempts: int = 1
    max_research_results: int = 3
    max_research_queries: int = 4
    request_timeout_seconds: int = 60
    job_timeout_seconds: int = 900
    search_timeout_seconds: int = 20
    image_timeout_seconds: int = 90
    cache_ttl_seconds: int = 900
    rate_limit_per_minute: int = 10
    worker_start_delay_seconds: float = 1.0


APP_CONFIG = AppConfig()


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()
