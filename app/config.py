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
    groq_api_key: str = ""
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

    # All chat models are full litellm routes. Gemini is primary; each model
    # carries its OWN free-tier daily quota, so a longer candidate list
    # multiplies the free budget. Verified live 2026-09-23 (probe script):
    # dead IDs (404) were dropped — gemini-2.5-flash-lite, gemini-2.0-flash,
    # groq llama-3.3-70b-versatile, groq llama-3.1-8b-instant.
    llm_model: str = "gemini/gemini-2.5-flash"
    llm_fallback_models: tuple[str, ...] = (
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.1-flash-lite",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-flash-latest",
        # Groq free tier — separate quota bucket, needs GROQ_API_KEY.
        "groq/openai/gpt-oss-20b",
        "groq/openai/gpt-oss-120b",
    )
    image_model: str = "gemini-2.5-flash-image"
    # Image-capable models, each with its own free-tier quota.
    image_fallback_models: tuple[str, ...] = (
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-image-preview",
        "gemini-3.1-flash-lite-image",
        "gemini-3-pro-image",
    )
    max_workers: int = 8
    max_revision_attempts: int = 1
    max_research_results: int = 3
    max_research_queries: int = 4
    request_timeout_seconds: int = 60
    job_timeout_seconds: int = 900
    rate_limit_per_minute: int = 10
    # Task-specific models for classification/evaluation-only nodes. These
    # point at WORKING lite routes (not the daily-capped primary): when the
    # primary hits its ~20 req/day Gemini free quota, the router calls that
    # start every job still get through instead of dying at step one.
    router_model: str = "gemini/gemini-3.5-flash-lite"
    quality_gate_model: str = "gemini/gemini-3.5-flash-lite"


APP_CONFIG = AppConfig()


@lru_cache
def get_secrets() -> Secrets:
    return Secrets()
