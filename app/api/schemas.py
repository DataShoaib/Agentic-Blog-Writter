from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=10)


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class SignupResponse(BaseModel):
    username: str
    message: str = "Account created successfully."


class GenerateRequest(BaseModel):
    topic: str = Field(min_length=5, max_length=500)
    as_of: date | None = None
    preferred_model: str | None = Field(default=None, max_length=200)

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, topic: str) -> str:
        topic = topic.strip()
        if len(topic) < 5:
            raise ValueError("Topic must contain at least 5 non-whitespace characters.")
        return topic


class GenerateResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    content: str | None = None
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    stage: str | None = None
    stage_detail: str | None = None
    progress: float | None = None
    plan: dict | None = None
    evidence: list[dict] = Field(default_factory=list)


class BlogSummary(BaseModel):
    job_id: str
    topic: str
    title: str
    created_at: str | None = None


class BlogListResponse(BaseModel):
    blogs: list[BlogSummary] = Field(default_factory=list)
