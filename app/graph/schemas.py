from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Task(BaseModel):
    id: int = Field(ge=1)
    title: str = Field(min_length=3, max_length=120)
    goal: str = Field(min_length=10, max_length=500)
    bullets: list[str] = Field(min_length=3, max_length=7)
    target_words: int = Field(ge=150, le=800)
    tags: list[str] = Field(default_factory=list, max_length=10)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

    @field_validator("bullets")
    @classmethod
    def remove_empty_bullets(cls, bullets: list[str]) -> list[str]:
        cleaned = [bullet.strip() for bullet in bullets if bullet and bullet.strip()]
        if len(cleaned) < 3:
            raise ValueError("Each task must contain at least 3 non-empty bullets.")
        return cleaned[:7]


class Plan(BaseModel):
    blog_title: str = Field(min_length=5, max_length=180)
    audience: str = Field(min_length=3, max_length=200)
    tone: str = Field(min_length=3, max_length=120)
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: list[str] = Field(default_factory=list, max_length=12)
    tasks: list[Task] = Field(min_length=5, max_length=9)

    @field_validator("tasks")
    @classmethod
    def require_unique_task_ids(cls, tasks: list[Task]) -> list[Task]:
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan task IDs must be unique.")
        return tasks


class EvidenceItem(BaseModel):
    title: str = Field(default="", max_length=300)
    url: str = Field(min_length=10, max_length=2000)
    published_at: Optional[str] = None
    snippet: Optional[str] = Field(default=None, max_length=2000)
    source: Optional[str] = Field(default=None, max_length=200)


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str = Field(min_length=5, max_length=500)
    queries: list[str] = Field(default_factory=list, max_length=10)
    max_results_per_query: int = Field(default=5, ge=1, le=10)


class EvidencePack(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=50)


class ImageSpec(BaseModel):
    placeholder: str = Field(pattern=r"^\[\[IMAGE_[1-3]\]\]$")
    filename: str = Field(min_length=3, max_length=160)
    alt: str = Field(min_length=3, max_length=300)
    caption: str = Field(min_length=3, max_length=300)
    prompt: str = Field(min_length=20, max_length=2000)
    aspect_ratio: Literal["1:1", "16:9", "9:16"] = "16:9"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: list[ImageSpec] = Field(default_factory=list, max_length=3)


class QualityResult(BaseModel):
    passed: bool
    factuality_score: float = Field(ge=0, le=1)
    completeness_score: float = Field(ge=0, le=1)
    citation_score: float = Field(ge=0, le=1)
    issues: list[str] = Field(default_factory=list, max_length=12)


class EvaluationResult(BaseModel):
    factuality_score: float = Field(ge=0, le=1)
    completeness_score: float = Field(ge=0, le=1)
    citation_score: float = Field(ge=0, le=1)
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list, max_length=10)
