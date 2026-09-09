from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: list[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., ge=120, le=550, description="Target words (120–550).")

    tags: list[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(..., min_length=1)

    @field_validator("tasks")
    @classmethod
    def require_unique_task_ids(cls, tasks: list[Task]) -> list[Task]:
        """Safety rail: merge_content sorts sections by task.id — duplicate ids
        would silently overwrite/lose sections."""
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan task IDs must be unique.")
        return tasks


class EvidenceItem(BaseModel):
    title: str = ""  # default empty: search results sometimes lack titles (never invent one)
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: list[str] = Field(default_factory=list)
    max_results_per_query: int = Field(default=5)


class EvidencePack(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., pattern=r"^\[\[IMAGE_[1-3]\]\]$", description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"
    section: str = Field(
        "",
        description=(
            "EXACT section heading from the article outline where this image "
            "belongs (e.g. 'The Double Descent Risk Curve'). Empty defaults to "
            "the end of the article."
        ),
    )


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str = Field(
        "",
        description=(
            "Legacy echo field. Placeholders are auto-injected into the full "
            "article by the graph node, so leave this empty."
        ),
    )
    images: list[ImageSpec] = Field(default_factory=list)


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
