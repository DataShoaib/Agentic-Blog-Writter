from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from app.graph.schemas import EvidenceItem, Plan


class GraphState(TypedDict, total=False):
    topic: str
    mode: str
    research_mode: str
    needs_research: bool
    queries: list[str]
    max_results_per_query: int
    evidence: list[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[list[tuple[int, str]], operator.add]
    merged_md: str
    quality: dict
    revision_count: int
    max_revision_attempts: int
    image_specs: list[dict]
    enable_images: bool
    final: str
    job_id: str
    user_id: str
    memory_note: str
