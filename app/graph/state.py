from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from app.graph.schemas import EvidenceItem, Plan


class GraphState(TypedDict, total=False):
    # From the user
    topic: str                  # blog topic
    as_of: str                  # run date — backend-owned, always today (/generate)
    model: str | None           # preferred model (None = auto fallback)

    # From the backend
    job_id: str                 # unique job id (also checkpointer thread_id)
    user_id: str                # logged-in user id (JWT)

    # From the router node
    mode: str                   # closed_book | hybrid | open_book
    needs_research: bool        # True → research node, False → skip
    queries: list[str]          # search queries for Tavily
    max_results_per_query: int  # results per query
    recency_days: int           # age limit for sources

    # From the research node
    evidence: list[EvidenceItem]  # deduped search results

    # From the planner node
    plan: Optional[Plan]        # section plan

    # From the worker nodes (one per plan task)
    sections: Annotated[list[tuple[int, str]], operator.add]  # (task_id, section markdown)

    # From merge + revise + quality gate
    merged_md: str              # full article after merge/revise
    quality: dict               # quality gate result (scores + issues)
    revision_count: int         # how many times it was revised
    max_revision_attempts: int  # revision limit (from config)

    # From the image nodes
    enable_images: bool         # master switch (OFF unless user enables Images)
    image_api_key: str | None   # explicit override (tests/legacy); run_job keeps the key in services.images._JOB_IMAGE_KEYS — never checkpointed
    md_with_placeholders: str   # article with [[IMAGE_n]] markers
    image_specs: list[dict]     # image plans (prompt, size, section)
    final: str                  # complete final blog markdown
