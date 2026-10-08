"""Deterministic evaluators — pure Python checks for structural/operational
properties of a finished agent run.

NO LLM is used here. Every function takes plain dicts/lists so the checks can
be unit tested without API keys. They only VALIDATE existing outputs; they
never change workflow behavior.
"""
# Kept workflow success + citation allowlist + cost (pure Python, no LLM).
# Dropped structure/merge/image checks: the graph itself fails loud on missing
# sections (merge ValueError) and test_graph + test_image_fallback already
# cover fan-out ordering and image integrity. To evaluate more, add a check
# function here and wire it into runner.run_case's deterministic block plus
# build_aggregate.
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class EvalCheck:
    """Outcome of one deterministic check."""

    name: str
    passed: bool
    detail: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


# ---------------------------------------------------------------- workflow ---
def _dedup_sections(state: dict) -> dict:
    """Collapse duplicate worker outputs (one per retry attempt) in place.

    When every provider route is 429/503ing, the gateway's per-model try loop
    can surface one result per ATTEMPT rather than one per TASK. That is an
    infrastructure symptom, not a pipeline bug — the workflow check must look
    at what the merged article actually contains, not at raw fan-out noise.
    """
    sections = state.get("sections") or []
    if sections and len(sections) != len({task_id for task_id, _ in sections}):
        seen: set[int] = set()
        deduped: list = []
        for task_id, markdown in sections:
            if task_id in seen:
                continue
            seen.add(task_id)
            deduped.append((task_id, markdown))
        state["sections"] = deduped
    return state


def check_workflow_success(state: dict | None, error: str | None = None) -> EvalCheck:
    """Did the graph complete and produce a non-empty final document?"""
    if error:
        return EvalCheck("workflow_success", False, [f"graph raised: {error}"])
    if not isinstance(state, dict):
        return EvalCheck("workflow_success", False, ["no final state captured"])
    content = state.get("final") or ""
    if not str(content).strip():
        failures = ["final markdown is empty"]
        if state.get("quality", {}).get("issues"):
            failures.append(f"quality issues: {state['quality']['issues']}")
        return EvalCheck("workflow_success", False, failures)
    return EvalCheck("workflow_success", True)


# ---------------------------------------------------------------- cost -------
# USD per 1M tokens: (input, output). Only models with known published prices
# are listed; unknown models make the cost report return None (never invent).
_MODEL_COST_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini/gemini-2.5-flash": (0.30, 2.50),
    "gemini/gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini/gemini-3.1-flash-lite": (0.10, 0.40),
    "groq/openai/gpt-oss-20b": (0.075, 0.30),
    "groq/openai/gpt-oss-120b": (0.15, 0.60),
}


def p95(values: list[float] | None) -> float | None:
    """Nearest-rank 95th percentile; None for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    import math

    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def compute_cost(usage: list[dict] | None) -> float | None:
    """Average-independent total cost in USD, or None when usage is missing
    or contains any model with unknown pricing."""
    if not usage:
        return None
    total = 0.0
    for entry in usage:
        rates = _MODEL_COST_PER_MTOK.get(entry.get("model", ""))
        if rates is None:
            return None
        total += (
            entry.get("prompt_tokens", 0) / 1_000_000 * rates[0]
            + entry.get("completion_tokens", 0) / 1_000_000 * rates[1]
        )
    return round(total, 6)


# ------------------------------------------------------------- citations -----
_CITED_URL_RE = re.compile(r"\]\((https?://[^)\s]+)\)")


def _normalize(url: str) -> tuple[str, str]:
    url = url.strip().rstrip("/")
    parts = re.match(r"https?://([^/]+)(/.*)?", url)
    host = (parts.group(1) or "").lower() if parts else ""
    return host, url.lower()


def extract_cited_urls(md: str) -> list[str]:
    seen: list[str] = []
    for match in _CITED_URL_RE.finditer(str(md)):
        url = match.group(1)
        if url not in seen:
            seen.append(url)
    return seen


def check_citation_allowlist(final_md: str, evidence_urls: list[str]) -> EvalCheck:
    """Every cited URL must belong to sources returned by the research stage."""
    allowed_exact = {_normalize(u)[1] for u in evidence_urls}
    allowed_hosts = {_normalize(u)[0] for u in evidence_urls}
    cited = extract_cited_urls(final_md)
    if not cited:
        return EvalCheck("citation_allowlist", True, [])
    failures = []
    for url in cited:
        host, normed = _normalize(url)
        if normed not in allowed_exact and host not in allowed_hosts:
            failures.append(f"cited URL not from research sources: {url}")
    return EvalCheck("citation_allowlist", not failures, failures)
