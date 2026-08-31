"""Deterministic evaluators — pure Python checks for structural/operational
properties of a finished agent run.

NO LLM is used here. Every function takes plain dicts/lists so the checks can
be unit tested without API keys. They only VALIDATE existing outputs; they
never change workflow behavior.
"""
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


# ------------------------------------------------------------ structure ------
def check_structural_integrity(state: dict) -> EvalCheck:
    """Plan vs worker outputs vs merged document consistency."""
    failures: list[str] = []
    plan = state.get("plan")
    if not plan:
        return EvalCheck("structural_integrity", False, ["missing plan"])
    if hasattr(plan, "model_dump"):
        plan = plan.model_dump()
    tasks = plan.get("tasks", [])

    count = len(tasks)
    if not 5 <= count <= 9:
        failures.append(f"task count {count} outside 5-9")

    ids = [t.get("id") for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        failures.append(f"duplicate task IDs: {duplicates}")

    sections = [(int(tid), md) for tid, md in state.get("sections", [])]
    planned_ids = set(ids)
    worker_ids = {tid for tid, _ in sections}
    missing = planned_ids - worker_ids
    if missing:
        failures.append(f"planned tasks with no worker output: {sorted(missing)}")
    unexpected = worker_ids - planned_ids
    if unexpected:
        failures.append(f"worker outputs for unplanned task IDs: {sorted(unexpected)}")

    return EvalCheck("structural_integrity", not failures, failures)


# ------------------------------------------------- merged document order -----
_DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"


def _flex_pattern(needle: str) -> re.Pattern | None:
    """Case/dash/whitespace-tolerant regex for locating headings."""
    words = str(needle).split()
    if not words:
        return None
    dash_class = "[" + "".join(re.escape(c) for c in _DASH_CHARS) + "\\-]"
    parts = []
    for word in words:
        pieces = []
        for char in word:
            if char == "-":
                pieces.append(dash_class)
            else:
                pieces.append(re.escape(char))
        parts.append("".join(pieces))
    return re.compile(r"[\s]*".join(parts), re.IGNORECASE)


def _flex_find(haystack: str, needle: str) -> int | None:
    pattern = _flex_pattern(needle)
    if pattern is None:
        return None
    match = pattern.search(str(haystack))
    return match.start() if match else None


def _section_signature(md: str) -> str:
    """First non-empty line of a worker output (used to locate it in merge)."""
    for line in str(md).splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""


def check_merge_completeness_and_order(state: dict) -> EvalCheck:
    """Every planned section must appear in merged_md, in task_id order.

    Matching tolerates trivial formatting drift (unicode dashes, letter case,
    extra whitespace) between a planner's section title and the heading the
    worker actually emitted; genuinely absent sections still fail.
    """
    plan = state.get("plan")
    if not plan:
        return EvalCheck("merge_integrity", False, ["missing plan"])
    if hasattr(plan, "model_dump"):
        plan = plan.model_dump()
    tasks = plan.get("tasks", [])
    merged = state.get("merged_md") or ""
    sections = [(int(tid), md) for tid, md in state.get("sections", [])]

    failures: list[str] = []
    positions: dict[int, int] = {}
    for task in tasks:
        pos = _flex_find(merged, f"## {task.get('title', '')}")
        if pos is None:
            failures.append(f"merged output missing section '## {task.get('title', '')}'")
        else:
            positions[task["id"]] = pos

    for tid, md in sections:
        marker = _section_signature(md)
        if marker and _flex_find(merged, marker.lstrip("#").strip()) is None:
            failures.append(f"worker output for task {tid} missing from merged_md")

    heading_order = [i for i, _ in sorted(positions.items(), key=lambda kv: kv[1])]
    expected_order = [task["id"] for task in tasks if task["id"] in positions]
    if heading_order != expected_order:
        failures.append(
            f"merged sections out of task_id order: {heading_order} != {expected_order}"
        )
    return EvalCheck("merge_integrity", not failures, failures)


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


# ---------------------------------------------------------------- images -----
_PLACEHOLDER_RE = re.compile(r"\[\[IMAGE_[1-3]\]\]")
_IMAGE_LINK_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def check_image_integrity(final_md: str, image_specs: list[dict]) -> EvalCheck:
    """Image plan / placeholders / generated links must stay consistent.

    The existing graceful fallback (no image link when generation fails) is
    ACCEPTED; we only verify image links that DO appear map back to the plan.
    """
    specs = image_specs or []
    failures: list[str] = []

    if len(specs) > 3:
        failures.append(f"{len(specs)} images planned; maximum is 3")

    placeholders = [s.get("placeholder", "") for s in specs]
    if len(placeholders) != len(set(placeholders)):
        failures.append("duplicate placeholders in image plan")

    unresolved = _PLACEHOLDER_RE.findall(str(final_md))
    if unresolved:
        failures.append(f"unresolved placeholders remain: {unresolved}")

    planned_names: set[str] = set()
    for spec in specs:
        raw = re.sub(r"[^a-zA-Z0-9._-]", "_", spec.get("filename", ""))
        planned_names.add(raw.lower())
        planned_names.add((raw + ".png").lower())
    for url in _IMAGE_LINK_RE.findall(str(final_md)):
        filename = url.rsplit("/", 1)[-1].lower()
        # Skip Sources-section external links; only /assets/ images count.
        if "/assets/" in url and filename not in planned_names:
            failures.append(f"image link does not match image plan: {url}")

    return EvalCheck("image_integrity", not failures, failures)



