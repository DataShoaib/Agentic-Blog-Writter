from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from app.config import APP_CONFIG
from app.graph.schemas import (
    EvidencePack,
    EvidenceItem,
    GlobalImagePlan,
    Plan,
    QualityResult,
    RouterDecision,
    Task,
)
from app.graph.state import GraphState
from app.services.citations import validate_citations
from app.services.images import generate_image
from app.services.llm import (
    LLMGatewayError,
    invoke_structured,
    invoke_text,
    model_candidates,
)
from app.services.search import dedupe_and_filter, search_web

logger = logging.getLogger(__name__)

# Canonical prompts live in app/graph/prompts.py (single source of truth).
from app.graph.prompts import (  # noqa: E402
    IMAGE_SYSTEM,
    PLANNER_SYSTEM,
    QUALITY_SYSTEM,
    REVISE_SYSTEM,
    ROUTER_SYSTEM,
    WORKER_SYSTEM,
    RESEARCH_SYSTEM,
)


def router_node(state: GraphState) -> dict:
    decision = invoke_structured(
        RouterDecision,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
        operation="router",
        preferred_model=state.get("model"),
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    needs_research = decision.needs_research
    queries = decision.queries[: APP_CONFIG.max_research_queries]
    if needs_research and not queries:
        queries = [state["topic"]]

    logger.info(
        "router mode=%s needs_research=%s reason=%s",
        decision.mode, needs_research, decision.reason[:120],
    )

    return {
        "needs_research": needs_research,
        "mode": decision.mode,
        "queries": queries,
        "max_results_per_query": min(
            decision.max_results_per_query, APP_CONFIG.max_research_results
        ),
        "recency_days": recency_days,
    }


def route_after_router(state: GraphState) -> str:
    return "research" if state.get("needs_research") else "planner"


def research_node(state: GraphState) -> dict:
    raw: list[dict] = []
    for query in state.get("queries", [])[: APP_CONFIG.max_research_queries]:
        raw.extend(search_web(query, state.get("max_results_per_query", 6)))

    if not raw:
        return {"evidence": []}

    pack = invoke_structured(
        EvidencePack,
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of: {state['as_of']}\n"
                    f"Mode: {state.get('mode')}\n"
                    f"Raw search results:\n{[item.model_dump() for item in raw]}"
                )
            ),
        ],
        operation="research_synthesis",
        preferred_model=state.get("model"),
    )

    evidence = dedupe_and_filter(
        pack.evidence,
        state["as_of"],
        state["recency_days"],
        strict_recency=state.get("mode") == "open_book",
    )
    return {"evidence": evidence[:30]}


def planner_node(state: GraphState) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = [item.model_dump() for item in state.get("evidence", [])]
    memory_note = state.get("memory_note") or ""
    memory_block = (
        f"\nPreviously generated articles by this user (do NOT repeat these angles or titles; keep tone consistent):\n{memory_note}\n"
        if memory_note.strip()
        else ""
    )
    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = invoke_structured(
        Plan,
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state.get('recency_days', 3650)})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{evidence[:16]}"
                )
            ),
        ],
        operation="planner",
        preferred_model=state.get("model"),
    )

    if forced_kind:
        plan.blog_kind = "news_roundup"
    total_words = sum(task.target_words for task in plan.tasks)
    logger.info(
        "planner tasks=%d total_target_words=%d title=%s",
        len(plan.tasks), total_words, plan.blog_title[:80],
    )
    return {"plan": plan}


def fanout(state: GraphState):
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Planner produced no plan.")

    evidence = [item.model_dump() for item in state.get("evidence", [])]
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "as_of": state["as_of"],
                "recency_days": state.get("recency_days", 3650),
                "plan": plan.model_dump(),
                "evidence": evidence,
                "memory_note": state.get("memory_note", ""),
                "model": state.get("model"),
            },
        )
        for task in plan.tasks
    ]


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    # Spread parallel workers across the model fallback chain so concurrent
    # sections draw from separate provider rate-limit buckets instead of all
    # hammering one pool at the same instant. A user-selected model wins; the
    # round-robin spread is the fallback when none is given.
    explicit_model = payload.get("model")
    if explicit_model:
        preferred = explicit_model if explicit_model in model_candidates() else None
    else:
        candidates = model_candidates()
        preferred = candidates[(task.id - 1) % len(candidates)] if candidates else None
    if task.id > 1:
        time.sleep(min(APP_CONFIG.worker_start_delay_seconds * (task.id - 1), 3.0))

    memory_note = payload.get("memory_note") or ""
    memory_block = (
        f"\nUser's earlier articles (avoid repeating angles/titles, keep tone consistent):\n{memory_note}\n"
        if memory_note.strip()
        else ""
    )

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section = invoke_text(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n"
                    f"{memory_block}\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ],
        operation=f"worker_{task.id}",
        preferred_model=preferred,
    )
    logger.info("worker_%d used preferred model %s (%d chars)", task.id, preferred, len(section))
    return {"sections": [(task.id, section)]}


def merge_content(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Cannot merge without a plan.")

    sections = sorted(state.get("sections", []), key=lambda item: item[0])
    if not sections:
        raise ValueError("Workers produced no sections.")

    body = "\n\n".join(markdown for _, markdown in sections).strip()
    return {"merged_md": f"# {plan.blog_title}\n\n{body}\n"}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def quality_gate(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Quality gate requires a plan.")

    evidence = [item.model_dump() for item in state.get("evidence", [])[:10]]
    try:
        result = invoke_structured(
            QualityResult,
            [
                SystemMessage(content=QUALITY_SYSTEM),
                HumanMessage(
                    content=(
                        f"Plan: {plan.model_dump()}\n"
                        f"Evidence: {evidence[:10]}\n"
                        f"Content:\n{state['merged_md']}"
                    )
                ),
            ],
            operation="quality_gate",
            preferred_model=state.get("model"),
        )
    except LLMGatewayError as exc:
        # The merged article already exists; losing it to an LLM quota error
        # would waste the whole run. Degrade: keep the content, note the skip,
        # and let the (LLM-free) citation check below still contribute.
        logger.warning("quality gate skipped: %s", exc)
        result = QualityResult(
            passed=True,
            factuality_score=0.75,
            completeness_score=0.75,
            citation_score=0.75,
            issues=["Quality gate skipped: LLM quota exhausted; article was not LLM-reviewed."],
        )
        skipped = True
    else:
        skipped = False

    citations_required = any(task.requires_citations for task in plan.tasks)
    citation_score, citation_issues = validate_citations(
        state["merged_md"], evidence, citations_required=citations_required
    )
    issues = list(dict.fromkeys([*result.issues, *citation_issues]))[:12]
    result.citation_score = min(result.citation_score, citation_score)

    # Hard length enforcement: a thin article must fail and trigger revision.
    planned_words = sum(task.target_words for task in plan.tasks)
    actual_words = _word_count(state["merged_md"])
    if actual_words < 0.9 * planned_words:
        issues.append(
            f"Article too short: {actual_words} words vs planned {planned_words}. "
            "Expand thin sections with deeper explanations, examples, tables, or code."
        )
        result.completeness_score = min(result.completeness_score, 0.5)
        result.factuality_score = min(result.factuality_score, 0.85)
    logger.info(
        "quality gate words=%d planned=%d scores(f=%s,c=%s,cit=%s) issues=%d",
        actual_words, planned_words,
        round(result.factuality_score, 2), round(result.completeness_score, 2),
        round(result.citation_score, 2), len(issues),
    )

    result.issues = issues[:12]
    result.passed = bool(
        result.passed
        and result.factuality_score >= 0.75
        and result.completeness_score >= 0.75
        and result.citation_score >= 0.75
        and not any(issue.startswith("Article too short") for issue in result.issues)
    )
    if skipped:
        # Quota is already gone; a revision attempt would fail too. Ship the
        # article with the skip note instead of burning the job.
        result.passed = True
    return {"quality": result.model_dump()}


def route_quality(state: GraphState) -> str:
    quality = state.get("quality", {})
    if quality.get("passed"):
        return "images"
    if state.get("revision_count", 0) < state.get(
        "max_revision_attempts", APP_CONFIG.max_revision_attempts
    ):
        return "revise"
    return "images"


def revise_content(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Revision requires a plan.")

    issues = state.get("quality", {}).get("issues", [])
    revised = invoke_text(
        [
            SystemMessage(content=REVISE_SYSTEM),
            HumanMessage(
                content=(
                    f"Plan: {plan.model_dump()}\n"
                    f"Issues: {issues}\n"
                    f"Current content:\n{state['merged_md']}"
                )
            ),
        ],
        operation="revision",
        preferred_model=state.get("model"),
    )
    return {
        "merged_md": revised,
        "revision_count": state.get("revision_count", 0) + 1,
    }


# Image planning operates on a compact section outline, NOT the full article.
# Sending the whole article forced the planning model to echo it back — and
# when the article exceeded the token budget the echo came back TRUNCATED,
# which silently cut every generated blog in half. We now let the planner only
# decide WHICH sections deserve a diagram (via the ImageSpec.section anchor)
# and inject the [[IMAGE_n]] placeholder into the full merged article below.
_IMAGE_PLANNING_MAX_SECTIONS = 40


def _section_outline(md: str) -> list[tuple[str, str]]:
    """Compact per-section summary for image planning.

    Returns (heading, leading body text) pairs. Only this outline is sent to
    the LLM, so the response can never overwrite or truncate the article.
    """
    outline: list[tuple[str, str]] = []
    current: tuple[str, list[str]] | None = None

    def _flush() -> None:
        nonlocal current
        if current is not None:
            heading, body = current
            outline.append((heading.strip(), " ".join(body).strip()))
            current = None

    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            _flush()
            current = (stripped[3:].strip("#").strip(), [])
        elif stripped and current is not None:
            current[1].append(stripped[:180])
    _flush()
    return outline[:_IMAGE_PLANNING_MAX_SECTIONS]


def _heading_insert_index(md: str, heading: str) -> int | None:
    """Return the line index after the best-matching ``## <heading>``."""
    want = re.sub(r"[^a-z0-9]+", " ", heading.lower()).strip()
    for idx, line in enumerate(md.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("## "):
            continue
        candidate = stripped[3:].strip("#").strip().lower()
        candidate = re.sub(r"[^a-z0-9]+", " ", candidate).strip()
        if candidate and candidate == want:
            return idx + 1
    return None


def _inject_placeholder(md: str, spec: dict) -> str:
    """Insert one ``[[IMAGE_n]]`` placeholder into the FULL article.

    Anchors to the matching section heading; falls back to just before the
    Sources section, then to the end of the article. The article itself is
    never truncated or rewritten.
    """
    placeholder = spec.get("placeholder", "")
    if not placeholder:
        return md

    lines = md.splitlines()
    target = _heading_insert_index(md, spec.get("section", ""))
    if target is None:
        target = next(
            (idx for idx, line in enumerate(lines) if line.strip().startswith("## Sources")),
            len(lines),
        )
    while target < len(lines) and not lines[target].strip():
        target += 1
    lines.insert(target, placeholder)
    lines.insert(target, "")
    return "\n".join(lines)


def decide_images(state: GraphState) -> dict:
    if not state.get("enable_images", True):
        return {"md_with_placeholders": state["merged_md"], "image_specs": []}

    merged_md = state["merged_md"]
    outline = _section_outline(merged_md)
    plan = state.get("plan")
    image_plan = invoke_structured(
        GlobalImagePlan,
        [
            SystemMessage(content=IMAGE_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind if plan else 'explainer'}\n"
                    f"Topic: {state['topic']}\n\n"
                    f"Section outline:\n{outline}\n\n"
                    "Decide which sections (max 3) genuinely benefit from a "
                    "technical diagram. For each, set ImageSpec.section to the "
                    "EXACT heading text from the outline. Only image specs are "
                    "used; md_with_placeholders stays empty."
                )
            ),
        ],
        operation="image_planning",
        preferred_model=state.get("model"),
    )

    specs = [item.model_dump() for item in image_plan.images]
    md_with_placeholders = merged_md
    for spec in specs:
        md_with_placeholders = _inject_placeholder(md_with_placeholders, spec)
    return {"md_with_placeholders": md_with_placeholders, "image_specs": specs}


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 _-]", "", text.lower())
    return re.sub(r"\s+", "_", cleaned).strip("_") or "article"


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    if not name.lower().endswith(".png"):
        name = f"{name}.png"
    return name


def generate_and_place_images(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Image generation requires a plan.")

    md = state.get("md_with_placeholders") or state.get("merged_md", "")
    specs = state.get("image_specs", [])
    job_id = state.get("job_id", "local")
    asset_dir = Path("images") / re.sub(r"[^a-zA-Z0-9_-]", "_", job_id)
    asset_dir.mkdir(parents=True, exist_ok=True)

    # ImageSpec.size (pixel notation) mapped onto Gemini's aspect ratios.
    size_to_aspect = {"1024x1024": "1:1", "1024x1536": "9:16", "1536x1024": "16:9"}

    for spec in specs[:3]:
        filename = _safe_filename(spec["filename"])
        path = asset_dir / filename
        try:
            if not path.exists():
                generate_image(
                    spec["prompt"],
                    path,
                    size_to_aspect.get(spec.get("size", "1024x1024"), "16:9"),
                )
            image_url = f"/assets/images/{asset_dir.name}/{filename}"
            replacement = f"![{spec['alt']}]({image_url})\n*{spec['caption']}*"
        except Exception as exc:
            logger.warning(
                "image generation failed for job=%s file=%s: %s: %s",
                job_id, filename, type(exc).__name__, str(exc)[:200],
            )
            replacement = (
                f"> **Image unavailable:** {spec.get('caption', '')}\n>\n"
                f"> Error: `{type(exc).__name__}`"
            )
        md = md.replace(spec["placeholder"], replacement)

    evidence = state.get("evidence", [])
    if evidence and "## Sources" not in md:
        source_lines = ["## Sources", ""]
        for item in evidence[:30]:
            title = item.title if hasattr(item, "title") else item.get("title", "Source")
            url = item.url if hasattr(item, "url") else item.get("url", "")
            if url:
                source_lines.append(f"- [{title or 'Source'}]({url})")
        md = md.rstrip() + "\n\n" + "\n".join(source_lines) + "\n"

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    job_id = state.get("job_id", "local")
    output_path = output_dir / f"{re.sub(r'[^a-zA-Z0-9_-]', '_', job_id)}.md"
    output_path.write_text(md, encoding="utf-8")
    return {"final": md}
