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
from app.services.llm import invoke_structured, invoke_text, model_candidates
from app.services.search import dedupe_and_filter, search_web

logger = logging.getLogger(__name__)

ROUTER_SYSTEM = """You are the research router for a technical-content workflow.
Choose exactly one mode:
- closed_book: a domain expert can write an accurate, complete article from training knowledge alone. This covers stable concepts, definitions, mathematics, algorithms, architecture patterns, and established tooling (for example: how self-attention works, transformer architecture, REST vs GraphQL, SQL joins, SOLID principles, Big-O analysis).
- hybrid: evergreen topic where current versions, benchmarks, ecosystem shifts, or recent tooling materially improve the article (for example: "production RAG stack in 2026", "best vector databases today").
- open_book: volatile information - news, pricing, releases, policies, weekly changes.
Bias toward closed_book. Research costs latency and quota; use it only when being wrong without sources is likely. When in doubt between closed_book and hybrid for a conceptual topic, choose closed_book.
Set needs_research=true only for hybrid/open_book, with 3-8 precise search queries.
Do not claim research has happened in this step."""


def router_node(state: GraphState) -> dict:
    override = state.get("research_mode") or "auto"

    if override == "skip":
        # User explicitly opted out of web research; skip the router LLM call.
        return {
            "needs_research": False,
            "mode": "closed_book",
            "queries": [],
            "max_results_per_query": APP_CONFIG.max_research_results,
            "recency_days": 3650,
        }

    if override == "force":
        # User explicitly wants web evidence regardless of topic stability.
        return {
            "needs_research": True,
            "mode": "hybrid",
            "queries": [state["topic"]],
            "max_results_per_query": APP_CONFIG.max_research_results,
            "recency_days": 45,
        }

    decision = invoke_structured(
        RouterDecision,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
        operation="router",
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    needs_research = decision.mode != "closed_book"
    queries = decision.queries[: APP_CONFIG.max_research_queries]
    if needs_research and not queries:
        # Keep the graph useful even if the model forgets to emit queries.
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


RESEARCH_SYSTEM = """You are a research evidence normalizer.
Convert raw search results into EvidenceItem objects.
Rules:
- Keep only useful, non-empty URLs.
- Prefer authoritative sources.
- Never invent dates. Use null when uncertain.
- Keep snippets concise.
- Do not infer unsupported facts from a search result."""


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
    )

    evidence = dedupe_and_filter(
        pack.evidence,
        state["as_of"],
        state["recency_days"],
        strict_recency=state.get("mode") == "open_book",
    )
    return {"evidence": evidence[:30]}


PLANNER_SYSTEM = """You are a senior technical content architect creating an in-depth article plan.
Create exactly 6 tasks. Each task targets 350-600 words; the SUM of all target_words must be between 2300 and 3200.
Keep every string field concise so the JSON stays compact: titles under 60 chars, goals under 160 chars, bullets short phrases (one line each).
Article structure requirements:
- Task 1: a compelling introduction - hook, why the topic matters now, what the reader will learn.
- Middle tasks: deep dives covering mechanisms, tradeoffs, common pitfalls, or a concrete worked example.
- Include one task with a step-by-step walkthrough or annotated code walkthrough when the topic allows code.
- One middle task should compare approaches/tools (a Markdown table will be rendered there).
- Task 6: practical takeaways plus a 3-5 question FAQ and next steps.
Every task needs 3-5 substantive bullets (each bullet expands into 60-120 words later).
Avoid filler sections. Depth over breadth.
For open_book mode, use blog_kind=news_roundup and do not invent events.
Mark tasks that need current evidence with requires_research=true and requires_citations=true."""


def planner_node(state: GraphState) -> dict:
    evidence = [item.model_dump() for item in state.get("evidence", [])[:10]]
    memory_note = state.get("memory_note") or ""
    memory_block = (
        f"\nPreviously generated articles by this user (do NOT repeat these angles or titles; keep tone consistent):\n{memory_note}\n"
        if memory_note.strip()
        else ""
    )
    plan = invoke_structured(
        Plan,
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {state.get('mode', 'closed_book')}\n"
                    f"As-of: {state['as_of']}\n"
                    f"{memory_block}"
                    f"Evidence:\n{evidence}"
                )
            ),
        ],
        operation="planner",
    )

    if state.get("mode") == "open_book":
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
                "plan": plan.model_dump(),
                "evidence": evidence,
                "memory_note": state.get("memory_note", ""),
            },
        )
        for task in plan.tasks
    ]


WORKER_SYSTEM = """Write exactly one Markdown section for a long-form technical article.
Start with `## <section title>`.

Depth requirements (this is what separates a great article from a thin one):
- Open with 1-2 framing sentences that connect to the article's flow.
- Explain WHY before HOW: mechanisms, intuition, and consequences - not just definitions.
- Include at least one of per section: a concrete example or mini-scenario, an annotated code snippet, a small Markdown comparison table, or a numbered walkthrough.
- Expand every bullet into its own sub-section or paragraph cluster (60-120 words each). Never list bullets back verbatim.
- Use short paragraphs (2-4 sentences), bold key terms, and occasional sub-headings (###) in longer sections.
- LENGTH FLOOR: write at least the target word count; never below 85% of it. If you finish early, deepen the weakest bullet instead of stopping.

Integrity rules:
- Never invent specific current events, company claims, model releases, pricing, or policy claims.
- For current claims, cite only the supplied approved evidence URLs.
- If requires_citations=true and evidence does not support a current claim, say it is not established by the provided sources.
- If requires_code=true, include a minimal useful code snippet.
Output only the section Markdown."""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = payload.get("evidence", [])

    # Spread parallel workers across the model fallback chain so concurrent
    # sections draw from separate provider rate-limit buckets instead of all
    # hammering one pool at the same instant.
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

    evidence_text = "\n".join(
        f"- {item.get('title', '')} | {item.get('url', '')} | "
        f"{item.get('published_at') or 'date:unknown'} | "
        f"{(item.get('snippet') or '')[:500]}"
        for item in evidence[:20]
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
                    f"Mode: {payload.get('mode')}\n"
                    f"{memory_block}"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"requires_research={task.requires_research}\n"
                    f"requires_citations={task.requires_citations}\n"
                    f"requires_code={task.requires_code}\n"
                    f"Bullets:\n- " + "\n- ".join(task.bullets) +
                    f"\nApproved evidence:\n{evidence_text}"
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


QUALITY_SYSTEM = """Review the generated technical article against its plan and approved evidence.
Be strict about:
- unsupported current claims,
- missing required bullets or thin sections that ignore planned depth,
- LENGTH: the article must be at least 90% of the summed plan target_words. A short article fails completeness.
- missing citations where required,
- obvious structural/instruction-following failures (no intro hook, no examples/tables/code where planned, no FAQ/takeaways ending).
Return QualityResult only. Do not rewrite the article."""


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def quality_gate(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Quality gate requires a plan.")

    evidence = [item.model_dump() for item in state.get("evidence", [])[:10]]
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
    )

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
            SystemMessage(
                content="""Revise the Markdown article to fix the listed quality issues.
When an issue says the article is too short, EXPAND the thin sections substantially:
deepen explanations (mechanisms, tradeoffs, pitfalls), add concrete examples, comparison
tables or annotated code, and grow every underdeveloped bullet into full paragraphs.
Preserve accurate content and overall section structure; never shrink the article.
Do not add unsupported facts or citations. Output only the revised Markdown."""
            ),
            HumanMessage(
                content=(
                    f"Plan: {plan.model_dump()}\n"
                    f"Issues: {issues}\n"
                    f"Current content:\n{state['merged_md']}"
                )
            ),
        ],
        operation="revision",
    )
    return {
        "merged_md": revised,
        "revision_count": state.get("revision_count", 0) + 1,
    }


IMAGE_SYSTEM = """You are a technical visual editor.
Choose at most 3 diagrams that materially improve understanding.
Prefer architecture, workflow, lifecycle, or comparison visuals.
Avoid decoration.
If no diagram is useful, return the original Markdown and an empty image list.
Use placeholders exactly [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]]."""


def decide_images(state: GraphState) -> dict:
    if not state.get("enable_images", True):
        return {"merged_md": state["merged_md"], "image_specs": []}

    plan = invoke_structured(
        GlobalImagePlan,
        [
            SystemMessage(content=IMAGE_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Article:\n{state['merged_md']}"
                )
            ),
        ],
        operation="image_planning",
    )
    return {
        "merged_md": plan.md_with_placeholders,
        "image_specs": [item.model_dump() for item in plan.images],
    }


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

    md = state.get("merged_md", "")
    specs = state.get("image_specs", [])
    job_id = state.get("job_id", "local")
    asset_dir = Path("images") / re.sub(r"[^a-zA-Z0-9_-]", "_", job_id)
    asset_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs[:3]:
        filename = _safe_filename(spec["filename"])
        path = asset_dir / filename
        try:
            if not path.exists():
                generate_image(spec["prompt"], path, spec.get("aspect_ratio", "16:9"))
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
