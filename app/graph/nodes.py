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
    REVISION_OUTPUT_TOKENS,
    TEXT_OUTPUT_TOKENS,
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
    # Hard floor: the planner sometimes emits thin 120-200 word sections even
    # when told not to. Clamp every task to >=280 words here so workers always
    # get a deep budget and the quality gate's per-section check can pass on
    # the first attempt. Never silently keep a shallow plan.
    for _t in plan.tasks:
        if _t.target_words < 280:
            logger.warning("planner task %d thin budget (%d) - clamped to 280", _t.id, _t.target_words)
            _t.target_words = 280
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
    evidence_lines = []
    for e in evidence[:20]:
        snippet = (e.snippet or "").strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320].rstrip() + "…"
        evidence_lines.append(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            + (f" | {snippet}" if snippet else "")
        )
    evidence_text = "\n".join(evidence_lines) or "(no evidence — write evergreen content, do not invent sources)"

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
                    f"Target words: {task.target_words} (MINIMUM {task.target_words} words — hitting this floor matters more than staying concise)\n"
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
        # 380 target words ≈ 500 tokens, but padding (bullets, citations,
        # formatting) plus model verbosity needs headroom: target_words * 4
        # tokens/word heuristic, floored so even small sections can't clip.
        max_tokens=max(TEXT_OUTPUT_TOKENS, int(task.target_words * 4)),
    )
    words = _word_count(section)
    paras = _substantial_paragraphs(section)
    logger.info(
        "worker_%d target=%d words=%d paras=%d used preferred model %s (%d chars)",
        task.id, task.target_words, words, paras, preferred, len(section),
    )
    # Retry when the section is short OR shallow: a 250-word two-paragraph
    # stub hits the word floor but still reads like an AI summary. The
    # quality gate enforces the same rule deterministically, so repair here.
    if words < 0.85 * task.target_words or words < 250 or paras < 2:
        logger.warning(
            "worker_%d short (%d/%d words) — retrying with explicit expansion ask",
            task.id, words, task.target_words,
        )
        section = invoke_text(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Your previous draft was only {words} words vs the required "
                        f"minimum {task.target_words}. EXPAND it now: keep the same "
                        f"'## {task.title}' heading, deepen every bullet with "
                        f"mechanisms, trade-offs, pitfalls and concrete examples, "
                        f"and add comparison tables or annotated code where the "
                        f"bullets allow it. Output only the expanded section.\n\n"
                        f"Previous draft:\n{section}\n\n"
                        f"Bullets:{bullets_text}\n\n"
                        f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                    )
                ),
            ],
            operation=f"worker_{task.id}_expand",
            preferred_model=preferred,
            max_tokens=max(TEXT_OUTPUT_TOKENS, int(task.target_words * 4)),
        )
        logger.info(
            "worker_%d expanded retry words=%d (target=%d)",
            task.id, _word_count(section), task.target_words,
        )
    return {"sections": [(task.id, section)]}


def merge_content(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Cannot merge without a plan.")

    sections = sorted(state.get("sections", []), key=lambda item: item[0])
    if not sections:
        raise ValueError("Workers produced no sections.")

    # Guard against silent section loss: every planned task must have exactly
    # one worker output. LangSmith showed all workers succeeding yet the final
    # blog missing sections — a dropped/duplicate write here would do that.
    planned_ids = [task.id for task in plan.tasks]
    got_ids = [task_id for task_id, _ in sections]
    missing = [tid for tid in planned_ids if tid not in got_ids]
    if missing:
        raise ValueError(
            f"Workers missing sections for planned tasks {missing} "
            f"(planned={planned_ids} got={got_ids})."
        )
    if len(set(got_ids)) != len(got_ids):
        raise ValueError(f"Duplicate worker outputs for tasks: {got_ids}.")

    body = "\n\n".join(markdown for _, markdown in sections).strip()
    # Repair: re-split CONCATENATED body on planned headings.
    # A worker echoing a sibling heading would otherwise overwrite it.
    _body_all = body
    _by_id: dict[int, str] = {}
    for _task in plan.tasks:
        _pat = re.compile(
            r"^##\s+" + re.escape(_task.title.strip()) + r"\s*$",
            re.MULTILINE,
        )
        _mm = list(_pat.finditer(_body_all))
        if _mm:
            _st = _mm[0].start()
            _fol = [
                m2.start()
                for _t2 in plan.tasks
                if _t2.id != _task.id
                for m2 in re.finditer(
                    r"^##\s+" + re.escape(_t2.title.strip()) + r"\s*$",
                    _body_all[_st:],
                    re.MULTILINE,
                )
            ]
            _en = _st + min(_fol) if _fol else len(_body_all)
            _by_id[_task.id] = _body_all[_st:_en].strip()
    for _tid, _md in sections:
        _by_id.setdefault(_tid, _md)
    if len(_by_id) == len(planned_ids):
        _ordered = [_by_id[tid] for tid in planned_ids]
        _rebody = "\n\n".join(_ordered).strip()
        if _rebody:
            body = _rebody
    merged = f"# {plan.blog_title}\n\n{body}\n"
    per_section_words = {task_id: _word_count(markdown) for task_id, markdown in sections}
    targets = {task.id: task.target_words for task in plan.tasks}
    thin = [
        f"task={tid} words={per_section_words.get(tid, 0)}/{targets.get(tid, '?')}"
        for tid in planned_ids
        if per_section_words.get(tid, 0) < 0.85 * targets.get(tid, 1)
    ]
    logger.info(
        "merge tasks=%d sections=%d merged_chars=%d merged_words=%d%s",
        len(planned_ids), len(sections), len(merged), _word_count(merged),
        f" thin=[{', '.join(thin)}]" if thin else "",
    )
    return {"merged_md": merged}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _substantial_paragraphs(section_md: str) -> int:
    """Count body paragraphs with >=40 words (headings excluded).

    Shared by the worker retry trigger, merge repair and the quality gate so
    a two-line stub can never pass as a finished section anywhere.
    """
    body = re.sub(r"^#{1,6}\s+.*$", "", section_md or "", flags=re.MULTILINE)
    return sum(1 for p in re.split(r"\n\s*\n", body) if len(re.findall(r"\b\w+\b", p)) >= 40)


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
    # Cap don't crush: one stray unapproved URL shouldn't nuke an otherwise
    # cited article to near-zero. Floor at 0.6 when at least one approved
    # citation exists; only a total absence fails the gate.
    from app.services.citations import approved_urls, citation_urls as _cited_urls

    _approved = approved_urls(evidence)
    _cited = _cited_urls(state["merged_md"])
    if _cited & _approved:
        citation_score = max(citation_score, 0.6)
    result.citation_score = min(result.citation_score, citation_score)

    # Hard length enforcement: a thin article must fail and trigger revision.
    # Threshold 0.85 (with an 85-90% warn band) so one slightly-short section
    # out of 7 doesn't condemn the whole run; per-section word logging in
    # merge_content identifies the real culprit.
    # Deterministic per-section depth: every planned section body
    # must be >=200 words with >=2 substantial paragraphs.
    _bodies: dict[int, str] = {}
    for _task in plan.tasks:
        _pat = re.compile(
            r"^##\s+" + re.escape(_task.title.strip()) + r"\s*$",
            re.MULTILINE,
        )
        _mm = list(_pat.finditer(state["merged_md"]))
        if _mm:
            _st = _mm[0].start()
            _fol = [
                m2.start()
                for _t2 in plan.tasks
                if _t2.id != _task.id
                for m2 in re.finditer(
                    r"^##\s+" + re.escape(_t2.title.strip()) + r"\s*$",
                    state["merged_md"][_st:],
                    re.MULTILINE,
                )
            ]
            _en = _st + min(_fol) if _fol else len(state["merged_md"])
            _bodies[_task.id] = state["merged_md"][_st:_en]
    for _task in plan.tasks:
        _w = _word_count(_bodies.get(_task.id, ""))
        _pp = _substantial_paragraphs(_bodies.get(_task.id, ""))
        if _w < 200 or _pp < 2:
            issues.append(
                f"Section '{_task.title}' too thin: {_w} words, {_pp} substantial "
                "paragraphs (need >=200 words and >=2 paragraphs of >=40 words). "
                "Expand with mechanisms, trade-offs, examples, tables or code."
            )
            result.completeness_score = min(result.completeness_score, 0.5)
    planned_words = sum(task.target_words for task in plan.tasks)
    actual_words = _word_count(state["merged_md"])
    if actual_words < 0.85 * planned_words:
        issues.append(
            f"Article too short: {actual_words} words vs planned {planned_words}. "
            "Expand thin sections with deeper explanations, examples, tables, or code."
        )
        result.completeness_score = min(result.completeness_score, 0.5)
        result.factuality_score = min(result.factuality_score, 0.85)
    elif actual_words < 0.9 * planned_words:
        issues.append(
            f"Article slightly short: {actual_words} words vs planned {planned_words} "
            "(within tolerance — consider expanding the thinnest section)."
        )
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


def _targeted_expand(state: GraphState, plan: Plan, thin_titles: list[str]) -> str | None:
    """Expand ONLY thin sections, splice back, keep good ones byte-for-byte."""
    try:
        thin = [t for t in plan.tasks if t.title in thin_titles]
        bullets = "\n".join(f"- {t.title}: " + "; ".join(t.bullets) for t in thin)
        expanded = invoke_text(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        "Expand ONLY the sections listed below. For EACH, output "
                        "a '## <exact title>' block of at least 300 words with "
                        "2+ substantial paragraphs (mechanisms, trade-offs, "
                        "concrete examples, pitfalls). Keep titles EXACT. "
                        "Output only the expanded section blocks.\n\n"
                        f"Sections to expand:\n{bullets}"
                    )
                ),
            ],
            operation="revision_targeted",
            preferred_model=state.get("model"),
            max_tokens=max(REVISION_OUTPUT_TOKENS // 2, 6000),
        )
        spliced = _splice_sections(state["merged_md"], list(plan.tasks), expanded)
        res = quality_gate({**state, "merged_md": spliced})
        if res["quality"]["passed"]:
            logger.info("targeted revision fixed %s", thin_titles)
            return spliced
        logger.info("targeted revision insufficient; full rewrite next")
    except Exception as exc:
        logger.warning("targeted revision failed (%s); full rewrite next", exc)
    return None


def _splice_sections(merged_md: str, tasks: list, expanded_md: str) -> str:
    """Replace named thin section bodies with expanded blocks."""
    out = merged_md
    for task in tasks:
        pat = re.compile(
            r"^##\s+" + re.escape(task.title.strip()) + r"\s*$",
            re.MULTILINE,
        )
        new_m = list(pat.finditer(expanded_md))
        if not new_m:
            continue
        new_start = new_m[0].start()
        new_fol = [
            m2.start()
            for t2 in tasks
            if t2.id != task.id
            for m2 in re.finditer(
                r"^##\s+" + re.escape(t2.title.strip()) + r"\s*$",
                expanded_md[new_start:],
                re.MULTILINE,
            )
        ]
        new_end = new_start + min(new_fol) if new_fol else len(expanded_md)
        new_block = expanded_md[new_start:new_end].strip()
        if _word_count(new_block) < 200:
            continue
        old_m = list(pat.finditer(out))
        if not old_m:
            continue
        old_start = old_m[0].start()
        old_fol = [
            m2.start()
            for t2 in tasks
            if t2.id != task.id
            for m2 in re.finditer(
                r"^##\s+" + re.escape(t2.title.strip()) + r"\s*$",
                out[old_start:],
                re.MULTILINE,
            )
        ]
        old_end = old_start + min(old_fol) if old_fol else len(out)
        out = (out[:old_start] + new_block + out[old_end:]).strip() + "\n"
    return out


def revise_content(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Revision requires a plan.")

    issues = state.get("quality", {}).get("issues", [])
    thin_titles = []
    for _issue in issues:
        for _t in plan.tasks:
            if _t.title and _t.title in _issue and _t.title not in thin_titles:
                thin_titles.append(_t.title)
    if thin_titles:
        _spliced = _targeted_expand(state, plan, thin_titles)
        if _spliced is not None:
            return {
                "merged_md": _spliced,
                "revision_count": state.get("revision_count", 0) + 1,
            }
    # Full-article rewrite: needs a much larger output budget than a single
    # section, otherwise the LLM response is cut mid-article and trailing
    # planned sections vanish from the final blog. On ANY truncation/shrink
    # keep the pre-revision merged article instead of shipping a shorter one.
    original = state["merged_md"]
    try:
        revised = invoke_text(
            [
                SystemMessage(content=REVISE_SYSTEM),
                HumanMessage(
                    content=(
                        f"Plan: {plan.model_dump()}\n"
                        f"Issues: {issues}\n"
                        f"Current content:\n{original}"
                    )
                ),
            ],
            operation="revision",
            preferred_model=state.get("model"),
            max_tokens=REVISION_OUTPUT_TOKENS,
        )
    except LLMGatewayError as exc:
        # The merged article already exists; losing it to a quota error would
        # waste the whole run. Keep the original and ship it.
        logger.warning("revision skipped: %s", exc)
        return {
            "merged_md": original,
            "revision_count": state.get("revision_count", 0) + 1,
        }
    if _word_count(revised) < 0.9 * _word_count(original):
        logger.warning(
            "revision shrank article (%d -> %d words); keeping original",
            _word_count(original), _word_count(revised),
        )
        revised = original
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
            # Never leak raw provider errors into the published article — drop
            # the placeholder and note the caption instead.
            replacement = f"*{spec.get('caption', '')}*" if spec.get("caption") else ""
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
