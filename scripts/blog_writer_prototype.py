"""Standalone blog-writer prototype — aligned with the production graph.

Everything that defines the "brain" of the workflow is IMPORTED from the
production codebase so it can never drift:

- Schemas  -> app.graph.schemas   (Task, Plan, ImageSpec, ...)
- State    -> app.graph.state    (GraphState — same keys as production)
- Prompts  -> app.graph.prompts  (PLANNER_SYSTEM, ...)

Only the INFRASTRUCTURE is standalone here: direct ChatOpenAI calls,
langchain-community Tavily search, and google-genai image generation —
no LiteLLM gateway, Redis, or job store required.

Run from the repo root:
    python scripts/blog_writer_prototype.py "Your topic here"
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.graph.prompts import (
    IMAGE_SYSTEM,
    PLANNER_SYSTEM,
    QUALITY_SYSTEM,
    REVISE_SYSTEM,
    RESEARCH_SYSTEM,
    ROUTER_SYSTEM,
    WORKER_SYSTEM,
)
from app.graph.schemas import (
    EvidenceItem,
    EvidencePack,
    GlobalImagePlan,
    ImageSpec,
    Plan,
    QualityResult,
    RouterDecision,
    Task,
)
from app.graph.state import GraphState

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Planner → Workers →
#              Merge → Quality → (Revise*) → Images)
# Schemas, state, and every system prompt are the canonical
# production versions — imported, never duplicated.
# ============================================================

_llm = None


def get_llm() -> ChatOpenAI:
    """Lazy ChatOpenAI client (no credentials needed at import time)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4.1-mini")
    return _llm


# -----------------------------
# Router
# -----------------------------
def router_node(state: State) -> dict:
    decision = get_llm().with_structured_output(RouterDecision).invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    needs_research = decision.needs_research
    queries = decision.queries[:10] or ([state["topic"]] if needs_research else [])

    return {
        "needs_research": needs_research,
        "mode": decision.mode,
        "queries": queries,
        "max_results_per_query": 3,
        "recency_days": recency_days,
    }


def route_after_router(state: GraphState) -> str:
    return "research" if state.get("needs_research") else "planner"


# -----------------------------
# Research (standalone Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore

        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        return [
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
            for r in results or []
        ]
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def research_node(state: GraphState) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, state.get("max_results_per_query", 6)))

    if not raw:
        return {"evidence": []}

    pack = get_llm().with_structured_output(EvidencePack).invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of: {state['as_of']}\n"
                    f"Mode: {state.get('mode')}\n"
                    f"Raw search results:\n{raw}"
                )
            ),
        ]
    )

    evidence = list({e.url: e for e in pack.evidence if e.url}.values())

    if state.get("mode") == "open_book":
        cutoff = date.fromisoformat(state["as_of"]) - timedelta(
            days=int(state.get("recency_days", 7))
        )
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence[:30]}


# -----------------------------
# Planner
# -----------------------------
def planner_node(state: GraphState) -> dict:
    evidence = [item.model_dump() for item in state.get("evidence", [])[:10]]
    plan = get_llm().with_structured_output(Plan).invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {state.get('mode', 'closed_book')}\n"
                    f"As-of: {state['as_of']}\n"
                    f"Evidence:\n{evidence}"
                )
            ),
        ]
    )

    if state.get("mode") == "open_book":
        plan.blog_kind = "news_roundup"
    return {"plan": plan}


# -----------------------------
# Fanout + Worker
# -----------------------------
def fanout(state: GraphState):
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Planner produced no plan.")
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "as_of": state["as_of"],
                "plan": plan.model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in plan.tasks
    ]


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = payload.get("evidence", [])

    evidence_text = "\n".join(
        f"- {item.get('title', '')} | {item.get('url', '')} | "
        f"{item.get('published_at') or 'date:unknown'} | "
        f"{(item.get('snippet') or '')[:500]}"
        for item in evidence[:20]
    )

    section_md = get_llm().invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Mode: {payload.get('mode')}\n"
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
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# -----------------------------
# Merge + Quality + Revise
# -----------------------------
def merge_content(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Cannot merge without a plan.")
    sections = sorted(state.get("sections", []), key=lambda item: item[0])
    if not sections:
        raise ValueError("Workers produced no sections.")
    body = "\n\n".join(md for _, md in sections).strip()
    return {"merged_md": f"# {plan.blog_title}\n\n{body}\n"}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def quality_gate(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Quality gate requires a plan.")

    evidence = [item.model_dump() for item in state.get("evidence", [])[:10]]
    result = get_llm().with_structured_output(QualityResult).invoke(
        [
            SystemMessage(content=QUALITY_SYSTEM),
            HumanMessage(
                content=(
                    f"Plan: {plan.model_dump()}\n"
                    f"Evidence: {evidence[:10]}\n"
                    f"Word count: {_word_count(state.get('merged_md', ''))}\n"
                    f"Article:\n{state.get('merged_md', '')}"
                )
            ),
        ]
    )
    return {"quality": result.model_dump()}


def route_quality(state: GraphState) -> str:
    quality = state.get("quality", {})
    if quality.get("passed"):
        return "images"
    if state.get("revision_count", 0) < state.get("max_revision_attempts", 2):
        return "revise"
    return "images"


def revise_content(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Revision requires a plan.")

    issues = state.get("quality", {}).get("issues", [])
    revised = get_llm().invoke(
        [
            SystemMessage(content=REVISE_SYSTEM),
            HumanMessage(
                content=(
                    f"Plan: {plan.model_dump()}\n"
                    f"Issues: {issues}\n"
                    f"Current content:\n{state['merged_md']}"
                )
            ),
        ]
    ).content.strip()
    return {"merged_md": revised, "revision_count": state.get("revision_count", 0) + 1}


# -----------------------------
# Images (Gemini standalone)
# -----------------------------
def decide_images(state: GraphState) -> dict:
    merged_md = state.get("merged_md", "")
    article = merged_md
    if len(article) > 4800:
        article = article[:4800] + "\n\n[...article truncated...]"

    plan = state.get("plan")
    image_plan = get_llm().with_structured_output(GlobalImagePlan).invoke(
        [
            SystemMessage(content=IMAGE_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind if plan else 'explainer'}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{article}"
                )
            ),
        ]
    )
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [item.model_dump() for item in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """Raw image bytes from Gemini (requires google-genai + GOOGLE_API_KEY)."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None
    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name if name.lower().endswith(".png") else f"{name}.png"


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 _-]", "", text.lower())
    return re.sub(r"\s+", "_", cleaned).strip("_") or "article"


def generate_and_place_images(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Image generation requires a plan.")

    md = state.get("md_with_placeholders") or state.get("merged_md", "")
    specs = state.get("image_specs", []) or []
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    if specs:
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)
        for spec in specs[:3]:
            spec = ImageSpec(**spec) if isinstance(spec, dict) else spec
            placeholder, filename = spec.placeholder, _safe_filename(spec.filename)
            out_path = images_dir / filename

            if not out_path.exists():
                try:
                    out_path.write_bytes(_gemini_generate_image_bytes(spec.prompt))
                except Exception as exc:
                    md = md.replace(
                        placeholder,
                        (
                            f"> **[IMAGE GENERATION FAILED]** {spec.caption}\n>\n"
                            f"> **Alt:** {spec.alt}\n>\n"
                            f"> **Prompt:** {spec.prompt}\n>\n"
                            f"> **Error:** {exc}\n"
                        ),
                    )
                    continue

            md = md.replace(placeholder, f"![{spec.alt}](images/{filename})\n*{spec.caption}*")

    out_path = out_dir / f"{_slug(plan.blog_title)}.md"
    out_path.write_text(md, encoding="utf-8")
    return {"final": md}


# ============================================================
# Build graph
#   merge → quality → (revise loop) → images  (canonical order)
# ============================================================
reducer_graph = StateGraph(GraphState)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("quality", quality_gate)
reducer_graph.add_node("revise", revise_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "quality")
reducer_graph.add_conditional_edges(
    "quality",
    route_quality,
    {"revise": "revise", "images": "decide_images"},
)
reducer_graph.add_edge("revise", "quality")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


g = StateGraph(GraphState)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("planner", planner_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_after_router, {"research": "research", "planner": "planner"})
g.add_edge("research", "planner")
g.add_conditional_edges("planner", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "How transformer attention works"
    result = app.invoke(
        {"topic": topic, "as_of": date.today().isoformat(), "sections": []},
        config={"recursion_limit": 60},
    )
    print(result.get("final", ""))
def router_node(state: State) -> dict:
    decision = get_llm().with_structured_output(RouterDecision).invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
    ])
    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650
    needs_research = decision.needs_research
    queries = decision.queries[:10] or ([state['topic']] if needs_research else [])
    return {
        "needs_research": needs_research,
        "mode": decision.mode,
        "queries": queries,
        "max_results_per_query": 3,
        "recency_days": recency_days,
    }


def route_after_router(state: GraphState) -> str:
    return "research" if state.get("needs_research") else "planner"


# -----------------------------
# Research (standalone Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore

        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        return [
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
            for r in results or []
        ]
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def research_node(state: GraphState) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, state.get("max_results_per_query", 6)))

    if not raw:
        return {"evidence": []}

    pack = get_llm().with_structured_output(EvidencePack).invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of: {state['as_of']}\n"
                    f"Mode: {state.get('mode')}\n"
                    f"Raw search results:\n{raw}"
                )
            ),
        ]
    )

    evidence = list({e.url: e for e in pack.evidence if e.url}.values())

    if state.get("mode") == "open_book":
        cutoff = date.fromisoformat(state["as_of"]) - timedelta(
            days=int(state.get("recency_days", 7))
        )
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence[:30]}


# -----------------------------
# Planner
# -----------------------------
def planner_node(state: GraphState) -> dict:
    evidence = [item.model_dump() for item in state.get("evidence", [])[:10]]
    plan = get_llm().with_structured_output(Plan).invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {state.get('mode', 'closed_book')}\n"
                    f"As-of: {state['as_of']}\n"
                    f"Evidence:\n{evidence}"
                )
            ),
        ]
    )

    if state.get("mode") == "open_book":
        plan.blog_kind = "news_roundup"
    return {"plan": plan}


# -----------------------------
# Fanout + Worker
# -----------------------------
def fanout(state: GraphState):
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Planner produced no plan.")
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state.get("mode", "closed_book"),
                "as_of": state["as_of"],
                "plan": plan.model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in plan.tasks
    ]


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = payload.get("evidence", [])

    evidence_text = "\n".join(
        f"- {item.get('title', '')} | {item.get('url', '')} | "
        f"{item.get('published_at') or 'date:unknown'} | "
        f"{(item.get('snippet') or '')[:500]}"
        for item in evidence[:20]
    )

    section_md = get_llm().invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Mode: {payload.get('mode')}\n"
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
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# -----------------------------
# Merge + Quality + Revise
# -----------------------------
def merge_content(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Cannot merge without a plan.")
    sections = sorted(state.get("sections", []), key=lambda item: item[0])
    if not sections:
        raise ValueError("Workers produced no sections.")
    body = "\n\n".join(md for _, md in sections).strip()
    return {"merged_md": f"# {plan.blog_title}\n\n{body}\n"}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def quality_gate(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Quality gate requires a plan.")

    evidence = [item.model_dump() for item in state.get("evidence", [])[:10]]
    result = get_llm().with_structured_output(QualityResult).invoke(
        [
            SystemMessage(content=QUALITY_SYSTEM),
            HumanMessage(
                content=(
                    f"Plan: {plan.model_dump()}\n"
                    f"Evidence: {evidence[:10]}\n"
                    f"Word count: {_word_count(state.get('merged_md', ''))}\n"
                    f"Article:\n{state.get('merged_md', '')}"
                )
            ),
        ]
    )
    return {"quality": result.model_dump()}


def route_quality(state: GraphState) -> str:
    quality = state.get("quality", {})
    if quality.get("passed"):
        return "images"
    if state.get("revision_count", 0) < state.get("max_revision_attempts", 2):
        return "revise"
    return "images"


def revise_content(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Revision requires a plan.")

    issues = state.get("quality", {}).get("issues", [])
    revised = get_llm().invoke(
        [
            SystemMessage(content=REVISE_SYSTEM),
            HumanMessage(
                content=(
                    f"Plan: {plan.model_dump()}\n"
                    f"Issues: {issues}\n"
                    f"Current content:\n{state['merged_md']}"
                )
            ),
        ]
    ).content.strip()
    return {"merged_md": revised, "revision_count": state.get("revision_count", 0) + 1}


# -----------------------------
# Images (Gemini standalone)
# -----------------------------
def decide_images(state: GraphState) -> dict:
    merged_md = state.get("merged_md", "")
    article = merged_md
    if len(article) > 4800:
        article = article[:4800] + "\n\n[...article truncated...]"

    plan = state.get("plan")
    image_plan = get_llm().with_structured_output(GlobalImagePlan).invoke(
        [
            SystemMessage(content=IMAGE_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind if plan else 'explainer'}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{article}"
                )
            ),
        ]
    )
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [item.model_dump() for item in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """Raw image bytes from Gemini (requires google-genai + GOOGLE_API_KEY)."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None
    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name if name.lower().endswith(".png") else f"{name}.png"


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9 _-]", "", text.lower())
    return re.sub(r"\s+", "_", cleaned).strip("_") or "article"


def generate_and_place_images(state: GraphState) -> dict:
    plan = state.get("plan")
    if plan is None:
        raise ValueError("Image generation requires a plan.")

    md = state.get("md_with_placeholders") or state.get("merged_md", "")
    specs = state.get("image_specs", []) or []
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    if specs:
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)
        for spec in specs[:3]:
            spec = ImageSpec(**spec) if isinstance(spec, dict) else spec
            placeholder, filename = spec.placeholder, _safe_filename(spec.filename)
            out_path = images_dir / filename

            if not out_path.exists():
                try:
                    out_path.write_bytes(_gemini_generate_image_bytes(spec.prompt))
                except Exception as exc:
                    md = md.replace(
                        placeholder,
                        (
                            f"> **[IMAGE GENERATION FAILED]** {spec.caption}\n>\n"
                            f"> **Alt:** {spec.alt}\n>\n"
                            f"> **Prompt:** {spec.prompt}\n>\n"
                            f"> **Error:** {exc}\n"
                        ),
                    )
                    continue

            md = md.replace(placeholder, f"![{spec.alt}](images/{filename})\n*{spec.caption}*")

    out_path = out_dir / f"{_slug(plan.blog_title)}.md"
    out_path.write_text(md, encoding="utf-8")
    return {"final": md}


# ============================================================
# Build graph
#   merge → quality → (revise loop) → images  (canonical order)
# ============================================================
reducer_graph = StateGraph(GraphState)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("quality", quality_gate)
reducer_graph.add_node("revise", revise_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "quality")
reducer_graph.add_conditional_edges(
    "quality",
    route_quality,
    {"revise": "revise", "images": "decide_images"},
)
reducer_graph.add_edge("revise", "quality")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


g = StateGraph(GraphState)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("planner", planner_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_after_router, {"research": "research", "planner": "planner"})
g.add_edge("research", "planner")
g.add_conditional_edges("planner", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "How transformer attention works"
    result = app.invoke(
        {"topic": topic, "as_of": date.today().isoformat(), "sections": []},
        config={"recursion_limit": 60},
    )
    print(result.get("final", ""))

