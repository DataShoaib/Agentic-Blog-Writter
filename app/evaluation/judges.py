"""LLM-as-a-judge evaluators for the existing agent pipeline.

Every judge:
- sends a focused prompt through the shared gateway (`invoke_structured`),
- receives a Pydantic-validated ``JudgeResult`` (free-form text is NEVER parsed),
- only PRODUCES a result — it never controls the LangGraph workflow.

PASS/FAIL is applied by the runner's Python logic using configurable
thresholds; the LLM's own ``passed`` field is not blindly trusted.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm import invoke_structured


class JudgeResult(BaseModel):
    """Reusable structured result for every semantic judge."""

    score: float = Field(ge=0, le=1, description="0.0 = total failure, 1.0 = perfect")
    passed: bool = Field(description="Judge's own opinion; Python re-applies thresholds")
    critical_errors: list[str] = Field(default_factory=list, max_length=10)
    reasoning: str = Field(min_length=5, max_length=2000)


# Configurable pass thresholds applied by runner-side Python logic.
JUDGE_THRESHOLDS: dict[str, float] = {
    "router_correctness": 0.8,
    "evidence_groundedness": 0.75,
    "plan_adherence": 0.75,
    "factuality": 0.75,
    "completeness": 0.75,
    "citation_correctness": 0.75,
    "final_task_success": 0.8,
}

JUDGE_SYSTEM = (
    "You are a strict, objective evaluator of AI-generated technical content. "
    "Grade ONLY what is present in the provided materials. Do not assume facts "
    "you cannot verify from the inputs. Score honestly; do not be generous."
)

_MAX_MD_CHARS = 10000  # keep judge prompts inside the token budget


def _clip(text: str) -> str:
    text = str(text)
    if len(text) <= _MAX_MD_CHARS:
        return text
    return text[:_MAX_MD_CHARS] + "\n...[truncated]"


def effective_pass(judge_name: str, result: JudgeResult) -> bool:
    """Python-side PASS/FAIL from thresholds + critical errors.

    Deliberately ignores the LLM's own `passed` field.
    """
    threshold = JUDGE_THRESHOLDS.get(judge_name, 0.75)
    return result.score >= threshold and not result.critical_errors


_CITATION_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def citation_contexts(final_md: str, limit: int = 12) -> list[str]:
    """Claim text immediately surrounding each markdown citation."""
    contexts: list[str] = []
    for match in _CITATION_RE.finditer(str(final_md)):
        start = max(match.start() - 240, 0)
        context = final_md[start : match.end()].strip()
        contexts.append(context.replace("\n", " ")[:400])
        if len(contexts) >= limit:
            break
    return contexts


def judge_router_correctness(
    query: str,
    router_decision: dict,
    research_happened: bool,
    evidence_count: int,
) -> JudgeResult:
    research_summary = (
        f"research was executed and returned {evidence_count} evidence items"
        if research_happened
        else "research was SKIPPED"
    )
    result = invoke_structured(
        JudgeResult,
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    "Judge whether the router made the CORRECT research/no-research "
                    "decision for this request.\n\n"
                    f"User request: {query}\n"
                    f"Router decision: {router_decision}\n"
                    f"What actually happened: {research_summary}\n\n"
                    "Rules:\n"
                    "- Requests about current events, recent releases, latest "
                    "versions, pricing or market data REQUIRE research.\n"
                    "- Stable evergreen concepts do NOT require research.\n"
                    "- score 1.0 = clearly correct decision, 0.5 = arguable, "
                    "0.0 = clearly wrong.\n"
                    "Return JudgeResult only."
                )
            ),
        ],
        operation="eval_router",
    )
    return result


def judge_evidence_groundedness(
    topic: str, evidence: list[dict], claims_material: str
) -> JudgeResult:
    evidence_lines = "\n".join(
        f"- {(e.get('title') or '')[:120]} | {e.get('url', '')}" for e in evidence[:20]
    ) or "(no evidence retrieved)"
    result = invoke_structured(
        JudgeResult,
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    "Are the important claims in this material SUPPORTED by the "
                    "retrieved evidence below? General common knowledge does not "
                    "need a supporting source; specific named facts about products, "
                    "versions, numbers, events DO.\n\n"
                    f"Topic: {topic}\n\nEvidence retrieved:\n{evidence_lines}\n\n"
                    f"Material to check:\n{_clip(claims_material)}\n\n"
                    "score 1.0 = every important claim supported/verifiable; "
                    "0.5 = roughly half supported; 0.0 = mostly unsupported. "
                    "List unsupported major claims as critical_errors. "
                    "Return JudgeResult only."
                )
            ),
        ],
        operation="eval_groundedness",
    )
    return result


def judge_plan_adherence(plan: dict, final_md: str) -> JudgeResult:
    task_list = "\n".join(
        f"{t['id']}. {t['title']} (target ~{t['target_words']} words)"
        for t in plan["tasks"]
    )
    result = invoke_structured(
        JudgeResult,
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    "Does the generated blog FOLLOW the planner's intended tasks?\n\n"
                    f"Planned tasks:\n{task_list}\n\n"
                    f"Generated blog:\n{_clip(final_md)}\n\n"
                    "Check: all planned sections appear, section order matches plan "
                    "order, per-section depth is proportional to target words. "
                    "Missing planned sections are critical errors. "
                    "Return JudgeResult only."
                )
            ),
        ],
        operation="eval_plan_adherence",
    )
    return result


def judge_factuality(topic: str, evidence: list[dict], final_md: str) -> JudgeResult:
    evidence_lines = "\n".join(f"- {e.get('url', '')}" for e in evidence[:20]) or (
        "(no external evidence was retrieved)"
    )
    result = invoke_structured(
        JudgeResult,
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    "Are important factual claims in this blog correct or plausibly "
                    "supported? Flag anything that contradicts well-established "
                    "knowledge or looks hallucinated.\n\n"
                    f"Topic: {topic}\nAllowed sources from research:\n"
                    f"{evidence_lines}\n\nBlog:\n{_clip(final_md)}\n\n"
                    "score 1.0 = no suspicious factual claims; 0.0 = many wrong or "
                    "invented claims. Put invented/wrong claims in critical_errors. "
                    "Return JudgeResult only."
                )
            ),
        ],
        operation="eval_factuality",
    )
    return result


def judge_completeness(
    query: str, expected_requirements: list[str], final_md: str
) -> JudgeResult:
    requirements = "\n".join(f"- {r}" for r in expected_requirements)
    result = invoke_structured(
        JudgeResult,
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    "Does the final blog satisfy the important requirements of the "
                    "ORIGINAL request?\n\n"
                    f"Original request: {query}\n\nRequirements that must hold:\n"
                    f"{requirements}\n\n"
                    f"Blog:\n{_clip(final_md)}\n\n"
                    "Any unmet requirement is a critical error with its number. "
                    "Return JudgeResult only."
                )
            ),
        ],
        operation="eval_completeness",
    )
    return result


def judge_citation_correctness(
    citation_contexts_list: list[str], evidence: list[dict]
) -> JudgeResult:
    evidence_lines = "\n".join(
        f"- {e.get('url', '')} | {(e.get('snippet') or '')[:200]}" for e in evidence[:20]
    ) or "(no evidence retrieved)"
    context_text = "\n---\n".join(citation_contexts_list) or "(no citations found)"
    result = invoke_structured(
        JudgeResult,
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    "For each cited passage below, does the cited source actually "
                    "support the claim it is attached to?\n\n"
                    f"Evidence available (URL | snippet):\n{evidence_lines}\n\n"
                    f"Cited passages:\n{context_text}\n\n"
                    "If there are NO citations but claims clearly needed them, score "
                    "low. Each mismatched citation is a critical error. "
                    "Return JudgeResult only."
                )
            ),
        ],
        operation="eval_citations",
    )
    return result


def judge_final_task_success(query: str, constraints: dict, final_md: str) -> JudgeResult:
    result = invoke_structured(
        JudgeResult,
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    "Did the final blog successfully satisfy the user's request as a "
                    "whole?\n\n"
                    f"User request: {query}\nStated constraints: {constraints}\n\n"
                    f"Blog:\n{_clip(final_md)}\n\n"
                    "Consider usefulness, coverage of the ask, audience/tone fit and "
                    "length sanity. Missing explicit deliverables are critical errors. "
                    "Return JudgeResult only."
                )
            ),
        ],
        operation="eval_final_success",
    )
    return result



