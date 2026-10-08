"""LLM-as-a-judge evaluators for the existing agent pipeline.

Every judge:
- sends a focused prompt through the shared gateway (`invoke_structured`),
- receives a Pydantic-validated ``JudgeResult`` (free-form text is NEVER parsed),
- only PRODUCES a result — it never controls the LangGraph workflow.

PASS/FAIL is applied by the runner's Python logic using configurable
thresholds; the LLM's own ``passed`` field is not blindly trusted.
"""
# Kept 3 judges (router, factuality, final-success) + 3 thresholds; dropped 4
# overlapping judges (groundedness/plan/completeness/citation-semantic) since
# the deterministic allowlist + graph quality gate already cover them. To
# evaluate more: add a judge fn here, a threshold below, and a dispatch
# branch in runner.run_case.

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
    "factuality": 0.75,
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


def judge_factuality(topic: str, evidence: list[dict], final_md: str) -> JudgeResult:
    """Factuality judge: permissive mode when research returned nothing.

    When evidence is empty, the workers wrote from trained knowledge (there is
    nothing to ground against), so judging becomes "does this contradict
    well-established knowledge?" rather than "is every claim backed by a
    retrieved URL?". Only hard contradictions or invented specifics count —
    penalising ungrounded-but-plausible prose would make every no-evidence
    case unwinnable by construction (a harness artifact, not a model bug).
    """
    has_evidence = bool(evidence)
    evidence_lines = "\n".join(f"- {e.get('url', '')}" for e in evidence[:20]) or (
        "(no external evidence was retrieved — the blog was written from "
        "the model's own trained knowledge)"
    )
    strictness = (
        "Every major factual claim must be traceable to one of the allowed "
        "sources. Claims with no supporting source go in critical_errors."
        if has_evidence
        else "There are NO retrieved sources to check against, so judge ONLY "
        "against well-established public knowledge: flag claims that are "
        "clearly wrong, self-contradictory, or invent over-specific facts "
        "(exact statistics, named studies, precise dates/versions) that a "
        "reader could not verify. Do NOT penalise claims merely for lacking "
        "a citation — ungrounded-but-plausible prose is expected here, not "
        "a failure. Only concrete contradictions count as critical_errors."
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
                    f"Grounding rule: {strictness}\n"
                    "score 1.0 = no suspicious factual claims; 0.0 = many wrong or "
                    "invented claims. Put invented/wrong claims in critical_errors. "
                    "Return JudgeResult only."
                )
            ),
        ],
        operation="eval_factuality",
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



