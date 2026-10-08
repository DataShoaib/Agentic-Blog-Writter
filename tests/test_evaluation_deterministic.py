"""Minimal unit tests for the deterministic evaluators and judge schemas.

Only deterministic code is tested here — no API keys required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.deterministic import (
    _dedup_sections,
    check_citation_allowlist,
    check_workflow_success,
    compute_cost,
    extract_cited_urls,
    p95,
)
from app.evaluation.judges import JudgeResult

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "app" / "evaluation" / "golden_cases.json")
    .read_text(encoding="utf-8")
)


# ------------------------------------------------------------ golden set -----
def test_golden_dataset_has_exactly_4_valid_cases():
    cases = GOLDEN["cases"]
    assert len(cases) == 4
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 4
    for case in cases:
        assert case["query"].strip()
        assert isinstance(case["requires_research"], bool)
        assert case["expected_requirements"]
        assert case["tests"]


# ----------------------------------------------------- workflow success ------
def test_workflow_success_fails_on_empty_final():
    result = check_workflow_success({"final": "   "})
    assert not result.passed


def test_workflow_success_passes_with_content():
    result = check_workflow_success({"final": "# Blog\n\nbody"})
    assert result.passed


def test_workflow_success_reports_graph_error():
    result = check_workflow_success(None, error="ValueError: boom")
    assert not result.passed
    assert any("boom" in d for d in result.detail)


def test_dedup_sections_keeps_first_output_per_task():
    """Quota-storm duplicate fan-out writes collapse to one per task."""
    state = {"sections": [(1, "a"), (1, "a-retry"), (2, "b"), (2, "b-retry")]}
    assert _dedup_sections(state)["sections"] == [(1, "a"), (2, "b")]


def test_dedup_sections_leaves_unique_outputs_alone():
    state = {"sections": [(1, "a"), (2, "b")]}
    assert _dedup_sections(state)["sections"] == [(1, "a"), (2, "b")]


# ---------------------------------------------------- citation allow-list ----
def test_citation_outside_allowlist_fails():
    final_md = "Claim one [Source](https://evil.example.com/page)."
    result = check_citation_allowlist(final_md, ["https://trusted.example.com/a"])
    assert not result.passed


def test_citation_from_allowed_host_passes():
    final_md = "Claim [Source](https://trusted.example.com/anything-else)."
    result = check_citation_allowlist(final_md, ["https://trusted.example.com/a"])
    assert result.passed


def test_extract_cited_urls_dedupes():
    md = "[a](https://x.com/1) [b](https://x.com/1) [c](https://y.com/2)"
    assert extract_cited_urls(md) == ["https://x.com/1", "https://y.com/2"]


# ------------------------------------------------------- latency & cost ------
def test_p95_nearest_rank():
    assert p95([]) is None
    values = list(range(1, 21))  # 1..20 -> nearest-rank P95 = 19
    assert p95(values) == 19


def test_compute_cost_requires_known_usage():
    assert compute_cost([]) is None
    unknown = [{"model": "mystery-model", "prompt_tokens": 10, "completion_tokens": 10}]
    assert compute_cost(unknown) is None  # never invent cost
    known = [{"model": "gemini/gemini-2.5-flash", "prompt_tokens": 1_000_000, "completion_tokens": 0}]
    assert compute_cost(known) == 0.30


# ------------------------------------------------------- judge schemas -------
def _valid_judge_payload(**overrides):
    payload = {
        "score": 0.9,
        "passed": True,
        "critical_errors": [],
        "reasoning": "Everything checks out.",
    }
    payload.update(overrides)
    return payload


def test_judge_result_accepts_valid_output():
    result = JudgeResult.model_validate(_valid_judge_payload())
    assert result.score == 0.9


@pytest.mark.parametrize(
    "bad",
    [
        {"score": 1.5},               # out of range
        {"score": -0.1},              # out of range
        {"score": "high"},            # wrong type
        {"reasoning": ""},            # too short
        {"reasoning": None},          # missing value
        {"critical_errors": "oops"},  # wrong type
    ],
)
def test_malformed_judge_output_fails_pydantic_validation(bad):
    with pytest.raises(ValidationError):
        JudgeResult.model_validate(_valid_judge_payload(**bad))


def test_missing_required_judge_fields_fail_validation():
    with pytest.raises(ValidationError):
        JudgeResult.model_validate({"score": 0.5})
