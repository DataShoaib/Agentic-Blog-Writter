"""Minimal unit tests for the deterministic evaluators and judge schemas.

Only deterministic code is tested here — no API keys required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.deterministic import (
    check_citation_allowlist,
    check_image_integrity,
    check_merge_completeness_and_order,
    check_structural_integrity,
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
def test_golden_dataset_has_exactly_8_valid_cases():
    cases = GOLDEN["cases"]
    assert len(cases) == 8
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 8
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


# ------------------------------------------------ structural integrity -------
def _plan(tasks):
    return {"plan": {"tasks": tasks}, "sections": [], "merged_md": ""}


def _task(tid):
    return {"id": tid, "title": f"T{tid}", "target_words": 100}


def test_task_count_outside_5_to_9_fails():
    tasks = [_task(i) for i in range(1, 4)]  # only 3 tasks
    result = check_structural_integrity(_plan(tasks))
    assert not result.passed
    assert any("outside 5-9" in d for d in result.detail)


def test_duplicate_task_ids_fail():
    tasks = [
        {"id": 1, "title": "A", "target_words": 100},
        {"id": 1, "title": "B", "target_words": 100},
        {"id": 2, "title": "C", "target_words": 100},
        {"id": 3, "title": "D", "target_words": 100},
        {"id": 4, "title": "E", "target_words": 100},
    ]
    result = check_structural_integrity(_plan(tasks))
    assert not result.passed
    assert any("duplicate task IDs" in d for d in result.detail)


def test_missing_worker_output_fails():
    tasks = [_task(i) for i in range(1, 6)]
    state = _plan(tasks)
    state["sections"] = [(1, "## T1\ntext")]
    result = check_structural_integrity(state)
    assert not result.passed
    assert any("no worker output" in d for d in result.detail)


def test_unexpected_worker_task_id_fails():
    tasks = [_task(i) for i in range(1, 6)]
    state = _plan(tasks)
    state["sections"] = [(i, f"## T{i}\ntext") for i in range(1, 7)]  # extra id 6
    result = check_structural_integrity(state)
    assert not result.passed
    assert any("unplanned task IDs" in d for d in result.detail)


# ------------------------------------------------------ merge integrity ------
def test_merge_order_mismatch_fails():
    tasks = [
        {"id": 1, "title": "Intro", "target_words": 100},
        {"id": 2, "title": "Body", "target_words": 100},
        {"id": 3, "title": "End", "target_words": 100},
        {"id": 4, "title": "More", "target_words": 100},
        {"id": 5, "title": "Extra", "target_words": 100},
    ]
    # merged_md contains headings in WRONG order (T2's heading before T1's)
    merged = "## Body\n\nb\n\n## Intro\n\na\n\n## End\n\nc\n\n## More\n\nd\n\n## Extra\n\ne"
    state = {
        "plan": {"tasks": tasks},
        "sections": [(1, "## Intro\n\na"), (2, "## Body\n\nb")],
        "merged_md": merged,
    }
    result = check_merge_completeness_and_order(state)
    assert not result.passed
    assert any("task_id order" in d for d in result.detail)


def test_merge_in_correct_order_passes():
    tasks = [{"id": i, "title": f"S{i}", "target_words": 100} for i in range(1, 6)]
    merged = "\n\n".join(f"## S{i}\n\ntext {i}" for i in range(1, 6))
    state = {
        "plan": {"tasks": tasks},
        "sections": [(i, f"## S{i}\n\ntext {i}") for i in range(1, 6)],
        "merged_md": merged,
    }
    result = check_merge_completeness_and_order(state)
    assert result.passed, result.detail


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


# ------------------------------------------------------- image integrity -----
def test_unresolved_placeholder_fails():
    result = check_image_integrity(
        "text [[IMAGE_1]] more", [{"placeholder": "[[IMAGE_1]]", "filename": "f.png"}]
    )
    assert not result.passed
    assert any("unresolved placeholders" in d for d in result.detail)


def test_resolved_placeholder_matches_plan_passes():
    md = "![alt](/assets/images/job1/f.png)\n*caption*"
    result = check_image_integrity(
        md, [{"placeholder": "[[IMAGE_1]]", "filename": "f.png"}]
    )
    assert result.passed, result.detail


def test_more_than_three_images_fails():
    specs = [{"placeholder": f"[[IMAGE_{i}]]", "filename": f"f{i}.png"} for i in range(1, 5)]
    result = check_image_integrity("clean text without placeholders", specs)
    assert not result.passed
    assert any("maximum is 3" in d for d in result.detail)


def test_graceful_fallback_without_link_is_accepted():
    md = "> **[IMAGE GENERATION FAILED]** caption"
    result = check_image_integrity(
        md, [{"placeholder": "[[IMAGE_1]]", "filename": "f.png"}]
    )
    assert result.passed


# ------------------------------------------------------- latency & cost ------
def test_p95_nearest_rank():
    assert p95([]) is None
    values = list(range(1, 21))  # 1..20 -> nearest-rank P95 = 19
    assert p95(values) == 19


def test_compute_cost_requires_known_usage():
    assert compute_cost([]) is None
    unknown = [{"model": "mystery-model", "prompt_tokens": 10, "completion_tokens": 10}]
    assert compute_cost(unknown) is None  # never invent cost
    known = [{"model": "openai/gpt-oss-20b", "prompt_tokens": 1_000_000, "completion_tokens": 0}]
    assert compute_cost(known) == 0.07


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
