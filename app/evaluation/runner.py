"""Evaluation runner for the existing agent.

For each golden case:
  1. load case -> 2. run the EXISTING graph unchanged -> 3. capture outputs
  -> 4. deterministic checks -> 5. applicable LLM judges (Pydantic-validated)
  -> 6. store per-case JSON + one aggregate summary.

Run:  python -m app.evaluation.runner
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from app.config import get_secrets
from app.evaluation import judges as J
from app.evaluation import deterministic as D
from app.graph.graph import build_graph

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
GOLDEN_PATH = HERE / "golden_cases.json"


def load_golden_cases() -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    if len(cases) != 8:
        raise ValueError(f"expected exactly 8 golden cases, found {len(cases)}")
    return cases


def _state_to_dict(state: dict) -> dict:
    """Normalize graph state (Pydantic models -> plain dicts) for checks."""
    out = dict(state)
    plan = out.get("plan")
    if plan is not None and hasattr(plan, "model_dump"):
        out["plan"] = plan.model_dump()
    evidence = out.get("evidence") or []
    out["evidence"] = [
        e.model_dump() if hasattr(e, "model_dump") else dict(e) for e in evidence
    ]
    return out


def run_case(graph, case: dict) -> dict:
    """Run the existing agent on one golden case and evaluate it.

    Latency is intentionally NOT captured here: it is already tracked per-node
    and per-run by LangSmith tracing, so re-measuring it in the evaluation
    layer would be redundant (see app/observability/tracing.py).
    """
    query = case["query"]

    error: str | None = None
    state: dict = {}
    try:
        final_state = graph.invoke(
            {
                "topic": query,
                "as_of": date.today().isoformat(),
                "research_mode": "auto",
                "sections": [],
                "evidence": [],
                "revision_count": 0,
                "enable_images": True,
                "job_id": case["id"],
            },
            config={
                "recursion_limit": 80,
                "configurable": {"thread_id": f"eval-{case['id']}"},
            },
        )
        state = _state_to_dict(final_state or {})
    except Exception as exc:  # keep evaluating even if the run exploded
        error = f"{type(exc).__name__}: {exc}"

    evidence_urls = [e.get("url", "") for e in state.get("evidence", [])]
    # The workflow's own routing decision IS the record of what executed: the
    # graph invokes research iff needs_research is set. Evidence can still come
    # back empty (search-provider outage / recency filter) without research
    # having been skipped, so evidence-count must NOT gate this flag.
    research_happened = bool(state.get("needs_research"))

    # ------------------------------------------------ deterministic checks ---
    workflow_ok = D.check_workflow_success(state, error=error)
    structural = D.check_structural_integrity(state)
    merge_integrity = D.check_merge_completeness_and_order(state)
    allowlist = D.check_citation_allowlist(
        state.get("final") or "", evidence_urls
    )
    image_integrity = D.check_image_integrity(
        state.get("final") or "", state.get("image_specs") or []
    )

    # ------------------------------------------------------ LLM judge runs ---
    semantic: dict[str, dict] = {}
    judge_errors: list[str] = []
    router_decision = {
        "mode": state.get("mode"),
        "queries": state.get("queries", []),
    }
    plan = state.get("plan") or {}
    final_md = state.get("final") or ""

    def record_judge(name: str, fn, *args) -> None:
        try:
            result: J.JudgeResult = fn(*args)
        except Exception as exc:
            judge_errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return
        semantic[name] = {
            "score": result.score,
            "passed": J.effective_pass(name, result),
            "critical_errors": result.critical_errors,
            "reasoning": result.reasoning,
        }

    tests = set(case["tests"])
    if error is None:
        if "router_correctness" in tests:
            record_judge(
                "router_correctness",
                J.judge_router_correctness,
                query, router_decision, research_happened, len(evidence_urls),
            )
        if "evidence_groundedness" in tests and research_happened:
            record_judge(
                "evidence_groundedness",
                J.judge_evidence_groundedness,
                query, state["evidence"], final_md,
            )
        if "plan_adherence" in tests and plan:
            record_judge(
                "plan_adherence",
                J.judge_plan_adherence,
                plan, final_md,
            )
        if "factuality" in tests:
            record_judge(
                "factuality",
                J.judge_factuality,
                query, state["evidence"], final_md,
            )
        if "completeness" in tests:
            record_judge(
                "completeness",
                J.judge_completeness,
                query, case["expected_requirements"], final_md,
            )
        if "citation_correctness" in tests:
            record_judge(
                "citation_correctness",
                J.judge_citation_correctness,
                J.citation_contexts(final_md), state["evidence"],
            )
        if "final_task_success" in tests:
            record_judge(
                "final_task_success",
                J.judge_final_task_success,
                query, case["constraints"], final_md,
            )

    # Dataset-level router ground truth (not LLM-decided): the case's expected
    # research requirement must match what the graph actually did.
    router_matches_ground_truth = (
        bool(state.get("needs_research")) == bool(case["requires_research"])
        if error is None
        else False
    )

    all_checks = [workflow_ok, structural, merge_integrity, allowlist, image_integrity]
    deterministic_failed = [c.as_dict() for c in all_checks if not c.passed]

    semantic_all_passed = bool(semantic) and all(s["passed"] for s in semantic.values())
    case_passed = (
        error is None
        and workflow_ok.passed
        and structural.passed
        and merge_integrity.passed
        and allowlist.passed
        and image_integrity.passed
        and router_matches_ground_truth
        and semantic_all_passed
    )

    return {
        "case_id": case["id"],
        "input": {"query": query, **case.get("constraints", {})},
        "expected": {
            "requires_research": case["requires_research"],
            "requirements": case["expected_requirements"],
        },
        "deterministic": {
            "workflow_success": workflow_ok.as_dict(),
            "structural_integrity": structural.as_dict(),
            "merge_integrity": merge_integrity.as_dict(),
            "citation_allowlist": allowlist.as_dict(),
            "image_integrity": image_integrity.as_dict(),
            "router_ground_truth_match": router_matches_ground_truth,
            "failures": deterministic_failed,
        },
        "semantic_judges": semantic,
        "judge_errors": judge_errors,
        "router_decision": router_decision,
        "research_executed": research_happened,
        "evidence_count": len(evidence_urls),
        "task_count": len((plan or {}).get("tasks", [])),
        "word_count": len(final_md.split()) if final_md else 0,
        "latency_sec": latency_sec,
        "passed_overall": case_passed,
        "error": error,
    }


# ------------------------------------------------------------ aggregates -----
SEMANTIC_KEYS = [
    "router_correctness",
    "evidence_groundedness",
    "plan_adherence",
    "factuality",
    "completeness",
    "citation_correctness",
    "final_task_success",
]


def build_aggregate(results: list[dict]) -> dict:
    """Aggregate scores from ACTUAL per-case results. Nothing is invented:
    metrics with no measured samples are reported as null."""
    n = len(results)
    semantic: dict[str, dict] = {}
    for key in SEMANTIC_KEYS:
        entries = [r["semantic_judges"][key] for r in results if key in r["semantic_judges"]]
        semantic[key] = {
            "runs": len(entries),
            "avg_score": round(sum(e["score"] for e in entries) / len(entries), 3)
            if entries
            else None,
            "passed": sum(1 for e in entries if e["passed"]),
        }

    latencies = [r["latency_sec"] for r in results if r.get("error") is None]
    deterministic_summary = {
        name: sum(
            1
            for r in results
            if r["deterministic"].get(name, {}).get("passed", False)
        )
        for name in (
            "workflow_success",
            "structural_integrity",
            "merge_integrity",
            "citation_allowlist",
            "image_integrity",
        )
    }
    return {
        "total_cases": n,
        "cases_passed_overall": sum(1 for r in results if r["passed_overall"]),
        "semantic": semantic,
        "deterministic": deterministic_summary,
        "router_ground_truth_match": sum(
            1 for r in results if r["deterministic"]["router_ground_truth_match"]
        ),
        "p95_latency_sec": D.p95(latencies),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "note_cost": "average cost reported only when actual token usage was captured",
    }


def main() -> int:
    from app.services.llm import get_recorded_token_usage

    if not get_secrets().groq_api_key:
        print("GROQ_API_KEY missing -- cannot run LLM judges or the agent.")
        return 2

    cases = load_golden_cases()
    graph = build_graph(checkpointer=None)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Running {len(cases)} golden evaluation cases...\n")
    results: list[dict] = []
    for index, case in enumerate(cases):
        # Bounded retry with cool-down: Groq free-tier token budgets recover
        # over minutes, so a case that died to a gateway outage is re-run a
        # limited number of times instead of poisoning the whole aggregate.
        # The attempt count is recorded in the result for full honesty.
        max_attempts = 3
        cooldown_sec = 180
        inter_case_gap = 90
        result: dict | None = None
        attempts_used = 0
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            if attempt > 1:
                print(f"   (retry {attempt - 1}/{max_attempts - 1} after "
                      f"180s provider cool-down)")
                time.sleep(cooldown_sec)
            result = run_case(graph, case)
            result["attempts"] = attempts_used
            infra_failure = (
                result["error"]
                and "LLMGatewayError" in str(result["error"])
                and result["task_count"] == 0
            )
            if not infra_failure:
                break
        if index < len(cases) - 1:
            # Spacing between cases keeps the provider's TPM window healthy.
            time.sleep(inter_case_gap)
        print(f"-> {case['id']}: {case['query'][:70]}...")
        out_path = RESULTS_DIR / f"{result['case_id']}.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        status = "PASS" if result["passed_overall"] else "FAIL"
        print(
            f"   {status} | latency={result['latency_sec']}s | "
            f"tasks={result['task_count']} | words={result['word_count']} | "
            f"judges={list(result['semantic_judges'])}"
        )
        results.append(result)

    usage = get_recorded_token_usage()
    avg_cost = D.compute_cost(usage)

    aggregate = build_aggregate(results)
    aggregate["average_cost_usd"] = avg_cost

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps({"aggregate": aggregate, "per_case_files": sorted(p.name for p in RESULTS_DIR.glob("eval_*.json"))},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n================ AGGREGATE ================")
    print(json.dumps(aggregate, indent=2))
    print(f"\nPer-case results: {RESULTS_DIR}")
    print(f"Summary: {summary_path}")

    # Restore old-style console block used by README examples.
    print("\n--- headline numbers ---")
    sem = aggregate["semantic"]
    print(f"Router Correctness:       {sem['router_correctness']['passed']}/{sem['router_correctness']['runs']}")
    for key in ("evidence_groundedness", "plan_adherence", "factuality",
                "completeness", "citation_correctness"):
        if sem[key]["avg_score"] is not None:
            print(f"{key.replace('_', ' ').title()}: ".ljust(26) + str(sem[key]["avg_score"]))
    print(f"Final Task Success:       {sem['final_task_success']['passed']}/{sem['final_task_success']['runs']}")
    det = aggregate["deterministic"]
    print(f"Workflow Success:         {det['workflow_success']}/{len(results)}")
    print(f"Structural Integrity:     {det['structural_integrity']}/{len(results)}")
    print(f"Citation Allow-list:      {det['citation_allowlist']}/{len(results)}")
    print(f"Image Integrity:          {det['image_integrity']}/{len(results)}")
    print(f"P95 Latency:              {aggregate['p95_latency_sec']} sec")
    if avg_cost is not None:
        print(f"Average Cost:             ${avg_cost}")
    else:
        print("Average Cost:             n/a (no actual token usage data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
