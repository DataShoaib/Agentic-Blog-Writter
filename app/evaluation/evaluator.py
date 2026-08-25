from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date
from statistics import mean

from langchain_core.messages import HumanMessage, SystemMessage

from app.graph.graph import build_graph
from app.graph.schemas import EvaluationResult
from app.services.llm import invoke_structured


JUDGE_SYSTEM = """Evaluate a generated technical article against the requested topic and explicit criteria.
Score each criterion from 0 to 1 based on whether the content clearly covers it.
Also score factuality, completeness and citation quality from 0 to 1.
Do not reward buzzwords. Return EvaluationResult only."""


def evaluate_case(topic: str, content: str, criteria: dict) -> dict:
    result = invoke_structured(
        EvaluationResult,
        [
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n"
                    f"Required criteria: {criteria}\n\n"
                    f"Generated article:\n{content}"
                )
            ),
        ],
        operation="evaluation_judge",
    )
    return result.model_dump()


def run_dataset(path: str = "app/evaluation/dataset.json") -> dict:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = build_graph()
    results: list[dict] = []

    for index, case in enumerate(cases, start=1):
        topic = case["topic"]
        state = graph.invoke(
            {
                "topic": topic,
                "as_of": date.today().isoformat(),
                "sections": [],
                "evidence": [],
                "revision_count": 0,
                "enable_images": False,
            },
            config={"configurable": {"thread_id": f"eval-{index}"}},
        )
        content = state.get("final", "")
        scores = evaluate_case(topic, content, case.get("criteria", {}))
        results.append({
            "topic": topic,
            "criteria": case.get("criteria", {}),
            "scores": scores,
        })

    criterion_values = [
        score
        for item in results
        for score in item["scores"].get("criterion_scores", {}).values()
    ]
    summary = {
        "cases": len(results),
        "average_factuality": mean(
            item["scores"]["factuality_score"] for item in results
        ) if results else 0.0,
        "average_completeness": mean(
            item["scores"]["completeness_score"] for item in results
        ) if results else 0.0,
        "average_citation_quality": mean(
            item["scores"]["citation_score"] for item in results
        ) if results else 0.0,
        "average_criterion_score": mean(criterion_values) if criterion_values else 0.0,
    }
    return {"summary": summary, "results": results}


if __name__ == "__main__":
    output = run_dataset(sys.argv[1] if len(sys.argv) > 1 else "app/evaluation/dataset.json")
    print(json.dumps(output, indent=2))
