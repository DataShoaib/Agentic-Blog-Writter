from langgraph.checkpoint.memory import InMemorySaver

from app.graph.graph import build_graph


def test_graph_compiles_with_checkpointer():
    graph = build_graph(InMemorySaver())
    assert graph is not None


def test_quality_gate_builds_prompt_with_evidence(monkeypatch):
    """Regression test: quality_gate referenced an undefined 'all_evidence' variable."""
    from app.graph.nodes import quality_gate
    from app.graph.schemas import EvidenceItem, Plan, QualityResult, Task

    def fake_invoke(schema, messages, *, operation):
        return QualityResult(
            passed=True,
            factuality_score=0.9,
            completeness_score=0.9,
            citation_score=0.9,
            issues=[],
        )

    monkeypatch.setattr("app.graph.nodes.invoke_structured", fake_invoke)

    tasks = [
        Task(
            id=i, title=f"Section {i}",
            goal="Explain the basic concept clearly.",
            bullets=["a", "b", "c"], target_words=200,
        )
        for i in range(1, 6)
    ]
    plan = Plan(
        blog_title="How RAG works", audience="Engineers", tone="Clear", tasks=tasks,
    )
    # Enough words to satisfy the >=90% length floor (5 x 200 = 1000 planned).
    body = " ".join(["retrieval", "augmented", "generation", "pipeline"] * 280)
    out = quality_gate(
        {
            "plan": plan,
            "evidence": [EvidenceItem(url="https://example.com/docs")],
            "merged_md": f"# How RAG works\n\n{body}\n\n[Source](https://example.com/docs)",
        }
    )
    assert "quality" in out
    assert out["quality"]["passed"] is True


def test_revision_route_is_bounded():
    from app.graph.nodes import route_quality

    assert route_quality({"quality": {"passed": False}, "revision_count": 0, "max_revision_attempts": 1}) == "revise"
    assert route_quality({"quality": {"passed": False}, "revision_count": 1, "max_revision_attempts": 1}) == "images"
    assert route_quality({"quality": {"passed": True}, "revision_count": 0, "max_revision_attempts": 1}) == "images"
