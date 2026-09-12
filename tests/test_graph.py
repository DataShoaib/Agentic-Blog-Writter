from langgraph.checkpoint.memory import InMemorySaver

from app.graph.graph import build_graph


def test_graph_compiles_with_checkpointer():
    graph = build_graph(InMemorySaver())
    assert graph is not None


def test_quality_gate_builds_prompt_with_evidence(monkeypatch):
    """Regression test: quality_gate referenced an undefined 'all_evidence' variable."""
    from app.graph.nodes import quality_gate
    from app.graph.schemas import EvidenceItem, Plan, QualityResult, Task

    def fake_invoke(schema, messages, *, operation, preferred_model=None):
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
    # Enough words to satisfy the >=90% length floor (5 x 200 = 1000 planned),
    # with one proper "## Section i" block each so the deterministic
    # per-section depth check (>=200 words, >=2 substantial paragraphs) passes.
    body = "\n\n".join(
        f"## Section {i}\n\n" + (" ".join(["retrieval"] * 70) + "\n\n")
        + (" ".join(["augmented"] * 70) + "\n\n")
        + (" ".join(["generation"] * 70))
        for i in range(1, 6)
    )
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


def test_merge_keeps_every_planned_section():
    """All worker outputs must reach the merged article, in task order."""
    from app.graph.nodes import merge_content
    from app.graph.schemas import Plan, Task

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
    # Workers finish out of order — merge must still emit 1..5 in order.
    sections = [(3, "## Section 3\n\nthree"), (1, "## Section 1\n\none"),
                (5, "## Section 5\n\nfive"), (2, "## Section 2\n\ntwo"),
                (4, "## Section 4\n\nfour")]
    out = merge_content({"plan": plan, "sections": sections})
    merged = out["merged_md"]
    positions = [merged.index(f"## Section {i}") for i in range(1, 6)]
    assert positions == sorted(positions)


def test_merge_fails_loudly_on_missing_section():
    """A dropped worker write must fail, never silently ship a short blog."""
    import pytest

    from app.graph.nodes import merge_content
    from app.graph.schemas import Plan, Task

    tasks = [
        Task(
            id=i, title=f"Section {i}",
            goal="Explain the basic concept clearly.",
            bullets=["a", "b", "c"], target_words=200,
        )
        for i in range(1, 4)
    ]
    plan = Plan(
        blog_title="How RAG works", audience="Engineers", tone="Clear", tasks=tasks,
    )
    with pytest.raises(ValueError, match="missing sections"):
        merge_content({"plan": plan, "sections": [(1, "## Section 1\n\none")]})


def test_revision_never_shrinks_article(monkeypatch):
    """A truncated revision response must not drop planned sections."""
    from app.graph.nodes import revise_content
    from app.graph.schemas import Plan, Task

    tasks = [
        Task(
            id=i, title=f"Section {i}",
            goal="Explain the basic concept clearly.",
            bullets=["a", "b", "c"], target_words=200,
        )
        for i in range(1, 4)
    ]
    plan = Plan(
        blog_title="How RAG works", audience="Engineers", tone="Clear", tasks=tasks,
    )
    original = (
        "# How RAG works\n\n"
        + "\n\n".join(f"## Section {i}\n\n" + ("word " * 200) for i in range(1, 4))
    )
    # Simulate an output-truncated LLM reply (only the first section).
    monkeypatch.setattr(
        "app.graph.nodes.invoke_text", lambda *a, **k: "## Section 1\n\nshort"
    )
    out = revise_content(
        {"plan": plan, "merged_md": original, "quality": {"issues": ["thin"]}}
    )
    assert out["merged_md"] == original




