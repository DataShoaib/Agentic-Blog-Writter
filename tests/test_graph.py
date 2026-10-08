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


def test_merge_drops_duplicate_worker_outputs():
    """Quota-storm duplicate fan-out writes collapse: first write wins."""
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
    sections = [
        (1, "## Section 1\n\none"), (1, "## Section 1\n\none-retry"),
        (2, "## Section 2\n\ntwo"), (3, "## Section 3\n\nthree"),
    ]
    out = merge_content({"plan": plan, "sections": sections})
    assert "## Section 1\n\none\n" in out["merged_md"]
    assert "one-retry" not in out["merged_md"]
    for i in range(1, 4):
        assert f"## Section {i}" in out["merged_md"]


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


def _plan_one_task(target_words: int = 200):
    from app.graph.schemas import Plan, Task

    task = Task(
        id=1,
        title="Section 1",
        goal="Explain it.",
        bullets=["a", "b", "c"],
        target_words=target_words,
    )
    return Plan(blog_title="T", audience="Engineers", tone="Clear", tasks=[task])


def _fat_body(paragraphs: int = 3, words: int = 70) -> str:
    return "\n\n".join(" ".join(["retrieval"] * words) for _ in range(paragraphs))


def test_quality_gate_fails_thin_closed_book_without_llm_call(monkeypatch):
    """closed_book skips only the LLM review — the LLM-free deterministic
    checks (thin sections, total length) must still fail a stub article."""
    from app.graph.nodes import quality_gate

    def no_llm(*args, **kwargs):
        raise AssertionError("closed_book must not call the LLM judge")

    monkeypatch.setattr("app.graph.nodes.invoke_structured", no_llm)

    out = quality_gate(
        {
            "mode": "closed_book",
            "plan": _plan_one_task(400),
            "evidence": [],
            "merged_md": "# T\n\n## Section 1\n\ntiny stub\n",
        }
    )
    assert out["quality"]["passed"] is False
    assert any("Article too short" in issue for issue in out["quality"]["issues"])
    assert any("too thin" in issue for issue in out["quality"]["issues"])


def test_quality_gate_passes_complete_closed_book_without_llm_call(monkeypatch):
    from app.graph.nodes import quality_gate

    def no_llm(*args, **kwargs):
        raise AssertionError("closed_book must not call the LLM judge")

    monkeypatch.setattr("app.graph.nodes.invoke_structured", no_llm)

    out = quality_gate(
        {
            "mode": "closed_book",
            "plan": _plan_one_task(200),
            "evidence": [],
            "merged_md": f"# T\n\n## Section 1\n\n{_fat_body()}",
        }
    )
    assert out["quality"]["passed"] is True


def test_quality_gate_stray_citation_does_not_sink_the_gate(monkeypatch):
    """Deterministic citation score for 1 approved + 1 stray URL is 0.5 —
    the floor at the 0.75 threshold must keep an otherwise-cited article
    passing (regression: the old 0.6 floor sat BELOW the 0.75 gate, so the
    floor could never actually save the article)."""
    from app.graph.nodes import quality_gate
    from app.graph.schemas import EvidenceItem, QualityResult

    def fake_invoke(schema, messages, *, operation, preferred_model=None):
        return QualityResult(
            passed=True,
            factuality_score=0.9,
            completeness_score=0.9,
            citation_score=0.9,
            issues=[],
        )

    monkeypatch.setattr("app.graph.nodes.invoke_structured", fake_invoke)

    merged = (
        f"# T\n\n## Section 1\n\n{_fat_body()}\n\n"
        "[Good](https://example.com/docs) and [Bad](https://evil.example)."
    )
    out = quality_gate(
        {
            "mode": "hybrid",
            "plan": _plan_one_task(200),
            "evidence": [EvidenceItem(url="https://example.com/docs")],
            "merged_md": merged,
        }
    )
    assert out["quality"]["citation_score"] >= 0.75
    assert out["quality"]["passed"] is True
    assert any("Unapproved citation URL" in issue for issue in out["quality"]["issues"])


def test_decide_images_degrades_on_gateway_error(monkeypatch):
    """Image planning runs at 92% — a quota error there must not lose the
    finished article (regression: unhandled LLMGatewayError killed the job)."""
    from app.graph.nodes import decide_images
    from app.services.llm import LLMGatewayError

    def boom(*args, **kwargs):
        raise LLMGatewayError("quota exhausted")

    monkeypatch.setattr("app.graph.nodes.invoke_structured", boom)
    monkeypatch.setattr("app.graph.nodes.has_image_api_key", lambda api_key=None: True)

    out = decide_images({"topic": "T", "merged_md": "# T\n", "enable_images": True})
    assert out == {"md_with_placeholders": "# T\n", "image_specs": []}


def test_research_keeps_raw_evidence_when_synthesis_fails(monkeypatch):
    """If the synthesis LLM is dead, the raw (already normalized) search
    hits survive instead of failing the whole research node."""
    from app.graph.nodes import research_node
    from app.graph.schemas import EvidenceItem
    from app.services.llm import LLMGatewayError

    monkeypatch.setattr(
        "app.graph.nodes.search_web",
        lambda q, n: [EvidenceItem(title="Doc", url="https://example.com/a", snippet="s")],
    )

    def boom(*args, **kwargs):
        raise LLMGatewayError("quota exhausted")

    monkeypatch.setattr("app.graph.nodes.invoke_structured", boom)

    out = research_node(
        {
            "queries": ["q1"],
            "max_results_per_query": 3,
            "as_of": "2026-09-28",
            "recency_days": 3650,
            "mode": "hybrid",
        }
    )
    assert [item.url for item in out["evidence"]] == ["https://example.com/a"]




