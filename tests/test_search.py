from app.graph.schemas import EvidenceItem
from app.services.search import dedupe_and_filter


def test_hybrid_keeps_undated_evidence():
    items = [EvidenceItem(title="Docs", url="https://example.com/docs")]
    out = dedupe_and_filter(items, "2026-08-20", 45, strict_recency=False)
    assert len(out) == 1


def test_open_book_rejects_undated_evidence():
    items = [EvidenceItem(title="Docs", url="https://example.com/docs")]
    out = dedupe_and_filter(items, "2026-08-20", 7, strict_recency=True)
    assert out == []
