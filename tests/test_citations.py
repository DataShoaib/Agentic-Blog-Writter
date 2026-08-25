from app.services.citations import has_valid_http_urls, validate_citations


def test_approved_citation_passes():
    score, issues = validate_citations(
        "[Source](https://example.com)",
        [{"url": "https://example.com"}],
        citations_required=True,
    )
    assert score == 1.0
    assert issues == []
    assert has_valid_http_urls("[Source](https://example.com)")


def test_unapproved_citation_fails():
    score, issues = validate_citations(
        "[Source](https://evil.example)",
        [{"url": "https://example.com"}],
        citations_required=True,
    )
    assert score == 0.0
    assert issues


def test_missing_required_citation_fails():
    score, issues = validate_citations(
        "No links here.",
        [{"url": "https://example.com"}],
        citations_required=True,
    )
    assert score == 0.0
    assert issues
