from __future__ import annotations

import re
from urllib.parse import urlparse


def approved_urls(evidence: list[dict]) -> set[str]:
    return {str(item.get("url", "")).strip() for item in evidence if item.get("url")}


def citation_urls(markdown: str) -> set[str]:
    return set(re.findall(r"\]\((https?://[^)\s]+)\)", markdown))


def validate_citations(
    markdown: str,
    evidence: list[dict],
    *,
    citations_required: bool = False,
) -> tuple[float, list[str]]:
    """Ensure every explicit URL citation belongs to the approved evidence set."""
    urls = approved_urls(evidence)
    cited = citation_urls(markdown)

    issues: list[str] = []
    if citations_required and not cited:
        issues.append("Citations are required for this content, but none were found.")

    bad = sorted(cited - urls)
    issues.extend(f"Unapproved citation URL: {url}" for url in bad)

    if not urls:
        return (0.0 if citations_required else 1.0), issues

    if not cited:
        return (0.0 if citations_required else 1.0), issues

    score = len(cited & urls) / len(cited)
    return score, issues


def has_valid_http_urls(markdown: str) -> bool:
    """Small helper used by tests to verify URL shape without doing network calls."""
    for url in citation_urls(markdown):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
    return True
