from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from app.config import APP_CONFIG, get_secrets
from app.graph.schemas import EvidenceItem


class TransientSearchError(Exception):
    """A search failure that may succeed when the research node is retried."""


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def search_web(query: str, max_results: int = 6) -> list[EvidenceItem]:
    """Search Tavily and return normalized EvidenceItem objects.

    No per-query result caching is done here: whole-blog caching lives at the
    job layer instead (see app.services.blog_cache).
    """
    secrets = get_secrets()
    if not secrets.tavily_api_key:
        return []

    max_results = max(1, min(max_results, APP_CONFIG.max_research_results))

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=secrets.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            timeout=APP_CONFIG.search_timeout_seconds,
        )
        rows = response.get("results", []) if isinstance(response, dict) else []
    except Exception as exc:
        if _is_transient_error(exc):
            raise TransientSearchError("Tavily search temporarily failed.") from exc
        return []

    evidence: list[EvidenceItem] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("url"):
            continue
        evidence.append(
            EvidenceItem(
                title=str(row.get("title") or "").strip(),
                url=str(row.get("url") or "").strip(),
                snippet=str(row.get("content") or "").strip()[:1200] or None,
                published_at=row.get("published_date") or row.get("published_at"),
                source=row.get("source") or None,
            )
        )
    return evidence


def dedupe_and_filter(
    items: list[EvidenceItem],
    as_of: str,
    recency_days: int,
    *,
    strict_recency: bool,
) -> list[EvidenceItem]:
    """Dedupe by URL and apply mode-specific recency policy."""
    unique: dict[str, EvidenceItem] = {}
    for item in items:
        url = item.url.strip()
        if url:
            unique[url] = item

    if recency_days >= 3650:
        return list(unique.values())

    try:
        as_of_date = date.fromisoformat(as_of)
        cutoff = as_of_date - timedelta(days=recency_days)
    except ValueError as exc:
        raise ValueError(f"Invalid as_of date: {as_of}") from exc

    filtered: list[EvidenceItem] = []
    for item in unique.values():
        published = _parse_date(item.published_at)
        if published is not None and cutoff <= published <= as_of_date:
            filtered.append(item)
        elif not strict_recency and published is None:
            filtered.append(item)
    return filtered
