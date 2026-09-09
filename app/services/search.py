from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from langchain_tavily import TavilySearch

from app.config import APP_CONFIG, get_secrets
from app.graph.schemas import EvidenceItem


class TransientSearchError(Exception):
    pass


def _is_transient_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return status in {408, 429, 500, 502, 503, 504} or isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _parse_date(value: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except (TypeError, ValueError):
        return None


def _normalize(results: object) -> list[EvidenceItem]:
    rows = results.get("results", []) if isinstance(results, dict) else results
    return [EvidenceItem(title=str(r.get("title") or "").strip(), url=url, snippet=str(r.get("content") or "").strip()[:1200] or None, published_at=r.get("published_date") or r.get("published_at"), source=r.get("source") or None) for r in rows if isinstance(r, dict) and (url := str(r.get("url") or "").strip())] if isinstance(rows, list) else []


def search_web(query: str, max_results: int = 6) -> list[EvidenceItem]:
    if not get_secrets().tavily_api_key:
        return []
    try:
        return _normalize(TavilySearch(max_results=max(1, min(max_results, APP_CONFIG.max_research_results)), topic="general").invoke({"query": query}))
    except Exception as exc:
        if _is_transient_error(exc):
            raise TransientSearchError("Tavily search temporarily failed.") from exc
        return []


def dedupe_and_filter(items: list[EvidenceItem], as_of: str, recency_days: int, *, strict_recency: bool = False) -> list[EvidenceItem]:
    unique = {item.url.strip(): item for item in items if item.url.strip()}
    if recency_days >= 3650:
        return list(unique.values())
    try:
        end = date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError(f"Invalid as_of date: {as_of}") from exc
    start = end - timedelta(days=recency_days)
    return [item for item in unique.values() if (published := _parse_date(item.published_at)) and start <= published <= end or (not strict_recency and published is None)]

