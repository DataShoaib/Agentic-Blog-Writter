"""Whole-blog result cache (local SQLite).

A generation is an expensive pipeline: research + planning + parallel writing +
quality gate + optional revision + image generation. When the same user asks
for the same topic (same as_of + research_mode), we short-circuit and return
the already-built blog instead of re-running the whole graph.

This replaces per-query Tavily result caching: caching the entire finished
article is simpler and gives instant returns for repeated topics.

Stored in outputs/blog_cache.sqlite3 so it survives restarts.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

BLOG_CACHE_DB_PATH = "outputs/blog_cache.sqlite3"
BLOG_CACHE_TTL_SECONDS = 7 * 24 * 3600  # one week


@dataclass
class CachedBlog:
    content: str
    plan: dict
    evidence: list[dict]


class BlogCache:
    """File-backed, thread-safe cache of finished blog output."""

    def __init__(
        self,
        path: str = BLOG_CACHE_DB_PATH,
        ttl_seconds: int = BLOG_CACHE_TTL_SECONDS,
    ):
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blog_cache (
                    cache_key TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    research_mode TEXT NOT NULL,
                    content TEXT NOT NULL,
                    plan_json TEXT,
                    evidence_json TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_blog_cache_user "
                "ON blog_cache(user_id, created_at)"
            )
            conn.commit()

    @staticmethod
    def _cache_key(user_id: str, topic: str, as_of: str, research_mode: str) -> str:
        raw = f"{user_id}|{topic.strip()}|{as_of}|{research_mode}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        user_id: str,
        topic: str,
        as_of: str,
        research_mode: str,
    ) -> Optional[CachedBlog]:
        """Return a fresh cached blog, or None if missing/expired."""
        key = self._cache_key(user_id, topic, as_of, research_mode)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT content, plan_json, evidence_json FROM blog_cache "
                "WHERE cache_key=? AND expires_at > ?",
                (key, now),
            ).fetchone()
            if row is None:
                return None
            plan = json.loads(row["plan_json"]) if row["plan_json"] else {}
            evidence = json.loads(row["evidence_json"]) if row["evidence_json"] else []
            return CachedBlog(content=row["content"], plan=plan, evidence=evidence)

    def set(
        self,
        user_id: str,
        topic: str,
        as_of: str,
        research_mode: str,
        content: str,
        plan: Any,
        evidence: list[Any],
    ) -> None:
        """Store (or refresh) the full blog for this key."""
        key = self._cache_key(user_id, topic, as_of, research_mode)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=self.ttl_seconds)
        plan_json = json.dumps(plan, default=str)
        evidence_json = json.dumps(evidence, default=str)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO blog_cache (
                    cache_key, user_id, topic, as_of, research_mode,
                    content, plan_json, evidence_json, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    content=excluded.content,
                    plan_json=excluded.plan_json,
                    evidence_json=excluded.evidence_json,
                    created_at=excluded.created_at,
                    expires_at=excluded.expires_at
                """,
                (
                    key, user_id, topic, as_of, research_mode,
                    content, plan_json, evidence_json,
                    now.isoformat(), expires.isoformat(),
                ),
            )
            conn.commit()

    def clear(self) -> int:
        """Delete every entry. Mainly for tests and manual resets."""
        with self._lock, self._connect() as conn:
            deleted = conn.execute("DELETE FROM blog_cache").rowcount
            conn.commit()
        return deleted or 0


def get_blog_cache() -> BlogCache:
    """Default cache instance backed by outputs/blog_cache.sqlite3."""
    return BlogCache(BLOG_CACHE_DB_PATH)
