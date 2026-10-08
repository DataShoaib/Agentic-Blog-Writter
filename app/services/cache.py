from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from app.config import APP_CONFIG, get_secrets
# Metrics removed - using LangSmith for observability instead


class RedisStore:
    def __init__(self) -> None:
        self._client = None
        self._incr_script = None
        self._lock = threading.Lock()

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        redis_url = get_secrets().redis_url
        if not redis_url:
            return
        try:
            import redis

            client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
            incr_script = client.register_script(
                "local count = redis.call('INCR', KEYS[1]) "
                "if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end "
                "return count"
            )
            with self._lock:
                self._client = client
                self._incr_script = incr_script
        except Exception:
            return

    @property
    def available(self) -> bool:
        self._ensure_client()
        return self._client is not None

    def get_json(self, key: str) -> Any | None:
        self._ensure_client()
        if not self._client:
            return None
        try:
            value = self._client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl: int) -> None:
        self._ensure_client()
        if not self._client:
            return
        try:
            self._client.setex(key, ttl, json.dumps(value, separators=(",", ":")))
        except Exception:
            return

    def incr_with_expiry(self, key: str, ttl: int) -> int | None:
        self._ensure_client()
        if self._client is None or self._incr_script is None:
            return None
        try:
            return int(self._incr_script(keys=[key], args=[str(ttl)]))
        except Exception:
            return None


_STORE = RedisStore()


def get_redis_store() -> RedisStore:
    return _STORE


def cache_key(namespace: str, value: str) -> str:
    digest = hashlib.sha256(value.strip().encode("utf-8")).hexdigest()
    return f"aco:{namespace}:{digest}"


def allow_request(identity: str) -> bool:
    """Redis-only rate limiting. Fails open when Redis is unavailable."""
    key = f"aco:rate:{identity}"
    store = get_redis_store()
    count = store.incr_with_expiry(key, 60)
    if count is not None:
        return count <= APP_CONFIG.rate_limit_per_minute
    return True


# Whole-blog result cache (Redis-backed, universal).

from dataclasses import dataclass  # noqa: E402
from typing import Optional  # noqa: E402 — Any already imported at top of module


BLOG_CACHE_TTL_SECONDS = 7 * 24 * 3600  # one week


@dataclass
class CachedBlog:
    """Snapshot of a finished blog stored in / loaded from Redis."""

    content: str
    plan: dict
    evidence: list[dict]


class BlogCache:
    # v3: the images on/off flag participates in the key — an images-ON
    # request must never be served a no-images article from cache (and the
    # reverse). v2|topic|date entries can never collide with the new format.
    CACHE_VERSION = 3

    def __init__(self, ttl_seconds: int = BLOG_CACHE_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._store = get_redis_store()

    @staticmethod
    def _cache_key(topic: str, as_of: str, enable_images: bool = False) -> str:
        raw = (
            f"v{BlogCache.CACHE_VERSION}|{topic.strip()}|{as_of}|"
            f"{1 if enable_images else 0}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _redis_key(self, topic: str, as_of: str, enable_images: bool = False) -> str:
        return cache_key("blog", self._cache_key(topic, as_of, enable_images))

    def available(self) -> bool:
        return self._store.available

    def get(
        self,
        topic: str,
        as_of: str,
        *,
        enable_images: bool = False,
    ) -> Optional[CachedBlog]:
        if not self._store.available:
            return None

        data = self._store.get_json(
            self._redis_key(topic, as_of, enable_images)
        )

        if not data:
            return None

        return CachedBlog(
            content=data.get("content", ""),
            plan=data.get("plan") or {},
            evidence=data.get("evidence") or [],
        )

    @staticmethod
    def _to_dict(obj: Any) -> Any:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return obj

    def set(
        self,
        topic: str,
        as_of: str,
        content: str,
        plan: Any,
        evidence: list[Any],
        *,
        enable_images: bool = False,
    ) -> None:
        if not self._store.available:
            return

        try:
            payload = {
                "content": content,
                "plan": self._to_dict(plan),
                "evidence": [self._to_dict(item) for item in evidence],
            }

            self._store.set_json(
                self._redis_key(topic, as_of, enable_images),
                payload,
                self.ttl_seconds,
            )
        except Exception:
            pass


def get_blog_cache() -> BlogCache:
    return BlogCache()