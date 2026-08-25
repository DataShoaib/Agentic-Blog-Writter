from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

from app.config import APP_CONFIG, get_secrets
from app.observability.metrics import CACHE_HITS, CACHE_MISSES


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
            # Fixed-window counter: EXPIRE only on the first increment so the
            # window always expires 60s after it opened, instead of sliding
            # forward on every request and locking out low-frequency users.
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
                CACHE_MISSES.inc()
                return None
            CACHE_HITS.inc()
            return json.loads(value)
        except Exception:
            CACHE_MISSES.inc()
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
            # Fixed-window counter: the Lua script only sets EXPIRE on the first
            # increment, so the window always expires ttl seconds after it opened.
            return int(self._incr_script(keys=[key], args=[str(ttl)]))
        except Exception:
            return None


_STORE = RedisStore()


def get_redis_store() -> RedisStore:
    return _STORE


def cache_key(namespace: str, value: str) -> str:
    digest = hashlib.sha256(value.strip().encode("utf-8")).hexdigest()
    return f"aco:{namespace}:{digest}"


class LocalRateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[int, float]] = {}

    def allow(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.monotonic()
        with self._lock:
            count, started = self._windows.get(key, (0, now))
            if now - started >= window_seconds:
                count, started = 0, now
            count += 1
            self._windows[key] = (count, started)
            return count <= limit


_LOCAL_LIMITER = LocalRateLimiter()


def allow_request(identity: str) -> bool:
    key = f"aco:rate:{identity}"
    store = get_redis_store()
    count = store.incr_with_expiry(key, 60)
    if count is not None:
        return count <= APP_CONFIG.rate_limit_per_minute
    return _LOCAL_LIMITER.allow(identity, APP_CONFIG.rate_limit_per_minute)
