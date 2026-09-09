"""Tests for the Redis-backed whole-blog cache and its job-layer short-circuit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.cache import BlogCache
from app.services.jobs import JobManager, JobStore

TOPIC = "Explain self-attention"
AS_OF = "2026-08-25"


class FakeRedisStore:

    def __init__(self):
        self._data = {}
        self.available = True

    def get_json(self, key):
        return self._data.get(key)

    def set_json(self, key, value, ttl):
        self._data[key] = value


def _cache() -> BlogCache:
    cache = BlogCache()
    cache._store = FakeRedisStore()
    return cache


def test_blog_cache_round_trip():
    cache = _cache()
    assert cache.get(TOPIC, AS_OF) is None

    cache.set(
        TOPIC, AS_OF,
        "# Title\n\nbody",
        {"blog_title": "T"},
        [{"url": "https://example.com"}],
    )

    got = cache.get(TOPIC, AS_OF)
    assert got is not None
    assert got.content == "# Title\n\nbody"
    assert got.plan == {"blog_title": "T"}
    assert got.evidence == [{"url": "https://example.com"}]


def test_blog_cache_is_universal_across_users():
    """Cache is shared, not user-scoped: alice's result serves bob too."""
    cache = _cache()
    cache.set(TOPIC, AS_OF, "shared-content", {}, [])

    assert cache.get(TOPIC, AS_OF).content == "shared-content"  # any user hits
    assert cache.get(TOPIC, "2026-01-01") is None  # different date misses


def test_blog_cache_unavailable_returns_none():
    cache = _cache()
    cache._store.available = False
    cache.set(TOPIC, AS_OF, "c", {}, [])

    assert cache.get(TOPIC, AS_OF) is None


def test_blog_cache_handles_plan_objects():
    """run_job stores pydantic objects; set() must coerce them to a dict."""
    cache = _cache()
    plan = SimpleNamespace(blog_title="T")
    cache.set(TOPIC, AS_OF, "c", plan, [])

    got = cache.get(TOPIC, AS_OF)
    assert got is not None
    assert got.plan == {"blog_title": "T"}


@pytest.mark.usefixtures("requires_db")
def test_job_submit_short_circuits_on_cache_hit(monkeypatch):
    """A cache hit completes the job without enqueueing + running the graph."""
    manager = JobManager()
    manager.store = JobStore()

    cache = _cache()
    cache.set(TOPIC, AS_OF, "cached-body", {"blog_title": "T"}, [])
    monkeypatch.setattr("app.services.jobs.get_blog_cache", lambda: cache)

    def _enqueue_should_not_run(*args, **kwargs):
        raise AssertionError("_enqueue_via_redis must not run on a cache hit")

    monkeypatch.setattr(manager, "_enqueue_via_redis", _enqueue_should_not_run)

    job_id = manager.submit("u", TOPIC, AS_OF)
    job = manager.store.get(job_id, "u")
    assert job["status"] == "completed"
    assert job["content"] == "cached-body"
    assert job["plan"] == {"blog_title": "T"}
    assert manager.execution_mode == "cached"


@pytest.mark.usefixtures("requires_db")
def test_job_submit_runs_when_cache_misses(monkeypatch):
    manager = JobManager()
    manager.store = JobStore()

    empty = _cache()
    monkeypatch.setattr("app.services.jobs.get_blog_cache", lambda: empty)

    # Simulate Redis path being taken (returns True => job stays queued for worker).
    monkeypatch.setattr(manager, "_enqueue_via_redis", lambda *a, **k: True)

    job_id = manager.submit("u", TOPIC, AS_OF)
    job = manager.store.get(job_id, "u")
    assert job["status"] == "queued"

