"""Tests for the whole-blog cache and its job-layer short-circuit."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.blog_cache import BlogCache
from app.services.jobs import JobManager, JobStore


TOPIC = "Explain self-attention"
AS_OF = "2026-08-25"
MODE = "auto"


def test_blog_cache_round_trip(tmp_path):
    cache = BlogCache(str(tmp_path / "cache.sqlite3"))
    assert cache.get("u", TOPIC, AS_OF, MODE) is None

    cache.set(
        "u", TOPIC, AS_OF, MODE,
        "# Title\n\nbody",
        {"blog_title": "T"},
        [{"url": "https://example.com"}],
    )

    got = cache.get("u", TOPIC, AS_OF, MODE)
    assert got is not None
    assert got.content == "# Title\n\nbody"
    assert got.plan == {"blog_title": "T"}
    assert got.evidence == [{"url": "https://example.com"}]


def test_blog_cache_respects_user_and_mode(tmp_path):
    cache = BlogCache(str(tmp_path / "cache.sqlite3"))
    cache.set("alice", TOPIC, AS_OF, MODE, "alice-content", {}, [])

    assert cache.get("bob", TOPIC, AS_OF, MODE) is None      # different user
    assert cache.get("alice", TOPIC, AS_OF, "force") is None  # different mode
    assert cache.get("alice", TOPIC, "2026-01-01", MODE) is None  # different date


def test_blog_cache_clear(tmp_path):
    cache = BlogCache(str(tmp_path / "cache.sqlite3"))
    cache.set("u", TOPIC, AS_OF, MODE, "c", {}, [])
    assert cache.clear() >= 1
    assert cache.get("u", TOPIC, AS_OF, MODE) is None


def test_blog_cache_handles_plan_objects(tmp_path):
    """run_job stores pydantic objects; json.dumps(default=str) must not choke."""
    cache = BlogCache(str(tmp_path / "cache.sqlite3"))
    plan = SimpleNamespace(blog_title="T")
    cache.set("u", TOPIC, AS_OF, MODE, "c", plan, [])
    got = cache.get("u", TOPIC, AS_OF, MODE)
    assert got is not None
    assert "T" in got.plan


def test_job_submit_short_circuits_on_cache_hit(tmp_path, monkeypatch):
    """A cache hit completes the job without enqueueing + running the graph."""
    manager = JobManager()
    manager.store = JobStore(str(tmp_path / "jobs.sqlite3"))

    cache = BlogCache(str(tmp_path / "cache.sqlite3"))
    cache.set("u", TOPIC, AS_OF, MODE, "cached-body", {"blog_title": "T"}, [])
    monkeypatch.setattr("app.services.jobs.get_blog_cache", lambda: cache)

    def _enqueue_should_not_run(*args, **kwargs):
        raise AssertionError("_enqueue_via_redis must not run on a cache hit")

    monkeypatch.setattr(manager, "_enqueue_via_redis", _enqueue_should_not_run)

    job_id = manager.submit("u", TOPIC, AS_OF, MODE)
    job = manager.store.get(job_id, "u")
    assert job["status"] == "completed"
    assert job["content"] == "cached-body"
    assert job["plan"] == {"blog_title": "T"}
    assert manager.execution_mode == "cached"


def test_job_submit_runs_when_cache_misses(tmp_path, monkeypatch):
    manager = JobManager()
    manager.store = JobStore(str(tmp_path / "jobs.sqlite3"))
    empty = BlogCache(str(tmp_path / "cache.sqlite3"))
    monkeypatch.setattr("app.services.jobs.get_blog_cache", lambda: empty)

    # Simulate Redis path being taken (returns True => job stays queued for worker).
    monkeypatch.setattr(manager, "_enqueue_via_redis", lambda *a, **k: True)

    job_id = manager.submit("u", TOPIC, AS_OF, MODE)
    job = manager.store.get(job_id, "u")
    assert job["status"] == "queued"
