import pytest

from app.services import db
from app.services.jobs import JobStore

pytestmark = pytest.mark.usefixtures("requires_db")


@pytest.fixture(autouse=True)
def _clean_test_jobs(requires_db):
    """The Postgres jobs table is shared; remove the fixed-id test rows first.

    Depends on ``requires_db`` so the suite skips cleanly when Postgres is not
    reachable (no DB connect is even attempted in that case).
    """
    store = JobStore()
    with store._lock, store._connect() as conn:
        conn.execute(db.q("DELETE FROM jobs WHERE job_id IN ('job-1', 'job-2', 'job-w', 'job-a')"))
        conn.commit()
    yield


def test_job_store_round_trip():
    store = JobStore()
    store.create("job-1", "user-1", "topic", "2026-08-20")
    store.update("job-1", status="completed", content="hello")
    job = store.get("job-1", "user-1")
    assert job is not None
    assert job["status"] == "completed"
    assert job["content"] == "hello"
    assert store.get("job-1", "user-2") is None


def test_mark_interrupted_jobs_fails_stale_running():
    store = JobStore()
    store.create("job-1", "user-1", "topic", "2026-08-20")
    store.update("job-1", status="running", stage="router")
    store.mark_interrupted_jobs()
    job = store.get("job-1", "user-1")
    assert job["status"] == "failed"
    assert "interrupted" in job["error"].lower()


def test_list_blogs_returns_only_own_blogs():
    """Per-user history: /blogs must never leak another user's blogs."""
    store = JobStore()
    store.create("job-1", "user-1", "Topic A", "2026-08-20")
    store.update("job-1", status="completed", content="A body")
    store.create("job-2", "user-2", "Topic B", "2026-08-21")
    store.update("job-2", status="completed", content="B body")

    mine = store.list_by_user("user-1")
    assert [blog["job_id"] for blog in mine] == ["job-1"]
    assert mine[0]["topic"] == "Topic A"

    assert store.list_by_user("user-2")[0]["job_id"] == "job-2"
    assert store.list_by_user("user-with-no-blogs") == []


def test_mark_interrupted_jobs_scopes_to_executor():
    """The worker must not reap API-owned thread jobs (and vice versa) —
    cross-kill used to fail live jobs on every restart of the other process."""
    store = JobStore()
    store.create("job-w", "user-1", "t", "2026-09-28")
    store.update("job-w", status="running", executor="worker")
    store.create("job-a", "user-1", "t", "2026-09-28")
    store.update("job-a", status="running", executor="api")

    store.mark_interrupted_jobs(executor="worker")
    assert store.get("job-w", "user-1")["status"] == "failed"
    assert store.get("job-a", "user-1")["status"] == "running"

    store.mark_interrupted_jobs(executor="api")
    assert store.get("job-a", "user-1")["status"] == "failed"
