import pytest

from app.services.jobs import JobStore

pytestmark = pytest.mark.usefixtures("requires_db")


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
