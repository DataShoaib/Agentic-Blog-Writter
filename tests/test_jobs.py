from app.services.jobs import JobStore


def test_job_store_round_trip(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    store.create("job-1", "user-1", "topic", "2026-08-20")
    store.update("job-1", status="completed", content="hello")
    job = store.get("job-1", "user-1")
    assert job is not None
    assert job["status"] == "completed"
    assert job["content"] == "hello"
    assert store.get("job-1", "user-2") is None


def test_mark_interrupted_jobs_fails_stale_running(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    store.create("job-1", "user-1", "topic", "2026-08-20")
    store.update("job-1", status="running", stage="router")
    store.mark_interrupted_jobs()
    job = store.get("job-1", "user-1")
    assert job["status"] == "failed"
    assert "interrupted" in job["error"].lower()
