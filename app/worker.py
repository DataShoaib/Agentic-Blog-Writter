from __future__ import annotations

from redis import Redis
from rq import Worker

from app.config import get_secrets
from app.observability.tracing import configure_langsmith
from app.services.jobs import JobStore, JOBS_DB_PATH


def main() -> None:
    configure_langsmith()
    redis_url = get_secrets().redis_url
    if not redis_url:
        raise RuntimeError("REDIS_URL must be configured to start the worker.")
    # Any job left in the 'running' state by a previously crashed worker process is
    # never going to complete again. Mark those as failed before consuming new work.
    try:
        JobStore(JOBS_DB_PATH).mark_interrupted_jobs()
    except Exception:
        pass
    connection = Redis.from_url(redis_url)
    Worker(
        ["blog-generation"],
        connection=connection,
        name="blog-generation-worker",
    ).work()


if __name__ == "__main__":
    main()