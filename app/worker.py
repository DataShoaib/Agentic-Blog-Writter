from __future__ import annotations

from redis import Redis
from rq import Worker

from app.config import get_secrets
from app.observability.tracing import configure_langsmith
from app.services.jobs import JOB_QUEUE_NAME, JOBS_DB_PATH, JobStore


def main() -> None:
    configure_langsmith()
    redis_url = get_secrets().redis_url
    if not redis_url:
        raise RuntimeError("REDIS_URL must be configured to start the worker.")

    try:
        JobStore(JOBS_DB_PATH).mark_interrupted_jobs()
    except Exception:
        pass

    connection = Redis.from_url(redis_url)
    Worker(
        [JOB_QUEUE_NAME],
        connection=connection,
        name="blog-generation-worker",
    ).work()


if __name__ == "__main__":
    main()