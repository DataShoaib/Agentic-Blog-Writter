from __future__ import annotations

import os

from redis import Redis
from rq import SimpleWorker, Worker

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
    # RQ's default worker forks a work horse, which is not available on
    # Windows. SimpleWorker runs jobs in-process there instead. A pid-unique
    # name avoids "active worker already" clashes after unclean restarts.
    worker_cls = SimpleWorker if os.name == "nt" else Worker
    worker_cls(
        [JOB_QUEUE_NAME],
        connection=connection,
        name=f"blog-generation-worker-{os.getpid()}",
    ).work()


if __name__ == "__main__":
    main()
