from __future__ import annotations

import os

from redis import Redis
from rq import SimpleWorker, Worker

from app.config import get_secrets
from app.observability.logging import setup_logging
from app.observability.tracing import configure_langsmith
from app.services.jobs import JOB_QUEUE_NAME, JobStore


def main() -> None:
    setup_logging()
    configure_langsmith()
    redis_url = get_secrets().redis_url
    if not redis_url:
        raise RuntimeError("REDIS_URL must be configured to start the worker.")

    try:
        # Reap worker-owned rows orphaned by a previous worker process
        # (legacy NULL executor rows count as worker-owned).
        JobStore().mark_interrupted_jobs(executor="worker")
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
