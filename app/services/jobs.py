from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langsmith import tracing_context

from app.config import APP_CONFIG
from app.observability.tracing import configure_langsmith
from app.graph.graph import build_graph
from app.observability.metrics import (
    GRAPH_COMPLETED,
    GRAPH_FAILURES,
    GRAPH_LATENCY,
    GRAPH_RUNS,
    JOB_FAILURES,
    JOB_QUEUE_DEPTH,
    JOB_SUBMISSIONS,
)


JOBS_DB_PATH = "outputs/jobs.sqlite3"

logger = logging.getLogger(__name__)

# One compiled graph per process: the InMemorySaver checkpointer then persists
# across jobs instead of being thrown away after every single generation.
_COMPILED_GRAPH = None


def get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Small SQLite registry for API-level job state and per-user ownership."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    content TEXT,
                    error TEXT,
                    stage TEXT,
                    plan_json TEXT,
                    evidence_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "user_id" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN user_id TEXT NOT NULL DEFAULT 'legacy'")
            for column, definition in (("stage", "TEXT"), ("plan_json", "TEXT"), ("evidence_json", "TEXT")):
                if column not in columns:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at)")
            conn.commit()

    def mark_interrupted_jobs(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status='failed', error=?, updated_at=?
                WHERE status='running'
                """,
                ("Job interrupted by application restart.", _utc_now()),
            )
            conn.commit()

    def create(self, job_id: str, user_id: str, topic: str, as_of: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id,user_id,status,topic,as_of,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (job_id, user_id, "queued", topic, as_of, now, now),
            )
            conn.commit()

    def update(
        self,
        job_id: str,
        *,
        status: str,
        content: str | None = None,
        error: str | None = None,
        stage: str | None = None,
        plan: dict | None = None,
        evidence: list[dict] | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status=?, content=COALESCE(?, content), error=?, stage=COALESCE(?, stage),
                    plan_json=COALESCE(?, plan_json), evidence_json=COALESCE(?, evidence_json), updated_at=?
                WHERE job_id=?
                """,
                (status, content, error, stage, json.dumps(plan) if plan is not None else None,
                 json.dumps(evidence) if evidence is not None else None, _utc_now(), job_id),
            )
            conn.commit()

    def get(self, job_id: str, user_id: str) -> dict | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id,user_id,status,topic,as_of,content,error,stage,plan_json,evidence_json,created_at,updated_at
                FROM jobs
                WHERE job_id=? AND user_id=?
                """,
                (job_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json") or "null")
        result["evidence"] = json.loads(result.pop("evidence_json") or "[]")
        return result


class JobManager:
    """Enqueue generation jobs on Redis/RQ with an in-process fallback.

    Redis/RQ stays the preferred path: a separate worker process keeps long
    graph runs isolated from the API. When Redis is not configured, is not
    reachable, or no worker is consuming the queue (the common single-machine
    setup), jobs run on a local daemon thread instead, so generation keeps
    working without extra infrastructure. This mirrors the local fallbacks the
    cache and rate limiter already use.
    """

    def __init__(self):
        self.store = JobStore("outputs/jobs.sqlite3")
        self._queue = None
        self._retry = None
        self.execution_mode: str | None = None

    def _get_queue(self):
        if self._queue is not None:
            return self._queue
        from redis import Redis
        from rq import Queue, Retry

        from app.config import get_secrets

        redis_url = get_secrets().redis_url
        if not redis_url:
            raise RuntimeError("REDIS_URL must be configured to submit jobs.")
        self._queue = Queue(
            "blog-generation",
            connection=Redis.from_url(redis_url),
            default_timeout=APP_CONFIG.job_timeout_seconds,
        )
        self._retry = Retry(max=2, interval=[10, 30])
        return self._queue

    @property
    def default_execution_mode(self) -> str:
        """Executor the next submission will use; useful for health checks."""
        if self.execution_mode:
            return self.execution_mode
        from app.config import get_secrets

        return "redis+rq" if get_secrets().redis_url else "in-process"

    def _enqueue_via_redis(
        self,
        job_id: str,
        user_id: str,
        topic: str,
        as_of: str,
        research_mode: str = "auto",
    ) -> bool:
        """Try RQ first. Returns False when the local executor should take over."""
        try:
            queue = self._get_queue()
            from rq import Worker

            # An enqueue without a live worker would strand the job as queued
            # forever; prefer the local thread when nobody is consuming.
            if Worker.count(queue=queue) == 0:
                logger.warning(
                    "No RQ worker consumes queue '%s'; using the in-process executor.",
                    queue.name,
                )
                return False
            queue.enqueue(
                run_job,
                job_id,
                user_id,
                topic,
                as_of,
                research_mode,
                job_id=job_id,
                retry=self._retry,
            )
            try:
                JOB_QUEUE_DEPTH.set(int(queue.count))
            except Exception:
                # Depth is best-effort observability; never fail a submit over it.
                pass
            self.execution_mode = "redis+rq"
            return True
        except Exception as exc:
            logger.warning("Redis queue unavailable (%s); using the in-process executor.", exc)
            return False

    def submit(
        self,
        user_id: str,
        topic: str,
        as_of: str,
        research_mode: str = "auto",
    ) -> str:
        job_id = str(uuid.uuid4())
        self.store.create(job_id, user_id, topic, as_of)
        if not self._enqueue_via_redis(job_id, user_id, topic, as_of, research_mode):
            threading.Thread(
                target=run_job,
                args=(job_id, user_id, topic, as_of, research_mode),
                name=f"blog-job-{job_id[:8]}",
                daemon=True,
            ).start()
            self.execution_mode = "in-process"
        JOB_SUBMISSIONS.inc()
        return job_id

    def get(self, job_id: str, user_id: str) -> dict | None:
        return self.store.get(job_id, user_id)


def run_job(
    job_id: str,
    user_id: str,
    topic: str,
    as_of: str,
    research_mode: str = "auto",
) -> None:
    """Graph entry point executed by the RQ worker or the in-process fallback."""
    configure_langsmith()
    from app.services.memory import get_blog_memory

    memory = get_blog_memory()
    store = JobStore(JOBS_DB_PATH)
    graph = get_compiled_graph()
    store.update(job_id, status="running", error=None, stage="router")
    GRAPH_RUNS.inc()
    started = time.perf_counter()
    try:
        initial_state = {
            "topic": topic,
            "as_of": as_of,
            "research_mode": research_mode or "auto",
            "user_id": user_id,
            "memory_note": memory.context(user_id),
            "sections": [],
            "evidence": [],
            "revision_count": 0,
            "max_revision_attempts": APP_CONFIG.max_revision_attempts,
            "enable_images": True,
            "job_id": job_id,
        }
        # tracing_context guarantees one root trace per generation even when
        # individual LLM calls fail, and the tags/metadata make every trace
        # searchable by job and user in LangSmith.
        with tracing_context(
            enabled=True,
            tags=[f"job_id:{job_id}", f"user_id:{user_id}"],
            metadata={
                "job_id": job_id,
                "user_id": user_id,
                "topic": topic,
                "as_of": as_of,
                "research_mode": research_mode,
            },
        ):
            result = graph.invoke(initial_state, config={"configurable": {"thread_id": job_id}})
        content = result.get("final", "")
        if not content:
            raise RuntimeError("Graph completed without a final document.")
        plan = result.get("plan")
        evidence = result.get("evidence", [])
        plan_data = plan.model_dump() if hasattr(plan, "model_dump") else plan
        evidence_data = [item.model_dump() if hasattr(item, "model_dump") else item for item in evidence]
        store.update(
            job_id, status="completed", content=content, error=None, stage="completed",
            plan=plan_data, evidence=evidence_data,
        )
        # Feed the user's cross-blog memory so future generations improve.
        title = ""
        if isinstance(plan_data, dict):
            title = plan_data.get("blog_title", "")
        if not title:
            first_heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = first_heading.group(1).strip() if first_heading else topic[:80]
        memory.remember(
            user_id,
            topic=topic,
            title=title,
            approx_words=len(re.findall(r"\b\w+\b", content)),
            markdown=content,
        )
        GRAPH_COMPLETED.inc()
    except Exception as exc:
        store.update(job_id, status="failed", error=str(exc))
        JOB_FAILURES.inc()
        GRAPH_FAILURES.inc()
        raise
    finally:
        GRAPH_LATENCY.observe(time.perf_counter() - started)
