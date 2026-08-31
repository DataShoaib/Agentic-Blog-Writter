"""Background job management: RQ queue with a transparent in-process fallback.

Execution flow for every generation request:

    submit() -> cache hit?            -> job completes instantly
             -> Redis + live worker?  -> enqueue on RQ ("blog-generation")
             -> otherwise             -> local daemon thread

RQ owns *execution* (queueing, retries, worker isolation). The SQLite
JobStore owns what the *API* needs: per-user ownership, status/progress,
and results that survive Redis TTLs and process restarts.
"""

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
from app.observability.tracing import configure_langsmith
from app.services.blog_cache import get_blog_cache

JOBS_DB_PATH = "outputs/jobs.sqlite3"
logger = logging.getLogger(__name__)

# One compiled graph per process; the SQLite checkpointer inside build_graph()
# keeps each job's workflow state durable across jobs and restarts.
_COMPILED_GRAPH = None


def get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_graph()
    return _COMPILED_GRAPH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class JobStore:
    """SQLite registry for job state and per-user ownership."""

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
        """Create the schema; add missing columns when upgrading an old DB."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS jobs ("
                "job_id TEXT PRIMARY KEY,"
                "user_id TEXT NOT NULL,"
                "status TEXT NOT NULL,"
                "topic TEXT NOT NULL,"
                "as_of TEXT NOT NULL,"
                "content TEXT,"
                "error TEXT,"
                "stage TEXT,"
                "plan_json TEXT,"
                "evidence_json TEXT,"
                "created_at TEXT NOT NULL,"
                "updated_at TEXT NOT NULL)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
            if "user_id" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN user_id TEXT NOT NULL DEFAULT 'legacy'")
            for column in ("stage", "plan_json", "evidence_json"):
                if column not in columns:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at)")
            conn.commit()

    def mark_interrupted_jobs(self) -> None:
        """Fail jobs left 'running' by a restart; they can never finish."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE status='running'",
                ("Job interrupted by application restart.", _utc_now()),
            )
            conn.commit()

    def create(self, job_id: str, user_id: str, topic: str, as_of: str) -> None:
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs(job_id,user_id,status,topic,as_of,created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?)",
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
        """Update a job; COALESCE keeps fields untouched when not passed."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, content=COALESCE(?, content), error=?,"
                " stage=COALESCE(?, stage), plan_json=COALESCE(?, plan_json),"
                " evidence_json=COALESCE(?, evidence_json), updated_at=?"
                " WHERE job_id=?",
                (
                    status, content, error, stage,
                    json.dumps(plan) if plan is not None else None,
                    json.dumps(evidence) if evidence is not None else None,
                    _utc_now(), job_id,
                ),
            )
            conn.commit()

    def get(self, job_id: str, user_id: str) -> dict | None:
        """Fetch a job scoped to its owner; other users must not see it."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT job_id,user_id,status,topic,as_of,content,error,stage,"
                "plan_json,evidence_json,created_at,updated_at"
                " FROM jobs WHERE job_id=? AND user_id=?",
                (job_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json") or "null")
        result["evidence"] = json.loads(result.pop("evidence_json") or "[]")
        return result

    def list_by_user(self, user_id: str) -> list[dict]:
        """Return every completed blog for a user, newest first.

        This is the per-user blog memory: each finished generation is kept
        durably in SQLite (keyed by job_id, which doubles as the graph's
        thread_id) so a user can always browse and reload all their articles.
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT job_id,user_id,status,topic,as_of,content,error,stage,"
                "plan_json,evidence_json,created_at,updated_at"
                " FROM jobs WHERE user_id=? AND status='completed'"
                " ORDER BY updated_at DESC, created_at DESC",
                (user_id,),
            ).fetchall()
        results: list[dict] = []
        for row in rows:
            result = dict(row)
            result["plan"] = json.loads(result.pop("plan_json") or "null")
            result["evidence"] = json.loads(result.pop("evidence_json") or "[]")
            results.append(result)
        return results

class JobManager:
    """Enqueue generation jobs on Redis/RQ with an in-process fallback.

    A separate RQ worker keeps long graph runs isolated from the API, but the
    common single-machine setup has no Redis/worker; then jobs run on a local
    daemon thread so generation still works without extra infrastructure.
    """

    def __init__(self):
        self.store = JobStore(JOBS_DB_PATH)
        self._queue = None
        self._retry = None
        self.execution_mode: str | None = None

    @property
    def default_execution_mode(self) -> str:
        """Executor the next submission will use; useful for health checks."""
        if self.execution_mode:
            return self.execution_mode
        from app.config import get_secrets

        return "redis+rq" if get_secrets().redis_url else "in-process"

    def _get_queue(self):
        if self._queue is None:
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

    def _enqueue_via_redis(
        self, job_id: str, user_id: str, topic: str, as_of: str, research_mode: str
    ) -> bool:
        """Try RQ first; False means the local executor should take over."""
        try:
            queue = self._get_queue()
            from rq import Worker

            # An enqueue without a live worker would strand the job as
            # 'queued' forever; prefer the local thread when nobody consumes.
            if Worker.count(queue=queue) == 0:
                logger.warning(
                    "No RQ worker consumes queue '%s'; using the in-process executor.",
                    queue.name,
                )
                return False
            queue.enqueue(
                run_job, job_id, user_id, topic, as_of, research_mode,
                job_id=job_id, retry=self._retry,
            )
            try:  # queue depth is best-effort observability only
                JOB_QUEUE_DEPTH.set(int(queue.count))
            except Exception:
                pass
            self.execution_mode = "redis+rq"
            return True
        except Exception as exc:
            logger.warning("Redis queue unavailable (%s); using the in-process executor.", exc)
            return False

    def submit(
        self, user_id: str, topic: str, as_of: str, research_mode: str = "auto"
    ) -> str:
        job_id = str(uuid.uuid4())
        self.store.create(job_id, user_id, topic, as_of)

        # Whole-blog cache: same user + topic (+ as_of + mode) returns the
        # finished article instantly instead of re-running the whole graph.
        cached = get_blog_cache().get(user_id, topic, as_of, research_mode or "auto")
        if cached is not None:
            self.store.update(
                job_id, status="completed", content=cached.content,
                stage="completed", plan=cached.plan, evidence=cached.evidence,
            )
            self.execution_mode = "cached"
        elif self._enqueue_via_redis(job_id, user_id, topic, as_of, research_mode):
            pass  # execution_mode already set by _enqueue_via_redis
        else:
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

    def list_blogs(self, user_id: str) -> list[dict]:
        """Return all of a user's completed blogs, newest first."""
        return self.store.list_by_user(user_id)

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
        # individual LLM calls fail; tags/metadata make traces searchable.
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
            result = graph.invoke(
                initial_state, config={"configurable": {"thread_id": job_id}}
            )

        content = result.get("final", "")
        if not content:
            raise RuntimeError("Graph completed without a final document.")
        plan = result.get("plan")
        evidence = result.get("evidence", [])
        dump = lambda x: x.model_dump() if hasattr(x, "model_dump") else x  # noqa: E731
        plan_data = dump(plan)
        evidence_data = [dump(item) for item in evidence]
        store.update(
            job_id, status="completed", content=content, error=None, stage="completed",
            plan=plan_data, evidence=evidence_data,
        )
        _cache_and_remember(
            user_id, topic, as_of, research_mode or "auto", content, plan_data, evidence_data
        )
        GRAPH_COMPLETED.inc()
    except Exception as exc:
        store.update(job_id, status="failed", error=str(exc))
        JOB_FAILURES.inc()
        GRAPH_FAILURES.inc()
        raise
    finally:
        GRAPH_LATENCY.observe(time.perf_counter() - started)


def _cache_and_remember(
    user_id: str,
    topic: str,
    as_of: str,
    research_mode: str,
    content: str,
    plan_data: dict | None,
    evidence_data: list[dict],
) -> None:
    """Persist the blog for instant cache hits and record user memory."""
    from app.services.memory import get_blog_memory

    try:
        get_blog_cache().set(
            user_id, topic, as_of, research_mode, content, plan_data, evidence_data
        )
    except Exception as exc:
        logger.warning("Unable to write blog cache: %s", exc)

    title = (plan_data or {}).get("blog_title", "") if isinstance(plan_data, dict) else ""
    if not title:
        heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = heading.group(1).strip() if heading else topic[:80]
    get_blog_memory().remember(
        user_id,
        topic=topic,
        title=title,
        approx_words=len(re.findall(r"\b\w+\b", content)),
        markdown=content,
    )
