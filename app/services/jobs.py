from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone

from app.config import APP_CONFIG
from app.graph.graph import build_graph
from app.services import db
from app.services.cache import get_blog_cache

JOB_QUEUE_NAME = "blog-generation"
JOBS_DB_PATH = "jobs"

# Personalization memory: rolling window of recent blogs per user
_MEMORY_HEADINGS_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_MAX_MEMORY_ENTRIES = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_column(raw: str | None) -> dict | list | None:
    """Safely parse a JSON string from a database column."""
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def derive_memory_note(recent_blogs: list[dict]) -> str:
    """Build a personalization note from a user's recent completed blogs.

    Injected into planner/writer prompts so new articles avoid repeating
    angles/titles and maintain tonal consistency. Derived from the jobs
    table — no separate memory store needed.
    """
    if not recent_blogs:
        return ""

    lines = []
    for blog in recent_blogs[:_MAX_MEMORY_ENTRIES]:
        plan = blog.get("plan") or {}
        title = (plan.get("blog_title") or blog.get("topic", ""))[:180]
        topic = blog.get("topic", "")[:160]
        sections = [m.strip() for m in _MEMORY_HEADINGS_RE.findall(blog.get("content", ""))][:6]
        sections_str = ", ".join(sections) if sections else "n/a"
        approx_words = len((blog.get("content") or "").split())

        lines.append(f'- "{title}" (~{approx_words} words) | topic: {topic} | sections: {sections_str}')

    return "\n".join(lines)


class JobStore:
    """Registry for API-level job state and per-user ownership.

    Uses PostgreSQL exclusively via DATABASE_URL. Connection is lazy — the
    first DB operation requires a reachable Postgres; imports never connect.
    """

    def __init__(self, path: str | None = None):
        self.path = path or JOBS_DB_PATH
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self):
        return db.connect(self.path)

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            with self._connect() as conn:
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
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at)"
                )
                conn.commit()
            self._ready = True

    def mark_interrupted_jobs(self) -> None:
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            conn.execute(
                db.q(
                    """
                    UPDATE jobs
                    SET status='failed', error=?, updated_at=?
                    WHERE status='running'
                    """
                ),
                ("Job interrupted by application restart.", _utc_now()),
            )
            conn.commit()

    def create(self, job_id: str, user_id: str, topic: str, as_of: str) -> None:
        self._ensure_ready()
        now = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                db.q(
                    """
                    INSERT INTO jobs(
                        job_id,user_id,status,topic,as_of,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?)
                    """
                ),
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
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            conn.execute(
                db.q(
                    """
                    UPDATE jobs
                    SET status=?, content=COALESCE(?, content), error=?, stage=COALESCE(?, stage),
                        plan_json=COALESCE(?, plan_json), evidence_json=COALESCE(?, evidence_json), updated_at=?
                    WHERE job_id=?
                    """
                ),
                (status, content, error, stage, json.dumps(plan) if plan is not None else None,
                 json.dumps(evidence) if evidence is not None else None, _utc_now(), job_id),
            )
            conn.commit()

    def get(self, job_id: str, user_id: str) -> dict | None:
        """Get job by ID, scoped to user ownership. Returns None if not found or wrong owner."""
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                db.q(
                    """
                    SELECT job_id,user_id,status,topic,as_of,content,error,stage,plan_json,evidence_json,created_at,updated_at
                    FROM jobs
                    WHERE job_id=? AND user_id=?
                    """
                ),
                (job_id, user_id),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["plan"] = _parse_json_column(result.pop("plan_json"))
        result["evidence"] = _parse_json_column(result.pop("evidence_json"))
        return result

    @staticmethod
    def _blog_title(record: dict) -> str:
        plan = record.get("plan") or {}
        title = plan.get("blog_title") if isinstance(plan, dict) else None
        if title:
            return str(title)
        content = record.get("content") or ""
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return record.get("topic") or "Untitled"

    def list_by_user(self, user_id: str, limit: int = 50) -> list[dict]:
        """Completed blogs for one user, newest first (per-user history)."""
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                db.q(
                    """
                    SELECT job_id, topic, status, stage, content, plan_json, created_at, updated_at
                    FROM jobs
                    WHERE user_id=? AND status='completed'
                    ORDER BY created_at DESC
                    LIMIT ?
                    """
                ),
                (user_id, limit),
            ).fetchall()
        blogs: list[dict] = []
        for row in rows:
            record = dict(row)
            record["plan"] = _parse_json_column(record.pop("plan_json"))
            blogs.append(
                {
                    "job_id": record["job_id"],
                    "topic": record["topic"],
                    "title": self._blog_title(record),
                    "created_at": record["created_at"],
                }
            )
        return blogs


class JobManager:
    """Enqueue generation jobs while the shared database stores their durable status."""

    def __init__(self):
        self.store = JobStore()
        self._queue = None
        self._retry = None
        self.execution_mode = "queued"

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
            JOB_QUEUE_NAME,
            connection=Redis.from_url(redis_url),
            default_timeout=APP_CONFIG.job_timeout_seconds,
        )
        self._retry = Retry(max=2, interval=[10, 30])
        return self._queue

    def _enqueue_via_redis(
        self,
        job_id: str,
        user_id: str,
        topic: str,
        as_of: str,
        preferred_model: str | None = None,
    ) -> bool:
        """Put the job on the shared RQ queue."""
        queue = self._get_queue()
        queue.enqueue(
            run_job,
            job_id,
            user_id,
            topic,
            as_of,
            preferred_model,
            job_id=job_id,
            retry=self._retry,
        )
        return True

    def submit(
        self,
        user_id: str,
        topic: str,
        as_of: str,
        preferred_model: str | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        self.store.create(job_id, user_id, topic, as_of)

        cached = get_blog_cache().get(topic, as_of)
        if cached is not None:
            self.store.update(
                job_id,
                status="completed",
                content=cached.content,
                error=None,
                stage="completed",
                plan=cached.plan,
                evidence=cached.evidence,
            )
            self.execution_mode = "cached"
            return job_id

        self.execution_mode = "queued"
        try:
            self._enqueue_via_redis(
                job_id, user_id, topic, as_of, preferred_model
            )
        except Exception:
            # Redis/RQ unavailable: fall back to running the job in a daemon
            # thread inside the API process. The HTTP contract stays intact —
            # the caller immediately receives 202 + job_id and polls
            # /jobs/{job_id} for progress, exactly like the queued path.
            self.execution_mode = "synchronous"

            def _run_inline() -> None:
                try:
                    run_job(job_id, user_id, topic, as_of, preferred_model)
                except Exception as exc:
                    self.store.update(
                        job_id,
                        status="failed",
                        error=str(exc) or "Synchronous generation failed.",
                        stage="run",
                    )

            threading.Thread(target=_run_inline, name=f"job-{job_id}", daemon=True).start()

        return job_id

    def get(self, job_id: str, user_id: str) -> dict | None:
        return self.store.get(job_id, user_id)

    def list_blogs(self, user_id: str) -> list[dict]:
        return self.store.list_by_user(user_id)


def run_job(
    job_id: str,
    user_id: str,
    topic: str,
    as_of: str,
    preferred_model: str | None = None,
) -> None:
    """RQ entry point executed by a separate worker process."""
    store = JobStore()
    graph = build_graph()
    store.update(job_id, status="running", error=None, stage="router")

    try:
        memory_note = ""
        try:
            recent_blogs = store.list_by_user(user_id, limit=_MAX_MEMORY_ENTRIES)
            memory_note = derive_memory_note(recent_blogs)
        except Exception:
            pass
        initial_state = {
            "topic": topic,
            "as_of": as_of,
            "sections": [],
            "evidence": [],
            "revision_count": 0,
            "max_revision_attempts": APP_CONFIG.max_revision_attempts,
            "enable_images": True,
            "job_id": job_id,
            "user_id": user_id,
            "memory_note": memory_note,
            "model": preferred_model,
        }
        config = {"configurable": {"thread_id": job_id}}
        # Resume from the last checkpoint if a previous attempt crashed mid-run:
        # invoke(None) continues from the pending node with checkpointed state.
        # Otherwise (first attempt) start a fresh run with the initial state.
        snapshot = graph.get_state(config)
        if snapshot.next:
            result = graph.invoke(None, config)
        else:
            result = graph.invoke(initial_state, config)
        content = result.get("final", "")
        if not content:
            raise RuntimeError("Graph completed without a final document.")

        plan = result.get("plan")
        evidence = result.get("evidence", [])
        plan_data = plan.model_dump() if hasattr(plan, "model_dump") else plan
        evidence_data = [
            item.model_dump() if hasattr(item, "model_dump") else item for item in evidence
        ]

        store.update(
            job_id,
            status="completed",
            content=content,
            error=None,
            stage="completed",
            plan=plan_data,
            evidence=evidence_data,
        )

        try:
            get_blog_cache().set(
                topic,
                as_of,
                content,
                plan_data,
                evidence_data,
            )
        except Exception:
            pass
    except Exception as exc:
        store.update(job_id, status="failed", error=str(exc))
        raise
