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
from app.services.images import clear_job_image_key, set_job_image_key

JOB_QUEUE_NAME = "blog-generation"


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


class JobStore:
    """Registry for API-level job state and per-user ownership.

    Uses PostgreSQL exclusively via DATABASE_URL. Connection is lazy — the
    first DB operation requires a reachable Postgres; imports never connect.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False

    def _connect(self):
        return db.connect()

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
                        stage_detail TEXT,
                        progress DOUBLE PRECISION,
                        plan_json TEXT,
                        evidence_json TEXT,
                        executor TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id, created_at)"
                )
                # Light in-place migration for rows created before stage_detail /
                # progress existed: ADD COLUMN IF NOT EXISTS is a safe no-op.
                conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS stage_detail TEXT")
                conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS progress DOUBLE PRECISION")
                # executor: which process owns the run — 'worker' (RQ) or
                # 'api' (in-process thread fallback). NULL = row created
                # before this column existed; treated as worker-owned when
                # sweeping so legacy stragglers are still reaped.
                conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS executor TEXT")
                conn.commit()
            self._ready = True

    def mark_interrupted_jobs(self, *, executor: str | None = None) -> None:
        """Fail running jobs orphaned by a dead process.

        ``executor='worker'`` reaps worker-owned rows (plus legacy NULL rows)
        — called on worker startup. ``executor='api'`` reaps thread-fallback
        rows from a previous API process — called on API startup, where every
        such row is dead by definition (assumes ONE API instance; multi-
        instance deployments should rely on the RQ worker path, see README
        "Production path"). ``None`` reaps every running row.
        """
        self._ensure_ready()
        sql = (
            "UPDATE jobs SET status='failed', error=?, updated_at=? "
            "WHERE status='running'"
        )
        params: tuple = ("Job interrupted by application restart.", _utc_now())
        if executor == "worker":
            sql += " AND (executor IS NULL OR executor='worker')"
        elif executor == "api":
            sql += " AND executor=?"
            params = (*params, executor)
        with self._lock, self._connect() as conn:
            conn.execute(db.q(sql), params)
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
        stage_detail: str | None = None,
        progress: float | None = None,
        plan: dict | None = None,
        evidence: list[dict] | None = None,
        executor: str | None = None,
    ) -> None:
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            conn.execute(
                db.q(
                    """
                    UPDATE jobs
                    SET status=?, content=COALESCE(?, content), error=?, stage=COALESCE(?, stage),
                        stage_detail=COALESCE(?, stage_detail), progress=COALESCE(?, progress),
                        plan_json=COALESCE(?, plan_json), evidence_json=COALESCE(?, evidence_json),
                        executor=COALESCE(?, executor), updated_at=?
                    WHERE job_id=?
                    """
                ),
                (status, content, error, stage, stage_detail, progress, json.dumps(plan) if plan is not None else None,
                 json.dumps(evidence) if evidence is not None else None, executor, _utc_now(), job_id),
            )
            conn.commit()

    def get(self, job_id: str, user_id: str) -> dict | None:
        """Get job by ID, scoped to user ownership. Returns None if not found or wrong owner."""
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                db.q(
                    """
                    SELECT job_id,user_id,status,topic,as_of,content,error,stage,stage_detail,progress,plan_json,evidence_json,created_at,updated_at
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

    def get_progress(self, job_id: str) -> float:
        """Last persisted progress fraction (0.0 when unknown) — seeds the
        monotonic writer so a resumed run never moves the bar backwards."""
        self._ensure_ready()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                db.q("SELECT progress FROM jobs WHERE job_id=?"), (job_id,)
            ).fetchone()
        try:
            return float(row["progress"]) if row and row["progress"] is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

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


def _mark_rq_failure(job, connection, exc_type, exc_value, tb=None, **_ignored) -> None:
    """RQ ``on_failure`` hook — records failures ``run_job`` itself can't.

    A hard timeout SIGKILLs the work horse, so run_job's own ``except``
    never executes and the row would stay 'running' forever. Signature is
    tolerant: RQ passes (job, connection, type, value, traceback).
    """
    try:
        job_id = (job.args or [None])[0]
        if not job_id:
            return
        JobStore().update(
            job_id,
            status="failed",
            error=str(exc_value) or f"Job failed: {getattr(exc_type, '__name__', exc_type)}",
            stage="failed",
            stage_detail="Worker failure.",
        )
    except Exception:
        pass  # never raise inside an RQ callback


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
        enable_images: bool = False,
        image_api_key: str | None = None,
    ) -> bool:
        """Put the job on the shared RQ queue."""
        # Record ownership BEFORE enqueueing: setting it afterwards could
        # race with a fast worker (update() would overwrite 'running' with
        # 'queued'), and a crash between the two writes still leaves a
        # correctly-owned 'queued' row.
        self.store.update(job_id, status="queued", executor="worker")
        queue = self._get_queue()
        queue.enqueue(
            run_job,
            job_id,
            user_id,
            topic,
            as_of,
            preferred_model,
            enable_images,
            image_api_key,
            job_id=job_id,
            retry=self._retry,
            on_failure=_mark_rq_failure,
        )
        return True

    def submit(
        self,
        user_id: str,
        topic: str,
        as_of: str,
        preferred_model: str | None = None,
        enable_images: bool = False,
        image_api_key: str | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        self.store.create(job_id, user_id, topic, as_of)

        cached = get_blog_cache().get(topic, as_of, enable_images=enable_images)
        if cached is not None:
            self.store.update(
                job_id,
                status="completed",
                content=cached.content,
                error=None,
                stage="completed",
                stage_detail="Article ready.",
                progress=1.0,
                plan=cached.plan,
                evidence=cached.evidence,
            )
            self.execution_mode = "cached"
            return job_id

        self.execution_mode = "queued"
        try:
            self._enqueue_via_redis(
                job_id, user_id, topic, as_of, preferred_model, enable_images, image_api_key
            )
        except Exception:
            # Redis/RQ unavailable: fall back to running the job in a daemon
            # thread inside the API process. The HTTP contract stays intact —
            # the caller immediately receives 202 + job_id and polls
            # /jobs/{job_id} for progress, exactly like the queued path.
            self.execution_mode = "synchronous"
            # Ownership recorded BEFORE the thread starts so an API restart
            # can reap this row if the thread dies with the process.
            self.store.update(job_id, status="queued", executor="api")

            def _run_inline() -> None:
                try:
                    run_job(
                        job_id, user_id, topic, as_of, preferred_model, enable_images, image_api_key
                    )
                except Exception as exc:
                    # run_job records its own failure; this is a last-resort
                    # net (e.g. DB error raised before run_job's try block).
                    self.store.update(
                        job_id,
                        status="failed",
                        error=str(exc) or "Synchronous generation failed.",
                        stage="failed",
                    )

            threading.Thread(target=_run_inline, name=f"job-{job_id}", daemon=True).start()

        return job_id

    def get(self, job_id: str, user_id: str) -> dict | None:
        return self.store.get(job_id, user_id)

    def list_blogs(self, user_id: str) -> list[dict]:
        return self.store.list_by_user(user_id)


# LangGraph node -> (job stage, progress 0..1 fraction at node completion).
# The frontend renders these as the live step checklist; the bar position is
# the last completed fraction so it never jumps backwards on retries.
    # Marker stages for the frontend checklist:
    # - "queued"  -> job accepted, worker not picked it up yet
    # - "skipped_research" -> router sent the job straight to planner
    #   (closed-book topic); the frontend renders Research as skipped
    #   instead of showing a fake tick for a node that never ran.
NODE_PROGRESS: dict[str, tuple[str, float]] = {
    "router": ("router", 0.08),
    "research": ("research", 0.22),
    "planner": ("planner", 0.38),
    "worker": ("writing", 0.66),
    "merge": ("merging", 0.72),
    "quality": ("quality_gate", 0.80),
    "revise": ("revising", 0.84),
    "images": ("images", 0.92),
    "generate_images": ("finishing", 0.98),
}

# Node -> human-readable one-liner shown under the progress bar while the next
# node executes.
NODE_STATUS_LABEL: dict[str, str] = {
    "router": "Classifying topic (closed-book / hybrid / open-book)…",
    "research": "Searching the web & building evidence…",
    "planner": "Planning sections…",
    "worker": "Writing sections in parallel…",
    "merge": "Merging sections in order…",
    "quality": "Reviewing factuality & citations…",
    "revise": "Revising thin sections…",
    "images": "Planning visuals…",
    "generate_images": "Generating images & assembling final…",
}


def _serialize_plan(plan) -> dict | None:
    if plan is None:
        return None
    if hasattr(plan, "model_dump"):
        return plan.model_dump()
    if isinstance(plan, dict):
        return plan
    return None


def _serialize_evidence(evidence) -> list[dict]:
    out: list[dict] = []
    for item in evidence or []:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump())
        elif isinstance(item, dict):
            out.append(item)
    return out


def _stream_graph_to_completion(
    store: JobStore,
    graph,
    job_id: str,
    initial_state: dict | None,
    config: dict,
) -> dict:
    """Run the graph event-by-event, persisting stage after every node.

    Uses ``graph.stream(..., stream_mode="updates")`` which yields one
    ``{node_name: node_output}`` dict per completed node. After each event
    the job row is updated so ``GET /jobs/{id}`` — and therefore the
    frontend — shows the current node, a monotonically increasing progress
    fraction, and any plan/evidence produced so far.

    Worker fan-out emits one event per parallel section; ``stage_detail``
    counts them (``Section 3/7``) using the plan's task count when known.

    ``initial_state=None`` resumes from the last checkpoint (crash resume).
    Returns the final full state dict.
    """
    # Do not touch progress here: on a crash-resume the row already holds
    # the last fraction and rewriting 0.02 would move the bar backwards.
    # Seed the monotonic writer from whatever is persisted instead.
    store.update(job_id, status="running", stage="router")
    last_progress = store.get_progress(job_id)
    final_state: dict = {}
    planned_total = 0

    stream_input = None if initial_state is None else initial_state
    for event in graph.stream(stream_input, config, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
        for node_name, node_output in event.items():
            final_state = graph.get_state(config).values or final_state
            stage, done_fraction = NODE_PROGRESS.get(node_name, ("working", 0.5))
            plan_data = _serialize_plan(final_state.get("plan"))
            if plan_data:
                tasks = plan_data.get("tasks") or []
                planned_total = len(tasks) or planned_total
            evidence_data = _serialize_evidence(final_state.get("evidence"))
            detail = NODE_STATUS_LABEL.get(node_name, f"{node_name}…")
            # The router jumped straight to the planner (needs_research=False):
            # keep stage=planner but swap the detail so the UI shows Research
            # was skipped instead of pretending it ran.
            if node_name == "planner" and final_state.get("needs_research") is False:
                detail = "Research skipped (evergreen topic) — planning sections…"
            if node_name == "worker" and planned_total:
                done = len(final_state.get("sections") or [])
                detail = (
                    f"Writing sections ({min(done, planned_total)}/{planned_total})…"
                )
                # Sections land gradually: interpolate writing progress between
                # the planner and merge fractions instead of sitting flat.
                done_fraction = 0.38 + 0.28 * (min(done, planned_total) / planned_total)
            # Monotonic: the revise loop revisits 'quality' (0.80) after
            # 'revise' (0.84), so NODE_PROGRESS ordering alone would let the
            # bar jump backwards. The bar only ever moves forward.
            done_fraction = max(done_fraction, last_progress)
            last_progress = done_fraction
            store.update(
                job_id,
                status="running",
                stage=stage,
                stage_detail=detail,
                progress=round(done_fraction, 3),
                plan=plan_data,
                evidence=evidence_data,
            )

    final_state = graph.get_state(config).values or final_state
    return final_state


def run_job(
    job_id: str,
    user_id: str,
    topic: str,
    as_of: str,
    preferred_model: str | None = None,
    enable_images: bool = False,
    image_api_key: str | None = None,
) -> None:
    """RQ entry point executed by a separate worker process."""
    store = JobStore()
    try:
        graph = build_graph()
        store.update(job_id, status="running", error=None, stage="router")
        # The user's per-job image key must NOT enter GraphState: every state
        # mutation is checkpointed to Postgres. Hold it in a process-local
        # registry for this run instead — the graph executes in THIS process
        # (RQ work horse or the thread fallback) and looks it up by job_id.
        set_job_image_key(job_id, image_api_key)
        initial_state = {
            "topic": topic,
            "as_of": as_of,
            "sections": [],
            "evidence": [],
            "revision_count": 0,
            "max_revision_attempts": APP_CONFIG.max_revision_attempts,
            "enable_images": enable_images,
            "job_id": job_id,
            "user_id": user_id,
            "model": preferred_model,
        }
        config = {"configurable": {"thread_id": job_id}}
        # Resume from the last checkpoint if a previous attempt crashed mid-run:
        # stream(None) continues from the pending node with checkpointed state.
        # Otherwise (first attempt) start a fresh run with the initial state.
        # Streaming (not invoke) so every finished node writes its stage to
        # the job row — the frontend polls /jobs/{id} and renders live nodes.
        snapshot = graph.get_state(config)
        if snapshot.next:
            result = _stream_graph_to_completion(store, graph, job_id, None, config)
        else:
            result = _stream_graph_to_completion(
                store, graph, job_id, initial_state, config
            )
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
            stage_detail="Article ready.",
            progress=1.0,
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
                enable_images=enable_images,
            )
        except Exception:
            pass
    except Exception as exc:
        try:
            store.update(
                job_id,
                status="failed",
                error=str(exc),
                stage="failed",
                stage_detail="Generation failed.",
            )
        except Exception:
            pass  # job store unreachable — RQ's on_failure hook is the backstop
        raise
    finally:
        clear_job_image_key(job_id)
