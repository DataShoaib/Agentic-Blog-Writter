# Agentic Content Orchestrator

A production-oriented research-to-content system built around LangGraph. The point of the project is not to stack agents on top of each other; it is to show a clean workflow that can research current information, split a writing task into parallel sections, validate the result, and return a usable document through an API.

## What it demonstrates

- LangGraph routing, fan-out with `Send`, state reducers, checkpointing and bounded revision
- Three research modes: `closed_book`, `hybrid`, `open_book`
- Evidence normalization, deduplication and recency rules
- Pydantic structured outputs at every decision boundary
- Parallel section workers with deterministic ordering
- Citation allow-list validation
- Quality gate with a bounded revision loop
- Optional contextual diagram generation with Gemini
- Redis/RQ background job execution with a separate worker process, and an in-process thread executor that takes over automatically when Redis is down or no worker is running
- Per-user JWT authentication and job ownership
- Redis-backed search caching and rate limiting, with a local fallback when Redis is unavailable
- SQLite job registry for API polling
- Prometheus metrics and LangSmith-compatible tracing
- Centralized LLM gateway: bounded retries with exponential backoff and ordered model fallbacks
- Executable evaluation dataset
- Tests for API, graph, the LLM gateway, security, jobs, search policy, citations and rate limiting
- Docker + Docker Compose + GitHub Actions CI

## Architecture

```text
Client
  |
  | JWT
  v
FastAPI -----------------------> /metrics
  |
  +--> Redis cache / rate limit
  |
  +--> Job Store (SQLite)
  |
  +--> Background executor
           |
           v
       LangGraph
           |
        Router
       /      \
 Research     Planner
    |            |
 Evidence     Dynamic Send fan-out
 validation   /    |     \
            W1    W2     WN
             \     |     /
               Merge
                 |
             Quality Gate
              /         \
          revise        pass
            |             |
            +-------> Images
                         |
                      Final MD
```

### Why these pieces exist

**Redis/RQ worker:** the API stores the job as `queued` in SQLite and publishes the work to Redis. A separate RQ worker consumes the job, runs the long graph, and updates SQLite with progress, results, or failures. The frontend continues polling the job-status endpoint, while Redis provides the cross-process queue and retry boundary.

**LangGraph checkpointer:** stores the workflow state of every job against its `thread_id` (one thread per `job_id`) as the graph moves `router → research → planner → writers → merge → quality → revise → images`. In this project it runs on a durable SQLite saver (`outputs/checkpoints.sqlite3`), so a worker restart or in-process thread crash mid-generation keeps every checkpoint already written, and concurrent jobs stay isolated instead of sharing RAM state. A deployment that needs multi-process horizontal scaling can point the same code at a PostgreSQL checkpointer with no other changes.

**SQLite job store:** tracks API-level status (`queued`, `running`, `completed`, `failed`) and the owner of each job. It is not a replacement for LangGraph checkpointing; the two solve different problems.

**Redis:** two uses that are actually useful here: short-lived Tavily search caching to reduce repeated latency/cost, and shared rate limiting when multiple API instances use the same Redis service. When Redis is not configured, the app falls back to an in-process rate limiter and simply skips the distributed cache.

**JWT:** protects the API and binds job reads to the authenticated user. Accounts are stored in the local SQLite user database. A production system would usually delegate identity to an external identity provider.

**Centralized LLM gateway:** every node reaches the chat model through `app/services/llm.py`, so one layer owns timeouts, transient-error classification, bounded retries with exponential backoff, and an ordered model fallback chain (`GROQ_MODEL` first, then `GROQ_FALLBACK_MODELS` or the committed defaults). Rate-limit responses honor Groq's "try again in Xs" hint, empty completions are retried like transient faults, and a permanent error switches to the next fallback model immediately instead of burning retries on a broken endpoint.

## Workflow details

### 1. Router

Classifies a topic as:

- `closed_book`: no current web evidence is materially required
- `hybrid`: current examples, tools or documentation improve an otherwise evergreen answer
- `open_book`: latest/news/pricing/policy topics where recency is part of correctness

### 2. Research

Search results are normalized into a small evidence model, deduplicated by URL and filtered by the chosen recency policy.

For `open_book`, undated results are not accepted. For `hybrid`, reliable undated documentation may stay because it can still be authoritative.

### 3. Planner + fan-out

The planner creates 5–9 independent tasks. LangGraph uses `Send` to dispatch the tasks to a common worker node. The state reducer collects `(task_id, markdown)` pairs and sorts them back into plan order before merging.

### 4. Quality gate

The first draft is reviewed for factuality, completeness and citation quality. If it fails, the graph can perform a bounded revision. `MAX_REVISION_ATTEMPTS` prevents a runaway loop and uncontrolled cost.

### 5. Images

The visual planner can choose up to three diagrams. Generated assets are served from `/assets/images/...`; the workflow keeps the article usable even when image generation is unavailable.

## API

Public endpoints:

```text
GET  /
GET  /api/v1/health
GET  /api/v1/metrics
POST /api/v1/auth/token
POST /api/v1/auth/signup
```

Authenticated endpoints:

```text
POST /api/v1/generate
GET  /api/v1/jobs/{job_id}
```

### Signup and login setup

Create an account using the public signup endpoint:

```http
POST /api/v1/auth/signup
Content-Type: application/json

{
  "username": "writer",
  "password": "a-secure-password"
}
```

The account is stored in SQLite. Only the Argon2 password hash is stored. A duplicate username
returns `409 Conflict`.

Then use the same username and password at `/api/v1/auth/token` to receive a JWT.

A convenient way to create a secret is:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

The Swagger UI at `/docs` can then obtain a bearer token from `/api/v1/auth/token`.

### Generate a job

After authorizing in Swagger:

```http
POST /api/v1/generate
Content-Type: application/json
Authorization: Bearer <token>

{
  "topic": "How should production RAG systems be evaluated?",
  "as_of": "2026-08-20"
}
```

The API returns immediately with a `job_id`. Poll:

```http
GET /api/v1/jobs/<job_id>
Authorization: Bearer <token>
```

## Configuration

Copy `.env.example` to `.env`. The real `.env` is ignored by Git and contains
only secrets or deployment-specific connection values. Version-controlled
operational defaults (model names, cache TTL, worker count, revision limit,
request timeout, and image settings) live in `app/config.py`.

### Streamlit frontend

Start the API and frontend in separate terminals:

```bash
uvicorn app.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py
```

The frontend opens at `http://localhost:8501` and uses the API at
`http://127.0.0.1:8000`. Set `API_BASE_URL` when the backend runs elsewhere.

Minimum local secrets for text generation:

```env
GROQ_API_KEY=...
JWT_SECRET_KEY=...
```

Web research:

```env
TAVILY_API_KEY=...
```

Images:

```env
GOOGLE_API_KEY=...
```

Redis (recommended for a multi-instance deployment and enabled by the included Compose file):

```env
REDIS_URL=redis://localhost:6379/0
```

## Run locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker Compose

The included Compose setup gives you the API and Redis service.

```bash
docker compose up --build
```

Then use the Swagger UI at `http://localhost:8000/docs`.

## Evaluation

The evaluator is an executable workflow, not a placeholder. It runs each dataset case through the graph and then asks a structured judge to score the generated article.

```bash
python -m app.evaluation.evaluator
```

The evaluator disables image generation itself so image generation does not dominate cost or latency.

## Observability

Prometheus metrics are exposed at:

```text
GET /api/v1/metrics
```

The project tracks graph runs/failures/latency, LLM calls/failures/retries/model-fallbacks, cache hits/misses, job submissions/failures, queue depth, authentication failures and rate-limit rejections.

Set:

```env
LANGSMITH_API_KEY=...
```

and tracing is switched on at API/worker startup for the entire graph run — not just the LLM calls. Every generation produces one root trace tagged with `job_id:<id>` and `user_id:<user>`, so runs are searchable per job even when individual LLM calls fail. The project defaults to `agentic-content-orchestrator`; override it with `LANGSMITH_PROJECT`.

## Project structure

```text
agentic-content-orchestrator/
|
+-- app/
|   +-- api/
|   |   +-- routes.py
|   |   +-- schemas.py
|   |
|   +-- graph/
|   |   +-- graph.py
|   |   +-- nodes.py
|   |   +-- schemas.py
|   |   +-- state.py
|   |
|   +-- security/
|   |   +-- auth.py
|   |
|   +-- services/
|   |   +-- cache.py
|   |   +-- citations.py
|   |   +-- images.py
|   |   +-- jobs.py
|   |   +-- llm.py
|   |   +-- search.py
|   |
|   +-- evaluation/
|   +-- observability/
|   +-- config.py
|   +-- main.py
|
+-- tests/
+-- images/
+-- outputs/
+-- Dockerfile
+-- docker-compose.yml
+-- requirements.txt
+-- .env.example
+-- .github/workflows/ci.yml
+-- README.md
```

## Production path

The repository intentionally avoids infrastructure that does not solve a real problem for this workload. The next scale step is straightforward:

```text
FastAPI instances
      |
      v
    AWS SQS
      |
      v
Worker instances
      |
      v
LangGraph + durable checkpointer
      |
      +--> PostgreSQL (application data)
      +--> Redis (cache/rate limit)
```

Kafka, Kubernetes and a vector database are not required merely to make this project look larger. They should only be introduced when a real workload justifies them.
