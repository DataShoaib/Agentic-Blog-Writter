# Agentic Content Orchestrator (Agentic Blog Writer)

A production-oriented **LangGraph agentic pipeline** that turns a topic into a fully researched, citation-grounded, multi-section article — with mode routing, parallel section generation, a quality gate with bounded revision, optional AI-generated visuals, and a full auth + job-queue API around it.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-1C3C3C)](https://www.langchain.com/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

---

## Overview

Give it a topic, and the pipeline:

1. **Routes** the request — decides whether it needs live web research or can rely on the model's own knowledge (evergreen mode).
2. **Researches** the topic in parallel via Tavily, when needed.
3. **Plans** a 5–7 section outline with target word counts.
4. **Writes** every section in parallel — up to 8 concurrent workers.
5. **Merges** sections back into order, failing loudly if any planned section is missing.
6. **Quality-gates** the draft with an LLM judge plus deterministic structural checks (section depth, length, citation coverage).
7. **Revises** (bounded to one pass) if the gate fails.
8. **Optionally illustrates** the article with AI-generated images.
9. Serves the result through a JWT-authenticated API with background job processing, Redis caching, and full workflow observability.

**Measured output:** 4,500–5,700-word articles, 6 sections, ~54.6s for evergreen topics vs. ~251.8s for research-backed topics.

---

## Features

### Agentic Pipeline (LangGraph)
- **Mode routing** — an LLM decides `open_book` (needs current info), `hybrid`, or `closed_book` (evergreen), with a recency window per mode (7 / 45 / 3650 days)
- **Parallel research** — up to 4 concurrent Tavily queries, deduplicated by URL and filtered for recency
- **Dynamic fan-out** — a planner produces 5–7 tasks, and LangGraph's `Send` API spins up one parallel worker per section (up to 8 concurrent)
- **Ordered, fail-loud merge** — sections are sorted back into plan order; a missing section raises an error instead of silently shipping an incomplete article
- **Quality gate** — an LLM judge plus deterministic checks (per-section word count, paragraph depth, citation coverage) gate the draft
- **Bounded revision loop** — up to one revision pass, with a shrink-guard that rejects any revision that comes back shorter or drops sections
- **Optional AI visuals** — up to 3 images per article via Gemini image models, with placeholder-based section targeting and graceful failure notes if generation fails
- **Crash-resumable** — LangGraph state is checkpointed to Postgres per `job_id`, so a restart resumes from the last completed node rather than starting over

### LLM Orchestration
- **Multi-provider fallback router** (LiteLLM) — Gemini primary with a 7-model fallback chain (4 Gemini variants + 2 Groq models), swapping automatically on quota/rate-limit errors
- **Separate structured-output chain** — JSON-schema-validated calls (planning, routing, research synthesis) use a Gemini-only chain, since Groq's JSON mode doesn't reliably satisfy strict multi-field Pydantic schemas
- **Defensive JSON repair** — a bracket-stack repair pass recovers malformed JSON from providers that don't always honor the requested schema

### API & Job Processing
- JWT authentication (Argon2 password hashing, access + refresh tokens)
- Background job execution: Redis-backed queue (RQ) with automatic fallback to an in-process thread pool if Redis or a worker is unavailable
- Live stage/progress reporting per job, polled by the frontend
- Whole-article Redis cache (versioned key, 7-day TTL) — a cache hit skips the pipeline entirely
- Fixed-window rate limiting (fail-open if Redis is down)
- Per-user job history and ownership checks (cross-user access returns 404, not 403, to avoid leaking existence)

### Observability
- End-to-end **LangSmith** tracing, with auth verified at startup (a bad key disables tracing rather than silently breaking traces)
- Structured logging with rotating file handlers

### Evaluation Harness
A small, high-signal, *executable* eval suite (distinct from the unit test suite) that runs the real pipeline end-to-end:
- 4 golden cases covering router accuracy, closed-book correctness, citation-heavy generation, and a full research-backed run
- Deterministic checks (workflow success, citation allow-list) plus 3 LLM-as-judge evaluators (router correctness, factuality, final-task success)
- Retry logic for infrastructure failures only (not model-quality failures), with per-case latency and token-cost tracking

### Frontend (Streamlit)
- Signup/login, topic composer with model picker and per-job image API key
- Live pipeline checklist that polls job stage/progress
- Article viewer with tabs for plan, evidence, markdown preview, images, and logs
- Job history with ZIP download

---

## Architecture

```
Client (Streamlit)
   │  JWT Bearer
   ▼
FastAPI
   ├─→ Redis whole-article cache (hit → skip pipeline)
   ├─→ Postgres JobStore (status, ownership, live progress)
   └─→ Background executor
          ├─ Redis + worker available → RQ queue
          └─ otherwise                → in-process thread pool (automatic fallback)
                 ▼
            LangGraph (Postgres-checkpointed, thread_id = job_id)

   START → router ──needs research?──► research
              │                            │
              └──────────► planner ◄───────┘
                              │
                     Send fan-out (dynamic N)
                    worker × up to 8 (parallel)
                              │
                        ordered merge (fail-loud)
                              │
                         quality gate
                       ┌──────┴──────┐
                  revise (≤1)   images → final article
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (state machine, `Send` fan-out, retry policies, conditional edges) |
| Checkpointing | `langgraph-checkpoint-postgres` (crash-resumable per-job state) |
| Backend API | FastAPI, Pydantic v2, `pydantic-settings` |
| LLM Gateway | LiteLLM Router — Gemini primary, Groq + Gemini-variant fallback chain |
| Image Generation | Gemini image models (`google-genai`), with fallback chain |
| Research | Tavily (`tavily-python` / `langchain-tavily`) |
| Job Queue | Redis + RQ, with in-process thread-pool fallback |
| Auth | PyJWT (HS256), `pwdlib` (Argon2) |
| Database | PostgreSQL (jobs, users, LangGraph checkpoints) |
| Frontend | Streamlit |
| Observability | LangSmith tracing |
| Infra | Docker Compose (API, worker, Postgres, Redis), GitHub Actions CI |
| Testing | Pytest (14 files, ~80 tests) |

---

## Project Structure

```
agentic-blog-writer/
├── app/
│   ├── main.py                  # FastAPI app, CORS, static image mount
│   ├── config.py                # Secrets (.env) + committed AppConfig defaults
│   │
│   ├── api/
│   │   ├── routes.py             # REST endpoints (/api/v1/...)
│   │   └── schemas.py            # Request/response models
│   │
│   ├── graph/                    # LangGraph pipeline
│   │   ├── state.py               # GraphState (22 keys)
│   │   ├── schemas.py             # Task/Plan/Evidence/Router/Quality/Image models
│   │   ├── prompts.py             # 8 canonical system prompts
│   │   ├── nodes.py               # 11 node/edge functions — pipeline core
│   │   ├── graph.py               # Graph wiring
│   │   └── checkpointer.py        # Postgres checkpointer singleton
│   │
│   ├── services/
│   │   ├── llm.py                 # LiteLLM gateway + structured output
│   │   ├── jobs.py                # JobStore (Postgres) + JobManager + stream runner
│   │   ├── cache.py               # Redis cache + rate limiter
│   │   ├── images.py              # Gemini image generation + fallback
│   │   ├── search.py              # Tavily search + recency filtering
│   │   ├── citations.py           # Citation allow-list scoring
│   │   ├── users.py               # UserStore (Postgres)
│   │   └── db.py                  # Postgres connection layer
│   │
│   ├── security/
│   │   └── auth.py                # JWT + Argon2 + OAuth2 dependencies
│   │
│   ├── observability/
│   │   ├── logging.py             # Console + rotating file logging
│   │   └── tracing.py             # LangSmith configuration
│   │
│   ├── evaluation/                # Executable eval harness
│   │   ├── runner.py               # 4-case eval loop → results/*.json
│   │   ├── judges.py                # 3 LLM-as-judge evaluators
│   │   ├── deterministic.py         # Pure-Python checks + p95/cost
│   │   └── golden_cases.json        # 4 golden test cases
│   │
│   └── worker.py                  # RQ worker entry point
│
├── frontend/
│   ├── config.py                  # Theme, polling config, pipeline stage labels
│   └── streamlit_app.py           # Full UI
│
├── tests/                          # 14 test files, ~80 tests
├── scripts/
│   └── e2e_check.py                # signup→login→generate→poll smoke test
│
├── Dockerfile
├── docker-compose.yml               # api + worker + postgres + redis
├── requirements.txt
├── pytest.ini
├── .env.example
└── .github/workflows/ci.yml
```

---

## Getting Started

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- API keys: Gemini, Tavily, (optional) Groq, LangSmith

### 1. Clone and configure
```bash
git clone <repo-url>
cd agentic-blog-writer
cp .env.example .env
# Fill in GEMINI_API_KEY, TAVILY_API_KEY, JWT_SECRET_KEY, etc.
```

### 2. Start infrastructure
```bash
docker-compose up -d postgres redis
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API and worker
```bash
uvicorn app.main:app --reload --port 8000
python -m app.worker
```

### 5. Run the frontend
```bash
streamlit run frontend/streamlit_app.py
```

### 6. Try it out
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- UI: `http://localhost:8501`

### Full stack via Docker
```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| POST | `/auth/signup` | rate-limited | 201, duplicate username → 409 |
| POST | `/auth/token` | rate-limited | OAuth2 password login |
| POST | `/auth/refresh` | rate-limited | Refresh → new token pair |
| POST | `/generate` | JWT | Enqueues a job, returns `202` + `job_id` |
| GET | `/jobs/{id}` | JWT (owner) | Live status/progress; other users get `404` |
| GET | `/blogs` | JWT | Per-user article history |
| GET | `/models` | — | Available model fallback order |
| GET | `/health` | — | `ok` iff Redis reachable and JWT configured |

---

## Evaluation Results

A 4-case golden evaluation set, run end-to-end against the live pipeline:

| Check | Result |
|---|---|
| Deterministic workflow success | 4/4 |
| Citation allow-list validation | 4/4 |
| Router ground-truth accuracy | 4/4 |
| Router judge score | 3/3 @ 1.0 |
| Factuality (avg) | 0.967 |
| Final-task-success judge | 0/1 (known issue — see below) |
| p95 latency | 251.8s |
| Average latency | 148.8s |
| Evergreen vs. research-backed latency | 54.6s vs. 251.8s (**4.6× faster** for evergreen topics) |

The eval set was deliberately trimmed from 8 cases / 7 judges to 4 cases / 3 judges — chosen to preserve router, closed-book, citation-heavy, and full end-to-end signal — cutting evaluation cost by roughly **50%** with no loss in coverage.

---

## Testing

```bash
pytest -q
```
14 test files, ~80 tests, 100% passing locally (tests requiring a live database auto-skip if Postgres isn't reachable). CI runs `compileall` + the full suite on every push and PR.

---

## Known Limitations

- The final-task-success judge currently fails on one eval case due to mid-article truncation on very long, table-heavy sections — a known, tracked bug rather than a pipeline design flaw.
- Structured LLM calls (planning, routing) run on a Gemini-only fallback chain — Groq's JSON mode does not reliably satisfy the strict multi-field schemas this pipeline relies on.
- Image generation is a single-provider integration (Gemini); a generation failure degrades gracefully to a note in the article rather than blocking the job.
- No Kubernetes or vector-database dependency by design — the production path intentionally stays on Postgres, Redis, and a single-process worker model for this scale.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Author

**Shoaib** ([@DataShoaib](https://github.com/DataShoaib))
