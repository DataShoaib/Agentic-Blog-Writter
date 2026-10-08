from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import JOB_MANAGER, router
from app.observability.logging import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Thread-fallback jobs from a previous API process died with it — fail
    # their rows so the frontend doesn't poll a phantom 'running' job forever.
    # RQ-owned rows are untouched (the worker reaps those on its own startup).
    # Assumes a single API instance; multi-instance deployments should rely on
    # the RQ worker path (see README "Production path").
    try:
        from app.services.jobs import JobStore

        JobStore().mark_interrupted_jobs(executor="api")
    except Exception:
        pass  # Postgres unreachable at boot — the worker-side sweep still covers us
    yield


def _cors_origins() -> list[str]:
    """Allowed browser origins. ``CORS_ORIGINS`` (comma-separated) overrides
    the local-dev defaults when the API is deployed (e.g. Render) and the
    frontend runs elsewhere."""
    raw = os.environ.get("CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]


app = FastAPI(
    title="Agentic Content Orchestrator",
    version="2.0.0",
    description=(
        "A research-to-content workflow with LangGraph orchestration, background jobs, "
        "quality gates, security, caching and observability."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets/images", StaticFiles(directory="images", check_dir=False), name="images")
app.include_router(router)


@app.get("/")
def root():
    return {
        "name": "Agentic Content Orchestrator",
        "version": "2.0.0",
        "docs": "/docs",
        "auth": "/api/v1/auth/token",
    }
