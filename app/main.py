from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import JOB_MANAGER, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
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
