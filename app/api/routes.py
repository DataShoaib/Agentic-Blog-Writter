from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.schemas import (
    BlogListResponse,
    BlogSummary,
    GenerateRequest,
    GenerateResponse,
    SignupRequest,
    SignupResponse,
    TokenResponse,
)
from app.config import get_secrets
from app.observability.metrics import AUTH_FAILURES, RATE_LIMIT_REJECTIONS
from app.observability.tracing import active_project, auth_state, configure_langsmith
from app.security.auth import authenticate, create_access_token, get_current_user, register_user
from app.services.cache import allow_request, get_redis_store
from app.services.jobs import JobManager

router = APIRouter(prefix="/api/v1")
JOB_MANAGER = JobManager()


def _require_rate_limit(user_id: str) -> None:
    if not allow_request(user_id):
        RATE_LIMIT_REJECTIONS.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": "60"},
        )


@router.post("/auth/signup", response_model=SignupResponse, status_code=201)
def signup(req: SignupRequest):
    if not allow_request(f"signup:{req.username}"):
        RATE_LIMIT_REJECTIONS.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many signup attempts. Please retry later.",
            headers={"Retry-After": "60"},
        )
    if not register_user(req.username, req.password):
        raise HTTPException(status_code=409, detail="Username is already registered.")
    return SignupResponse(username=req.username)


@router.post("/auth/token", response_model=TokenResponse)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    if not allow_request(f"login:{form.username}"):
        RATE_LIMIT_REJECTIONS.inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please retry later.",
            headers={"Retry-After": "60"},
        )
    if not authenticate(form.username, form.password):
        AUTH_FAILURES.inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        access_token = create_access_token(form.username)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the server.",
        ) from exc
    return TokenResponse(access_token=access_token)


@router.post("/generate", response_model=GenerateResponse, status_code=202)
def generate(
    req: GenerateRequest,
    user_id: Annotated[str, Depends(get_current_user)],
):
    _require_rate_limit(user_id)
    as_of = (req.as_of or date.today()).isoformat()
    try:
        job_id = JOB_MANAGER.submit(user_id, req.topic, as_of, req.research_mode)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background job service is unavailable.",
        ) from exc
    return GenerateResponse(job_id=job_id, status="queued")


@router.get("/jobs/{job_id}", response_model=GenerateResponse)
def job(
    job_id: str,
    user_id: Annotated[str, Depends(get_current_user)],
):
    record = JOB_MANAGER.get(job_id, user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return GenerateResponse(
        job_id=job_id,
        status=record["status"],
        content=record.get("content"),
        error=record.get("error"),
        stage=record.get("stage"),
        plan=record.get("plan"),
        evidence=record.get("evidence", []),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
    )


@router.get("/blogs", response_model=BlogListResponse)
def list_blogs(user_id: Annotated[str, Depends(get_current_user)]):
    """Return every blog the current user has generated, newest first.

    Blogs are looked up from their job records server-side, so a user's full
    history is available even after a frontend reload or backend restart.
    """
    blogs = []
    for record in JOB_MANAGER.list_blogs(user_id):
        title = extract_blog_title(record)
        blogs.append(
            BlogSummary(
                job_id=record["job_id"],
                topic=record.get("topic", ""),
                title=title,
                created_at=record.get("created_at"),
                updated_at=record.get("updated_at"),
            )
        )
    return BlogListResponse(blogs=blogs)


def extract_blog_title(record: dict) -> str:
    """Best-effort title from the plan JSON, falling back to the first # heading."""
    plan = record.get("plan")
    if isinstance(plan, dict) and plan.get("blog_title"):
        return plan["blog_title"]
    content = record.get("content") or ""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return record.get("topic", "")[:80]


@router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/health")
def health():
    configure_langsmith()
    secrets = get_secrets()
    redis_ready = get_redis_store().available if secrets.redis_url else False
    jwt_ok = bool(secrets.jwt_secret_key)
    llm_ok = bool(secrets.groq_api_key)
    project = active_project()
    state, auth_detail = auth_state()
    return {
        # With the in-process job fallback, Redis being down no longer makes
        # the service unhealthy: auth plus an LLM key are what generation needs.
        "status": "ok" if jwt_ok and llm_ok else "degraded",
        "authentication": "required",
        "jwt_configured": jwt_ok,
        "llm_configured": llm_ok,
        "redis_configured": bool(secrets.redis_url),
        "redis_ready": redis_ready,
        "jobs_executor": JOB_MANAGER.default_execution_mode,
        "images_enabled": bool(secrets.pollinations_api_key),
        "langsmith_tracing": project is not None,
        "langsmith_project": project,
        "langsmith_auth": state if state == "ok" else (auth_detail or state),
    }
