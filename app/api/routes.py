from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from app.api.schemas import (
    BlogListResponse,
    BlogSummary,
    GenerateRequest,
    GenerateResponse,
    RefreshRequest,
    SignupRequest,
    SignupResponse,
    TokenResponse,
)
from app.config import get_secrets
from app.security.auth import (
    authenticate,
    decode_token,
    get_current_user,
    issue_token_pair,
    register_user,
)
from app.services.cache import allow_request, get_redis_store
from app.services.jobs import JobManager

router = APIRouter(prefix="/api/v1")
JOB_MANAGER = JobManager()


def _require_rate_limit(user_id: str) -> None:
    if not allow_request(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": "60"},
        )


@router.post("/auth/signup", response_model=SignupResponse, status_code=201)
def signup(req: SignupRequest):
    if not allow_request(f"signup:{req.username}"):
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
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please retry later.",
            headers={"Retry-After": "60"},
        )
    user = authenticate(form.username, form.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        access_token, refresh_token = issue_token_pair(str(user["id"]))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the server.",
        ) from exc
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/auth/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest):
    """Exchange a valid refresh token for a fresh (access, refresh) pair."""
    try:
        subject = decode_token(req.refresh_token, expected_type="refresh")
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    _require_rate_limit(f"refresh:{subject}")
    try:
        access_token, refresh_token = issue_token_pair(subject)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the server.",
        ) from exc
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/generate", response_model=GenerateResponse, status_code=202)
def generate(
    req: GenerateRequest,
    user_id: Annotated[str, Depends(get_current_user)],
):
    _require_rate_limit(user_id)
    as_of = (req.as_of or date.today()).isoformat()
    try:
        job_id = JOB_MANAGER.submit(
            user_id,
            req.topic,
            as_of,
            preferred_model=req.preferred_model,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc) or "Background job service is unavailable.",
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
        stage_detail=record.get("stage_detail"),
        progress=record.get("progress"),
        plan=record.get("plan"),
        # evidence_json is NULL for queued/running jobs; the schema only
        # accepts a list, so coerce None to [] here.
        evidence=record.get("evidence") or [],
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
    )


@router.get("/blogs", response_model=BlogListResponse)
def list_blogs(user_id: Annotated[str, Depends(get_current_user)]):
    """Per-user history: every completed blog generated by the caller."""
    return BlogListResponse(blogs=JOB_MANAGER.list_blogs(user_id))


@router.get("/models")
def list_models():
    """Available LLM candidates, in fallback order (frontend model picker)."""
    from app.services.llm import model_candidates

    try:
        return {"models": model_candidates()}
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get("/health")
def health():
    secrets = get_secrets()
    redis_ready = get_redis_store().available if secrets.redis_url else False
    return {
        "status": "ok" if redis_ready and bool(secrets.jwt_secret_key) else "degraded",
        "authentication": "required",
        "redis_configured": bool(secrets.redis_url),
        "redis_ready": redis_ready,
        "jwt_configured": bool(secrets.jwt_secret_key),
        "images_enabled": bool(secrets.gemini_api_key or secrets.google_api_key),
    }
