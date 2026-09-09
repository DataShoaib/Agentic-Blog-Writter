from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

from app.config import get_secrets
from app.services.users import get_user_store

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
JWT_ALGORITHM = "HS256"
TOKEN_TTL_MINUTES = 60
REFRESH_TOKEN_TTL_DAYS = 7


def _encode_token(subject: str, *, token_type: str, ttl: timedelta) -> str:
    secrets = get_secrets()
    if not secrets.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": now,
        "exp": now + ttl,
    }
    return jwt.encode(payload, secrets.jwt_secret_key, algorithm=JWT_ALGORITHM)


def create_access_token(subject: str) -> str:
    return _encode_token(subject, token_type="access", ttl=timedelta(minutes=TOKEN_TTL_MINUTES))


def create_refresh_token(subject: str) -> str:
    return _encode_token(subject, token_type="refresh", ttl=timedelta(days=REFRESH_TOKEN_TTL_DAYS))


def issue_token_pair(subject: str) -> tuple[str, str]:
    """Fresh (access, refresh) pair — used by the login and refresh endpoints."""
    return create_access_token(subject), create_refresh_token(subject)


def decode_token(token: str, *, expected_type: str = "access") -> str:
    """Validate a JWT and return its subject.

    Enforces the `type` claim so a refresh token can never be used as an
    access credential (and vice versa). Raises InvalidTokenError on any
    problem; the HTTP layer translates that into 401 responses.
    """
    secrets = get_secrets()
    if not secrets.jwt_secret_key:
        raise InvalidTokenError("JWT authentication is not configured.")
    payload = jwt.decode(token, secrets.jwt_secret_key, algorithms=[JWT_ALGORITHM])
    if payload.get("type") != expected_type:
        raise InvalidTokenError(f"Expected a {expected_type} token.")
    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject:
        raise InvalidTokenError("Missing subject")
    return subject


def authenticate(username: str, password: str) -> dict | None:
    """Return the user row (with numeric id) on success, None on failure."""
    user = get_user_store().get(username)
    if user and password_hash.verify(password, user["password_hash"]):
        return user
    return None


def register_user(username: str, password: str) -> bool:
    return get_user_store().create(
        username=username,
        password_hash=password_hash.hash(password),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> str:
    secrets = get_secrets()
    if not secrets.jwt_secret_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="JWT authentication is not configured.",
        )
    try:
        return decode_token(token, expected_type="access")
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
