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


def create_access_token(subject: str) -> str:
    secrets = get_secrets()
    if not secrets.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=TOKEN_TTL_MINUTES),
    }
    return jwt.encode(payload, secrets.jwt_secret_key, algorithm=JWT_ALGORITHM)


def authenticate(username: str, password: str) -> bool:
    user = get_user_store().get(username)
    if user:
        return password_hash.verify(password, user["password_hash"])
    return False


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
        payload = jwt.decode(
            token,
            secrets.jwt_secret_key,
            algorithms=[JWT_ALGORITHM],
        )
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject:
            raise InvalidTokenError("Missing subject")
        return subject
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
