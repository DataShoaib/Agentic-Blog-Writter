import jwt
import pytest
from fastapi import HTTPException

from app.config import get_secrets
from app.security import auth


def test_authentication_helpers(monkeypatch):
    secrets = get_secrets()
    monkeypatch.setattr(secrets, "jwt_secret_key", "test-secret")

    class UserStore:
        def get(self, username):
            return {"password_hash": auth.password_hash.hash("secret")} if username == "demo" else None

    monkeypatch.setattr(auth, "get_user_store", lambda: UserStore())
    assert auth.authenticate("demo", "secret")
    assert not auth.authenticate("missing", "secret")
    token = auth.create_access_token("demo")
    payload = jwt.decode(token, "test-secret", algorithms=[auth.JWT_ALGORITHM])
    assert payload["sub"] == "demo"
    assert payload["type"] == "access"


def test_refresh_token_round_trip_and_type_enforcement(monkeypatch):
    secrets = get_secrets()
    monkeypatch.setattr(secrets, "jwt_secret_key", "test-secret")

    refresh_token = auth.create_refresh_token("demo")
    payload = jwt.decode(refresh_token, "test-secret", algorithms=[auth.JWT_ALGORITHM])
    assert payload["sub"] == "demo"
    assert payload["type"] == "refresh"

    # A refresh token must NEVER be accepted where an access token is required.
    with pytest.raises(auth.InvalidTokenError):
        auth.decode_token(refresh_token, expected_type="access")
    assert auth.decode_token(refresh_token, expected_type="refresh") == "demo"

    # ...and vice versa: an access token is not a refresh credential.
    access_token = auth.create_access_token("demo")
    with pytest.raises(auth.InvalidTokenError):
        auth.decode_token(access_token, expected_type="refresh")
    assert auth.decode_token(access_token, expected_type="access") == "demo"


def test_issue_token_pair_returns_valid_pair(monkeypatch):
    secrets = get_secrets()
    monkeypatch.setattr(secrets, "jwt_secret_key", "test-secret")

    access_token, refresh_token = auth.issue_token_pair("demo")
    assert auth.decode_token(access_token, expected_type="access") == "demo"
    assert auth.decode_token(refresh_token, expected_type="refresh") == "demo"


def test_expired_token_is_rejected(monkeypatch):
    secrets = get_secrets()
    monkeypatch.setattr(secrets, "jwt_secret_key", "test-secret")

    from datetime import datetime, timedelta, timezone

    expired = jwt.encode(
        {"sub": "demo", "type": "access", "exp": datetime.now(timezone.utc) - timedelta(hours=2)},
        "test-secret",
        algorithm=auth.JWT_ALGORITHM,
    )
    with pytest.raises(auth.InvalidTokenError):
        auth.decode_token(expired, expected_type="access")
