import jwt

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
