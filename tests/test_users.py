from pwdlib import PasswordHash

from app.services.users import UserStore


def test_user_store_creates_and_reads_hashed_user(tmp_path):
    store = UserStore(str(tmp_path / "users.sqlite3"))
    password_hash = PasswordHash.recommended().hash("secret-password")

    assert store.create("writer", password_hash, "2026-08-20T00:00:00+00:00")
    user = store.get("writer")

    assert user is not None
    assert user["username"] == "writer"
    assert user["password_hash"] != "secret-password"
    assert PasswordHash.recommended().verify("secret-password", user["password_hash"])


def test_user_store_rejects_duplicate_username_case_insensitively(tmp_path):
    store = UserStore(str(tmp_path / "users.sqlite3"))
    password_hash = PasswordHash.recommended().hash("secret-password")

    assert store.create("writer", password_hash, "2026-08-20T00:00:00+00:00")
    assert not store.create("WRITER", password_hash, "2026-08-20T00:00:00+00:00")