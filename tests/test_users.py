import pytest
from pwdlib import PasswordHash

from app.services import db
from app.services.users import get_user_store

pytestmark = pytest.mark.usefixtures("requires_db")


@pytest.fixture(autouse=True)
def _clean_users(requires_db):
    """The Postgres users table is shared; start every test from an empty one.

    Depends on ``requires_db`` so the suite skips cleanly when Postgres is not
    reachable (no DB connect is even attempted in that case).
    """
    store = get_user_store()
    with store._lock, store._connect() as conn:
        conn.execute(db.q("DELETE FROM users"))
        conn.commit()
    yield


def test_user_store_creates_and_reads_hashed_user():
    store = get_user_store()
    password_hash = PasswordHash.recommended().hash("secret-password")

    assert store.create("writer", password_hash, "2026-08-20T00:00:00+00:00")
    user = store.get("writer")

    assert user is not None
    assert user["username"] == "writer"
    assert user["password_hash"] != "secret-password"
    assert PasswordHash.recommended().verify("secret-password", user["password_hash"])


def test_user_store_rejects_duplicate_username_case_insensitively():
    store = get_user_store()
    password_hash = PasswordHash.recommended().hash("secret-password")

    assert store.create("writer", password_hash, "2026-08-20T00:00:00+00:00")
    assert not store.create("WRITER", password_hash, "2026-08-20T00:00:00+00:00")
