from __future__ import annotations

import sqlite3
import threading
from pathlib import Path



class UserStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def create(self, username: str, password_hash: str, created_at: str) -> bool:
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
                    (username, password_hash, created_at),
                )
                connection.commit()
        except sqlite3.IntegrityError:
            return False
        return True

    def get(self, username: str) -> dict | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT username, password_hash, created_at FROM users WHERE username=?",
                (username,),
            ).fetchone()
        return dict(row) if row else None


_USER_STORE = UserStore("outputs/users.sqlite3")


def get_user_store() -> UserStore:
    return _USER_STORE
