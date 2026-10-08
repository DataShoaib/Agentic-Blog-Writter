"""Render pre-deploy hook: create application tables if they are missing.

Idempotent and NEVER destructive — every statement is
``CREATE TABLE IF NOT EXISTS`` / ``ADD COLUMN IF NOT EXISTS`` via
``JobStore._ensure_ready()`` / ``UserStore._ensure_ready()`` /
``PostgresSaver.setup()``, so re-running on every deploy is safe.

Run with the production DATABASE_URL already injected (render.yaml wires
``preDeployCommand: python scripts/render_init_db.py``). Exits non-zero —
which correctly blocks the deploy — only when the database is unreachable
or misconfigured.
"""
from __future__ import annotations

from app.graph.checkpointer import get_default_checkpointer
from app.services.jobs import JobStore
from app.services.users import get_user_store


def main() -> None:
    JobStore()._ensure_ready()
    get_user_store()._ensure_ready()
    get_default_checkpointer()  # creates the langgraph checkpoint tables on first run
    print("render pre-deploy: jobs/users/checkpoint tables ready")


if __name__ == "__main__":
    main()
