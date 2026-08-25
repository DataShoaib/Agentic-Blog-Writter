"""End-to-end smoke test: signup -> login -> generate -> poll -> report.

Usage:
    python scripts/e2e_check.py [topic]

Exits non-zero when generation fails so it can be used in CI or locally.
"""

from __future__ import annotations

import sys
import time
import uuid

import requests

BASE = "http://127.0.0.1:8000"
TOPIC = sys.argv[1] if len(sys.argv) > 1 else "How does self-attention work in transformers?"
RESEARCH_MODE = sys.argv[2] if len(sys.argv) > 2 else "auto"
POLL_TIMEOUT = 600


def main() -> int:
    user = f"smoke_{uuid.uuid4().hex[:8]}"
    password = "SmokeTest#2026"

    r = requests.post(f"{BASE}/api/v1/auth/signup", json={"username": user, "password": password}, timeout=15)
    print(f"signup: {r.status_code} {r.json()}")
    assert r.status_code == 201, r.text

    r = requests.post(f"{BASE}/api/v1/auth/token", data={"username": user, "password": password}, timeout=15)
    print(f"login : {r.status_code}")
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Wrong password must be rejected.
    bad = requests.post(f"{BASE}/api/v1/auth/token", data={"username": user, "password": "wrong-password"}, timeout=15)
    print(f"bad login rejected: {bad.status_code} (expect 401)")
    assert bad.status_code == 401

    health = requests.get(f"{BASE}/api/v1/health", timeout=10).json()
    print(f"health: executor={health.get('jobs_executor')} langsmith={health.get('langsmith_tracing')}")

    r = requests.post(
        f"{BASE}/api/v1/generate",
        headers=headers,
        json={"topic": TOPIC, "research_mode": RESEARCH_MODE},
        timeout=30,
    )
    print(f"generate: {r.status_code} {r.json()}")
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    start = time.time()
    seen_stage = None
    while time.time() - start < POLL_TIMEOUT:
        time.sleep(3)
        job = requests.get(f"{BASE}/api/v1/jobs/{job_id}", headers=headers, timeout=15).json()
        stage = job.get("stage") or job["status"]
        if stage != seen_stage:
            print(f"  [{int(time.time()-start):>4}s] status={job['status']} stage={stage}")
            seen_stage = stage
        if job["status"] == "completed":
            content = job.get("content", "")
            words = len(content.split())
            plan = job.get("plan") or {}
            planned = sum(t.get("target_words", 0) for t in plan.get("tasks", []))
            headings = [line.lstrip("# ").strip() for line in content.splitlines() if line.startswith("## ")]
            print("COMPLETED in", int(time.time() - start), "s")
            print(f"content: {len(content)} chars | {words} words | planned={planned} words")
            print(f"evidence items: {len(job.get('evidence') or [])} | sections: {len(headings)}")
            print("sections:", " | ".join(headings[:10]))
            return 0
        if job["status"] == "failed":
            print(f"FAILED at stage={stage}: {job.get('error')}")
            return 1
    print(f"TIMEOUT after {POLL_TIMEOUT}s; job_id={job_id}")
    return 2


if __name__ == "__main__":
    sys.exit(main())