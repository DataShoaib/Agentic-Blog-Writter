"""Structural deployment checks — Render blueprint, Dockerfile and env template.

stdlib only (no yaml dependency): these assert on exact config lines, so a
bad edit to render.yaml / Dockerfile / .env.example fails fast in CI.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDER = (ROOT / "render.yaml").read_text(encoding="utf-8")
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
ENV_EXAMPLE = (ROOT / ".env.example").read_text(encoding="utf-8")


def test_render_blueprint_declares_api_worker_postgres_and_keyvalue():
    assert "- type: web" in RENDER
    assert "name: blogwriter-api" in RENDER
    assert "- type: worker" in RENDER
    assert "name: blogwriter-worker" in RENDER
    assert "name: blogwriter-db" in RENDER
    assert "name: blogwriter-cache" in RENDER


def test_render_api_uses_docker_health_check_and_prededeploy():
    assert "runtime: docker" in RENDER
    assert "healthCheckPath: /api/v1/health" in RENDER
    assert "preDeployCommand: python scripts/render_init_db.py" in RENDER


def test_render_worker_uses_existing_command():
    assert "dockerCommand: python -m app.worker" in RENDER


def test_render_binds_urls_from_resources_not_hardcoded_hosts():
    assert "key: DATABASE_URL" in RENDER
    assert "fromDatabase:" in RENDER
    assert "property: connectionString" in RENDER
    assert "key: REDIS_URL" in RENDER
    assert "fromService:" in RENDER
    assert "redis://redis" not in RENDER
    assert "postgres:postgres@postgres" not in RENDER


def test_render_secrets_are_sync_false_not_values():
    for secret in (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "TAVILY_API_KEY",
        "GOOGLE_API_KEY",
        "LANGSMITH_API_KEY",
        "JWT_SECRET_KEY",
    ):
        assert f"- key: {secret}\n        sync: false" in RENDER


def test_dockerfile_expands_render_port_with_local_fallback():
    # exec-JSON form keeps SIGTERM delivery; $PORT expands at container
    # start, 8000 keeps local `docker compose up` working.
    assert "${PORT:-8000}" in DOCKERFILE
    assert "exec uvicorn app.main:app" in DOCKERFILE


def test_env_example_covers_all_configured_secrets_and_no_real_ones():
    for key in (
        "GEMINI_API_KEY=",
        "GROQ_API_KEY=",
        "TAVILY_API_KEY=",
        "GOOGLE_API_KEY=",
        "LANGSMITH_API_KEY=",
        "JWT_SECRET_KEY=",
        "LLM_MODEL=",
        "LLM_FALLBACK_MODELS=",
        "REDIS_URL=",
        "DATABASE_URL=",
        "CORS_ORIGINS=",
    ):
        assert key in ENV_EXAMPLE, f".env.example is missing {key}"


def test_no_hardcoded_secrets_in_deploy_surface():
    import re

    deploy_files = [
        ROOT / "render.yaml",
        ROOT / ".env.example",
        ROOT / "docker-compose.yml",
        ROOT / "Dockerfile",
        ROOT / ".github" / "workflows" / "ci.yml",
    ]
    # Real provider keys are long; placeholders/test values are short.
    suspicious = re.compile(
        r"(AIza[0-9A-Za-z\-_]{20,}|sk-[A-Za-z0-9]{20,}|tavily-[A-Za-z0-9\-]{20,}"
        r"|xox[bpas]-[A-Za-z0-9\-]{10,})"
    )
    for path in deploy_files:
        text = path.read_text(encoding="utf-8")
        assert not suspicious.search(text), f"possible secret in {path}"
    # No assignment of a non-empty credential-looking value anywhere here
    # (value must sit on the SAME line as the key: `\s` must not hop lines).
    assigned = re.compile(
        r"(?m)^\s*(?:key:\s*)?(?:JWT_SECRET_KEY|GEMINI_API_KEY|TAVILY_API_KEY"
        r"|GROQ_API_KEY|GOOGLE_API_KEY|LANGSMITH_API_KEY)\s*[:=][^\S\r\n]*\S[^\r\n]{15,}\s*$"
    )
    for path in deploy_files:
        text = path.read_text(encoding="utf-8")
        assert not assigned.search(text), f"possible assigned secret in {path}"
