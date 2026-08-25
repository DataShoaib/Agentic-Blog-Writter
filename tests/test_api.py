from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import JOB_MANAGER
from app.security.auth import get_current_user


def override_user():
    return "test-user"


app.dependency_overrides[get_current_user] = override_user


def test_generate_returns_queued_job(monkeypatch):
    monkeypatch.setattr(
        JOB_MANAGER,
        "submit",
        lambda user_id, topic, as_of, research_mode="auto": "job-test-1",
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/generate",
        json={"topic": "Explain production RAG evaluation", "as_of": "2026-08-20"},
    )
    assert response.status_code == 202
    assert response.json()["job_id"] == "job-test-1"
    assert response.json()["status"] == "queued"


def test_job_is_not_visible_to_another_user(monkeypatch):
    monkeypatch.setattr(
        JOB_MANAGER,
        "get",
        lambda job_id, user_id: None if user_id != "owner" else {"job_id": job_id, "status": "completed"},
    )
    app.dependency_overrides[get_current_user] = lambda: "other"
    client = TestClient(app)
    response = client.get("/api/v1/jobs/job-1")
    assert response.status_code == 404
    app.dependency_overrides[get_current_user] = override_user


def test_metrics_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert "agentic_graph_runs_total" in response.text
