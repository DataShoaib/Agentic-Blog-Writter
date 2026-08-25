from app.api.schemas import GenerateRequest
from app.graph.schemas import Plan, Task


def test_task_validation():
    task = Task(
        id=1,
        title="Introduction",
        goal="Explain the basic concept clearly.",
        bullets=["a", "b", "c"],
        target_words=200,
    )
    assert task.target_words == 200


def test_generate_request_parses_date():
    request = GenerateRequest(topic="Explain production RAG", as_of="2026-08-20")
    assert request.as_of.isoformat() == "2026-08-20"
