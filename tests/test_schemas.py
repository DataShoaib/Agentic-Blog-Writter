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


def test_generate_request_defaults():
    request = GenerateRequest(topic="Explain production RAG")
    assert request.topic == "Explain production RAG"
    assert request.preferred_model is None
    assert request.enable_images is False
    assert request.image_api_key is None
