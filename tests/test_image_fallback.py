# -*- coding: utf-8 -*-
"""Focused tests for the image pipeline: Gemini-only generation + key handling.

Gemini (Google AI Studio) is the ONLY image provider — no third-party image
API exists in this codebase. Images are opt-in: they are generated only when
the user turns the Images switch ON *and* a Gemini API key is available (the
per-job key typed into the composer, else ``GEMINI_API_KEY`` from ``.env``).
"""
from unittest.mock import MagicMock, patch

import pytest

from app.graph import nodes
from app.graph.schemas import EvidenceItem, Plan, Task
from app.services import images


def _image_response(payload: bytes = b"png-bytes") -> MagicMock:
    """A fake Gemini response carrying exactly one inline image part."""
    part = MagicMock()
    part.inline_data = MagicMock()
    part.inline_data.data = payload
    response = MagicMock()
    response.parts = [part]
    return response


def _mock_client(payload: bytes = b"png-bytes") -> MagicMock:
    """A fake genai.Client whose generate_content returns one image."""
    client = MagicMock()
    client.models.generate_content.return_value = _image_response(payload)
    return client


def _plan() -> Plan:
    return Plan(
        blog_title="How RAG works",
        audience="Engineers",
        tone="Clear",
        tasks=[
            Task(
                id=1,
                title="Retrieval",
                goal="Explain retrieval.",
                bullets=["a", "b", "c"],
                target_words=200,
            )
        ],
    )


def test_generate_image_prefers_override_key(tmp_path):
    """Per-job frontend key overrides GEMINI_API_KEY from .env."""
    output = tmp_path / "img.png"
    client = _mock_client()

    with (
        patch.object(images, "get_secrets") as secrets,
        patch.object(images.genai, "Client", return_value=client) as client_cls,
    ):
        secrets.return_value.gemini_api_key = "env-key"
        secrets.return_value.google_api_key = ""
        images.generate_image("prompt", output, "1:1", api_key="job-key")

    client_cls.assert_called_once_with(api_key="job-key")
    assert output.read_bytes() == b"png-bytes"


def test_generate_image_falls_back_to_env_key(tmp_path):
    """When no per-job key is given, GEMINI_API_KEY is used."""
    output = tmp_path / "img.png"
    client = _mock_client()

    with (
        patch.object(images, "get_secrets") as secrets,
        patch.object(images.genai, "Client", return_value=client) as client_cls,
    ):
        secrets.return_value.gemini_api_key = "env-key"
        secrets.return_value.google_api_key = ""
        images.generate_image("prompt", output, "1:1")

    client_cls.assert_called_once_with(api_key="env-key")
    assert output.read_bytes() == b"png-bytes"


def test_generate_image_raises_without_any_key(tmp_path):
    """If neither per-job nor .env key is present, raise immediately."""
    output = tmp_path / "img.png"
    with patch.object(images, "get_secrets") as secrets:
        secrets.return_value.gemini_api_key = ""
        secrets.return_value.google_api_key = ""
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            images.generate_image("prompt", output, "1:1")


def test_image_model_candidates_are_gemini_only():
    """Every candidate is a Gemini image model, primary first, no duplicates."""
    candidates = images._image_model_candidates()
    assert candidates[0] == images.APP_CONFIG.image_model
    assert all(model.startswith("gemini-") and "image" in model for model in candidates)
    assert len(candidates) == len(set(candidates))


def test_is_retryable_image_error_classifies_provider_failures():
    assert images._is_retryable_image_error(RuntimeError("429 quota exceeded"))
    assert images._is_retryable_image_error(RuntimeError("503 UNAVAILABLE"))
    assert not images._is_retryable_image_error(RuntimeError("404 model not found"))


def test_generate_image_tries_next_model_when_one_fails(tmp_path):
    """A non-retryable failure on one model moves on to the next candidate;
    when every candidate fails, the error is propagated."""
    output = tmp_path / "img.png"
    tried: list[str] = []

    def failing_fetch(client, prompt, aspect_ratio, model, output_path):
        tried.append(model)
        raise RuntimeError("404 model not found")

    with (
        patch.object(images, "_image_model_candidates") as candidates,
        patch.object(images, "get_secrets") as secrets,
        patch.object(images.genai, "Client"),
        patch.object(images, "_generate_bytes_with_client", failing_fetch),
    ):
        candidates.return_value = ["gemini-2.5-flash-image", "gemini-3.1-flash-image"]
        secrets.return_value.gemini_api_key = "key"
        secrets.return_value.google_api_key = ""
        with pytest.raises(RuntimeError, match="Image generation failed after retries"):
            images.generate_image("prompt", output, "16:9")

    assert tried == ["gemini-2.5-flash-image", "gemini-3.1-flash-image"]


def test_decide_images_skips_planning_without_switch_or_key(monkeypatch):
    """No image planning (and no LLM call) when Images is off or has no key."""

    def boom(*args, **kwargs):
        raise AssertionError("image planning must not run")

    monkeypatch.setattr(nodes, "invoke_structured", boom)
    monkeypatch.setattr(nodes, "has_image_api_key", lambda api_key=None: False)

    assert nodes.decide_images(
        {"merged_md": "# T\n", "enable_images": False}
    ) == {"md_with_placeholders": "# T\n", "image_specs": []}
    assert nodes.decide_images(
        {"merged_md": "# T\n", "enable_images": True}
    ) == {"md_with_placeholders": "# T\n", "image_specs": []}


def test_image_node_skips_every_provider_call_when_switch_is_off(tmp_path, monkeypatch):
    """Images OFF (default): no provider call and no [[IMAGE_n]] left behind,
    while the Sources section is still appended."""
    monkeypatch.chdir(tmp_path)
    calls: list[tuple] = []
    monkeypatch.setattr(nodes, "generate_image", lambda *a, **k: calls.append(a))

    out = nodes.generate_and_place_images(
        {
            "plan": _plan(),
            "merged_md": "# Title\n\nBody\n",
            "image_specs": [{"placeholder": "[[IMAGE_1]]"}],
            "enable_images": False,
            "job_id": "job-off",
            "evidence": [EvidenceItem(url="https://example.com/a", title="A")],
        }
    )

    assert calls == []
    assert "[[IMAGE_1]]" not in out["final"]
    assert "## Sources" in out["final"]
    written = (tmp_path / "outputs" / "job-off.md").read_text(encoding="utf-8")
    assert written == out["final"]


def test_image_node_skips_images_when_no_gemini_key(tmp_path, monkeypatch):
    """Images ON but no Gemini key: no provider call, explicit note instead."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nodes, "has_image_api_key", lambda api_key=None: False)
    calls: list[tuple] = []
    monkeypatch.setattr(nodes, "generate_image", lambda *a, **k: calls.append(a))

    out = nodes.generate_and_place_images(
        {
            "plan": _plan(),
            "md_with_placeholders": "# Title\n\ntext\n\n[[IMAGE_1]]\n\ntail\n",
            "image_specs": [
                {
                    "placeholder": "[[IMAGE_1]]",
                    "filename": "diagram.png",
                    "prompt": "p",
                    "alt": "Diagram",
                    "caption": "Caption",
                    "size": "1024x1024",
                }
            ],
            "enable_images": True,
            "image_api_key": None,
            "job_id": "job-nokey",
        }
    )

    assert calls == []
    assert "no Gemini API key was provided" in out["final"]
    assert "[[IMAGE_1]]" not in out["final"]


def test_image_node_generates_and_embeds_image_with_key(tmp_path, monkeypatch):
    """Images ON with a key: the provider is called once and the image is
    embedded in the article."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nodes, "has_image_api_key", lambda api_key=None: True)
    calls: list[tuple] = []

    def fake_generate(prompt, path, aspect_ratio="16:9", api_key=None):
        calls.append((prompt, aspect_ratio, api_key))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    monkeypatch.setattr(nodes, "generate_image", fake_generate)

    out = nodes.generate_and_place_images(
        {
            "plan": _plan(),
            "md_with_placeholders": "# Title\n\ntext\n\n[[IMAGE_1]]\n\ntail\n",
            "image_specs": [
                {
                    "placeholder": "[[IMAGE_1]]",
                    "filename": "diagram.png",
                    "prompt": "Draw the flow",
                    "alt": "Diagram",
                    "caption": "Caption",
                    "size": "1024x1024",
                }
            ],
            "enable_images": True,
            "image_api_key": "job-key",
            "job_id": "job-on",
        }
    )

    assert calls == [("Draw the flow", "1:1", "job-key")]
    assert "![Diagram](/assets/images/job-on/diagram.png)" in out["final"]
    assert "[[IMAGE_1]]" not in out["final"]

