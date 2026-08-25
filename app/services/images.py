from __future__ import annotations

from pathlib import Path

from app.config import APP_CONFIG, get_secrets


def _extract_inline_image_bytes(response) -> bytes | None:
    parts = getattr(response, "parts", None)
    if not parts:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            try:
                parts = candidates[0].content.parts
            except (AttributeError, IndexError):
                parts = None

    for part in parts or []:
        inline = getattr(part, "inline_data", None)
        data = getattr(inline, "data", None) if inline else None
        if data:
            return data
    return None


def generate_image(prompt: str, output_path: Path, aspect_ratio: str = "16:9") -> None:
    """Generate an image using Gemini 2.5 Flash Image and write raw bytes to disk."""
    secrets = get_secrets()
    if not secrets.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=secrets.google_api_key,
        http_options=types.HttpOptions(
            timeout=APP_CONFIG.image_timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(
                attempts=2,
                initial_delay=1.0,
                max_delay=8.0,
                exp_base=2.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )
    response = client.models.generate_content(
        model=APP_CONFIG.image_model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspectRatio=aspect_ratio),
        ),
    )

    image_bytes = _extract_inline_image_bytes(response)
    if not image_bytes:
        raise RuntimeError("Image model returned no inline image bytes.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)
