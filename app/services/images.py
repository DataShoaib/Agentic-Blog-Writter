"""Google AI Studio (Gemini) image generation for the blog pipeline.

Uses the ``gemini-2.5-flash-image`` model through the official ``google-genai``
SDK. Every diagram is rendered from a prompt in the requested aspect ratio.
When the primary image model is exhausted (429) or overloaded (503), the
gateway transparently retries alternate image models before degrading.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from google import genai
from google.genai import types

from app.config import APP_CONFIG, get_secrets

logger = logging.getLogger(__name__)

# Gemini image models accept these aspect ratios (matches ImageSpec Literal).
_ASPECT_RATIOS = {"16:9", "9:16", "1:1"}

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


_client: "genai.Client | None" = None


def get_image_client() -> genai.Client:
    """Reusable, singleton client (the SDK closes its HTTP client after use)."""
    global _client
    if _client is None:
        api_key = get_secrets().gemini_api_key or get_secrets().google_api_key
        if not api_key:
            raise RuntimeError(
                "Gemini image generation needs GEMINI_API_KEY (or GOOGLE_API_KEY)."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _is_retryable_image_error(exc: Exception) -> bool:
    """True if a Gemini image-generation failure is transient/retryable.

    The google-genai SDK raises varied exception types and never sets a
    ``_retryable`` flag, so we inspect status codes + message text instead.
    """
    text = str(exc)
    for code in _RETRYABLE_CODES:  # {408, 429, 500, 502, 503, 504}
        if str(code) in text:
            return True
    lowered = text.lower()
    return any(
        hint in lowered
        for hint in (
            "resource_exhausted",
            "rate limit",
            "quota",
            "unavailable",
            "internal error",
        )
    )


def _image_model_candidates() -> list[str]:
    """Primary image model first, then alternates — each may carry its own
    free-tier quota, so a dead primary can still yield an image."""
    configured = APP_CONFIG.image_model
    fallbacks = getattr(get_secrets(), "image_fallback_models", None) or APP_CONFIG.image_fallback_models
    if not fallbacks:
        fallbacks = APP_CONFIG.image_fallback_models
    fallbacks = [m for m in fallbacks if m and m != configured]
    return [m for m in (configured, *fallbacks) if m]


def _generate_bytes(prompt: str, aspect_ratio: str, model: str) -> bytes:
    """One Gemini image-generation call; returns raw image bytes."""
    ratio = aspect_ratio if aspect_ratio in _ASPECT_RATIOS else "16:9"
    response = get_image_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=ratio),
        ),
    )

    parts = getattr(response, "parts", None)
    if not parts and getattr(response, "candidates", None):
        try:
            parts = response.candidates[0].content.parts
        except Exception:
            parts = None
    if not parts:
        raise RuntimeError("Gemini returned no image content (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data
    raise RuntimeError("Gemini response contained no inline image bytes.")


def _retry(fetch: Callable[[], bytes], output_path: Path) -> None:
    """Fetch image bytes, cycling through retryable model candidates with
    exponential backoff. Non-retryable errors raise immediately."""
    models = _image_model_candidates()
    last_error: Exception | None = None
    attempt = 0
    for model in models:
        for sub in range(1, _MAX_ATTEMPTS + 1):
            attempt += 1
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(fetch(model))
                return
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last_error = exc
                logger.warning(
                    "image gen attempt=%d model=%s: %s",
                    attempt,
                    model,
                    str(exc)[:160],
                )
                # On a non-retryable error for THIS model, try the next model.
                if not _is_retryable_image_error(exc):
                    break
                if sub < _MAX_ATTEMPTS:
                    time.sleep(min(2**sub, 8))
    raise RuntimeError(f"Image generation failed after retries: {last_error}")


def generate_image(
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "16:9",
) -> None:
    """Generate one Gemini image and write it to ``output_path``."""
    _retry(lambda model: _generate_bytes(prompt, aspect_ratio, model), output_path)