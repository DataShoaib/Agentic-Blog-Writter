"""Google AI Studio (Gemini) image generation for the blog pipeline.

Gemini is the ONLY image provider — no third-party image service is used.
Images are rendered with the ``gemini-2.5-flash-image`` family through the
official ``google-genai`` SDK. When the primary model is exhausted (429) the
remaining image-capable Gemini models are tried; if none can produce an image
the error is propagated so the node can write a visible note into the article.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from google import genai
from google.genai import types

from app.config import APP_CONFIG, get_secrets

logger = logging.getLogger(__name__)

# Gemini image models accept these aspect ratios (matches ImageSpec Literal).
_ASPECT_RATIOS = {"16:9", "9:16", "1:1"}

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3

# Per-job Gemini keys live here for the duration of one run instead of in
# GraphState: every state mutation is checkpointed to Postgres, and a user's
# API key must never be persisted there. run_job registers the key before the
# graph starts and clears it afterwards; the image nodes look it up by job_id
# (same process — RQ work horse or thread fallback). The GraphState field
# remains as an explicit override for tests and direct graph invocation.
_JOB_IMAGE_KEYS: dict[str, str] = {}


def set_job_image_key(job_id: str, key: str | None) -> None:
    if key:
        _JOB_IMAGE_KEYS[job_id] = key


def get_job_image_key(job_id: str) -> str | None:
    return _JOB_IMAGE_KEYS.get(job_id)


def clear_job_image_key(job_id: str) -> None:
    _JOB_IMAGE_KEYS.pop(job_id, None)


def resolve_image_api_key(api_key: str | None = None) -> str | None:
    """Effective Gemini (Google AI Studio) key for one job.

    A per-job key typed into the frontend wins; otherwise the ``.env``
    ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is used. ``None`` means no key is
    available at all, in which case image generation is skipped entirely.
    """
    return (
        (api_key or "").strip()
        or get_secrets().gemini_api_key
        or get_secrets().google_api_key
        or None
    )


def has_image_api_key(api_key: str | None = None) -> bool:
    """True when an image key is available (per-job override or ``.env``)."""
    return bool(resolve_image_api_key(api_key))


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
    """Primary image model first, then the alternates.

    Each Gemini image model carries its OWN free-tier quota, so a dead primary
    can still yield an image via the retry loop in :func:`generate_image`.
    """
    configured = APP_CONFIG.image_model
    fallbacks = APP_CONFIG.image_fallback_models or ()
    fallbacks = [m for m in fallbacks if m and m != configured]
    return [m for m in (configured, *fallbacks) if m]


def generate_image(
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "16:9",
    api_key: str | None = None,
) -> None:
    """Generate one image with Gemini and write it to ``output_path``.

    All image-capable Gemini models are tried in order. If Gemini cannot
    produce an image the error is propagated so the caller (the image node)
    can inject a visible note into the article instead of failing the job.

    ``api_key`` is the per-job key typed into the frontend; when empty the
    ``.env`` ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` is used.
    """
    key = resolve_image_api_key(api_key)
    if not key:
        raise RuntimeError(
            "Gemini image generation needs GEMINI_API_KEY (or GOOGLE_API_KEY)."
        )
    client = genai.Client(api_key=key)
    models = [m for m in _image_model_candidates() if m]
    if not models:
        raise RuntimeError("No Gemini image model candidates configured.")

    last_error: Exception | None = None
    for model in models:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _generate_bytes_with_client(client, prompt, aspect_ratio, model, output_path)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "image gen attempt=%d model=%s: %s",
                    attempt,
                    model,
                    str(exc)[:160],
                )
                if not _is_retryable_image_error(exc):
                    break
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Image generation failed after retries: {last_error}")


def _generate_bytes_with_client(
    client: genai.Client,
    prompt: str,
    aspect_ratio: str,
    model: str,
    output_path: Path,
) -> None:
    """One Gemini image-generation call using an explicit client instance."""
    ratio = aspect_ratio if aspect_ratio in _ASPECT_RATIOS else "16:9"
    response = client.models.generate_content(
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
            output_path.write_bytes(inline.data)
            return
    raise RuntimeError("Gemini response contained no inline image bytes.")