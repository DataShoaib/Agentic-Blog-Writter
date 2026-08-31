"""Image generation for the blog pipeline.

Two providers, chosen per image:

1. **Mermaid.ink** (preferred for diagrams) — renders Mermaid diagram code into
   a PNG with crisp, perfectly readable text labels. Free, no API key needed.
   Used whenever an ``ImageSpec`` carries a ``mermaid`` field.
2. **Pollinations** (fallback for conceptual/decorative images) — text-to-image
   via a single GET request. Quality on the free tier is limited; the app reads
   POLLINATIONS_API_KEY when set, which unlocks better models.
"""

from __future__ import annotations

import base64
import json
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import requests

from app.config import APP_CONFIG, get_secrets

POLLINATIONS_IMAGE_URL = "https://image.pollinations.ai/prompt/{prompt}"
MERMAID_INK_URL = "https://mermaid.ink/img/{state}"

# Native pixel sizes per supported aspect ratio (ImageSpec's Literal values).
# 1280px keeps diagrams crisp on modern displays.
_ASPECT_SIZES = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1280, 1280),
}

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3


def _fetch_image(url: str, params: dict | None = None) -> bytes:
    """One HTTP call; raises on permanent errors, flags transient ones."""
    response = requests.get(url, params=params, timeout=APP_CONFIG.image_timeout_seconds)
    if response.status_code in _RETRYABLE_STATUS_CODES:
        raise TimeoutError(f"Image provider returned transient status {response.status_code}.")
    response.raise_for_status()
    if "image" not in response.headers.get("content-type", ""):
        raise RuntimeError("Image provider returned a non-image response.")
    return response.content


_UNSAFE_LABEL_CHARS = re.compile(r"[^\x20-\x7E]")  # keep printable ASCII only


def _sanitize_mermaid(mermaid_code: str) -> str:
    """Strip characters LLMs like to add that mermaid.ink cannot parse
    (middle dots, superscripts, smart quotes, emoji, etc.)."""
    cleaned = _UNSAFE_LABEL_CHARS.sub(" ", mermaid_code)
    # Collapse runs of spaces left behind by removals.
    return re.sub(r"[ \t]{2,}", " ", cleaned)


def _render_mermaid(mermaid_code: str) -> bytes:
    """Render Mermaid code to an image via mermaid.ink (free, no key)."""
    state = base64.urlsafe_b64encode(
        json.dumps(
            {"code": _sanitize_mermaid(mermaid_code), "mermaid": {"theme": "default"}}
        ).encode("utf-8")
    ).decode("ascii")
    return _fetch_image(MERMAID_INK_URL.format(state=state))


def _fetch_pollinations_once(prompt: str, params: dict) -> bytes:
    return _fetch_image(
        POLLINATIONS_IMAGE_URL.format(prompt=quote(prompt, safe="")), params
    )


def _retry(fetch: Callable[[], bytes], output_path: Path) -> None:
    """Fetch image bytes up to _MAX_ATTEMPTS, retrying only transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            image_bytes = fetch()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)
            return
        except Exception as exc:
            if not isinstance(exc, TimeoutError):
                raise
            last_error = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Image generation failed after retries: {last_error}")


def generate_image(
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "16:9",
    mermaid: str = "",
) -> None:
    """Generate one image and write it to ``output_path``.

    If ``mermaid`` code is provided the diagram is rendered via mermaid.ink
    (crisp text labels); if that render fails, or no mermaid code is given,
    the prompt falls back to Pollinations.
    """
    if mermaid.strip():
        try:
            _retry(lambda: _render_mermaid(mermaid), output_path)
            return
        except Exception:
            pass  # fall through to Pollinations

    secrets = get_secrets()
    width, height = _ASPECT_SIZES.get(aspect_ratio, _ASPECT_SIZES["16:9"])
    params: dict = {
        "width": width,
        "height": height,
        "model": APP_CONFIG.image_model,
        "nologo": "true",
        "seed": random.randint(0, 10**9),
    }
    if secrets.pollinations_api_key:
        params["token"] = secrets.pollinations_api_key
    _retry(lambda: _fetch_pollinations_once(prompt, params), output_path)

