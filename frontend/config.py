"""Central configuration for the Agentic Writer Streamlit frontend.

Keeping these values here (instead of scattered through app.py) means the
backend URL, theme colors, and pipeline stage labels can be tuned or reused
(e.g. by a future admin page or tests) without touching UI code.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_TITLE = "Agentic Writer"
APP_ICON = "🖋️"
APP_TAGLINE = "research → article"

# ---------------------------------------------------------------------------
# Backend connection
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30

# ---------------------------------------------------------------------------
# Generation polling
# A full run (research, section drafts, quality gate, revision, images) can
# comfortably exceed ten minutes on free-tier model quotas, so the poll loop
# is deliberately generous rather than giving up while the backend is still
# working.
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 1
POLL_MAX_ATTEMPTS = 5400  # ~90 minutes at a 1s poll interval

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
IMAGES_DIR = Path("images")

# ---------------------------------------------------------------------------
# Theme palette — "Slate & Signal"
# Dark slate SaaS/enterprise chrome, flat surfaces, a single blue accent, and
# a clean white article card (no serif/paper texture — sans-serif only).
# ---------------------------------------------------------------------------
class Theme:
    BG = "#0d0f14"
    SIDEBAR_BG = "#0b0d12"
    CARD_BG = "#12151c"
    CARD_BORDER = "#232838"
    INPUT_BG = "#0f1218"

    TEXT_PRIMARY = "#e7e9f0"
    TEXT_SECONDARY = "#b7bdd0"
    TEXT_MUTED = "#7d84a3"

    ACCENT = "#3d8bfd"
    ACCENT_HOVER = "#5b9dfd"
    ACCENT_TEXT_ON_FILL = "#06111f"

    CARD_LIGHT_BG = "#ffffff"
    CARD_LIGHT_BORDER = "#e3e6ec"
    CARD_LIGHT_TEXT = "#14161f"
    CARD_LIGHT_MUTED = "#6b7180"

    SUCCESS = "#5cc98a"
    ERROR = "#e15b5b"


# Streamlit's *native* theme is set via env vars, which must be read before
# `import streamlit` runs. app.py applies these with os.environ.setdefault
# so widgets Streamlit renders itself (dataframe grid, progress bar, radio,
# checkbox, toggle) stay consistent with the custom CSS applied afterwards.
STREAMLIT_THEME_ENV: dict[str, str] = {
    "STREAMLIT_THEME_BASE": "dark",
    "STREAMLIT_THEME_PRIMARY_COLOR": Theme.ACCENT,
    "STREAMLIT_THEME_BACKGROUND_COLOR": Theme.BG,
    "STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR": Theme.CARD_BG,
    "STREAMLIT_THEME_TEXT_COLOR": Theme.TEXT_PRIMARY,
}

# ---------------------------------------------------------------------------
# Pipeline stages shown as the live generation checklist.
# Keys must match the `stage` values written by
# app.services.jobs._stream_graph_to_completion, including the "queued"
# pre-start state and the "skipped_research" marker written when the router
# goes straight to the planner (closed-book / evergreen topics).
# ---------------------------------------------------------------------------
PIPELINE_STEPS: list[tuple[str, str]] = [
    ("router", "Router"),
    ("research", "Research"),
    ("planner", "Planner"),
    ("writing", "Writer"),
    ("merging", "Merge"),
    ("quality_gate", "Quality gate"),
    ("revising", "Revise"),
    ("images", "Images"),
    ("finishing", "Final blog"),
]

# Friendly display names for every raw stage value the backend may report,
# including a couple of back-compat aliases from older worker versions.
STAGE_LABELS: dict[str, str] = {
    "queued": "Queued",
    "router": "Router",
    "research": "Researcher",
    "skipped_research": "Researcher",
    "planner": "Planner",
    "writing": "Writer",
    "merging": "Merge",
    "quality_gate": "Quality gate",
    "revising": "Revise",
    "images": "Images",
    "finishing": "Final blog",
    "completed": "Final blog",
    "failed": "Failed",
    # Back-compat with stage values written by older worker versions.
    "run": "Writer",
    "workers": "Writer",
    "quality gate": "Quality gate",
}