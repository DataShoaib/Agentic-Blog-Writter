"""Process-wide logging: console + one rotating file (logs/app.log).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

_configured = False

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"
MAX_BYTES = 2_000_000
BACKUP_COUNT = 3

# Third-party packages are chatty at INFO; keep them at WARNING.
_NOISY_LOGGERS = (
    "litellm",
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "google_genai",
    "langsmith",
)


def setup_logging() -> None:
    """Console + rotating logs/app.log (idempotent)."""
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if "pytest" not in sys.modules or os.getenv("LOG_TO_FILE") == "1":
        LOG_DIR.mkdir(exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                LOG_FILE,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
        )

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in handlers:
        handler.setFormatter(fmt)
        logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(level)

    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
