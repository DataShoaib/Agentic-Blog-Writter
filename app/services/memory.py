"""Per-user memory of recently generated blogs (in-process).

The LangGraph checkpointer keeps workflow state per job thread; this store gives
the planner and writers lightweight awareness of what the same user generated
before, so new articles avoid repeating angles/titles and keep a consistent tone.
"""

from __future__ import annotations

import re
import threading
from collections import deque

_SECTION_HEADINGS = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_MAX_TRACKED_USERS = 500


class BlogMemory:
    def __init__(self, maxlen: int = 5):
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._recent: dict[str, deque] = {}

    def remember(
        self,
        user_id: str,
        *,
        topic: str,
        title: str,
        approx_words: int,
        markdown: str,
    ) -> None:
        headings = [match.strip() for match in _SECTION_HEADINGS.findall(markdown or "")][:12]
        entry = {
            "topic": (topic or "")[:160],
            "title": (title or "")[:180],
            "approx_words": int(approx_words),
            "sections": headings,
        }
        with self._lock:
            bucket = self._recent.setdefault(user_id, deque(maxlen=self._maxlen))
            bucket.appendleft(entry)
            while len(self._recent) > _MAX_TRACKED_USERS:
                self._recent.pop(next(iter(self._recent)))

    def context(self, user_id: str) -> str:
        """Compact summary of the user's recent articles; '' when none exist."""
        with self._lock:
            entries = list(self._recent.get(user_id, ()))
        lines: list[str] = []
        for entry in entries:
            sections = ", ".join(entry["sections"][:6]) or "n/a"
            lines.append(
                f'- "{entry["title"]}" (~{entry["approx_words"]} words) '
                f"| topic: {entry['topic']} | sections: {sections}"
            )
        return "\n".join(lines)


_BLOG_MEMORY = BlogMemory()


def get_blog_memory() -> BlogMemory:
    return _BLOG_MEMORY