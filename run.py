from __future__ import annotations

from datetime import date

from app.config import APP_CONFIG
from app.graph.graph import build_graph


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke(
        {
            "topic": "How should production RAG systems be evaluated?",
            "as_of": date.today().isoformat(),
            "sections": [],
            "evidence": [],
            "revision_count": 0,
            "max_revision_attempts": APP_CONFIG.max_revision_attempts,
            "enable_images": True,
        },
        {"configurable": {"thread_id": "demo-run"}},
    )
    print(result["final"])
