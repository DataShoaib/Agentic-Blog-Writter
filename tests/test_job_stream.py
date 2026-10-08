"""Unit tests for the job-status stream writer — injected fakes, no DB."""
from types import SimpleNamespace

from app.services.jobs import _stream_graph_to_completion


class FakeStore:
    def __init__(self):
        self.progress = 0.0
        self.updates: list[dict] = []

    def update(self, job_id, **kw):
        self.updates.append(kw)
        if kw.get("progress") is not None:
            self.progress = kw["progress"]

    def get_progress(self, job_id) -> float:
        return self.progress


class FakeGraph:
    """Yields prepared {node: output} events; state stays empty (falsy →
    the writer keeps its local accumulator, which is all it needs here)."""

    def __init__(self, events):
        self._events = events

    def stream(self, stream_input, config, stream_mode=None):
        assert stream_mode == "updates"
        yield from self._events

    def get_state(self, config):
        return SimpleNamespace(values={})


def test_progress_never_moves_backwards_across_the_revise_loop():
    """Regression: quality(0.80) → revise(0.84) → quality(0.80) used to
    write 0.80 after 0.84 — the persisted bar jumped backwards."""
    events = [
        {"router": {}},
        {"planner": {}},
        {"merge": {}},
        {"quality": {}},
        {"revise": {}},
        {"quality": {}},  # revisit: NODE_PROGRESS 0.80 < previous peak 0.84
    ]
    store = FakeStore()
    graph = FakeGraph(events)

    _stream_graph_to_completion(
        store,
        graph,
        "job-x",
        {"topic": "t"},
        {"configurable": {"thread_id": "job-x"}},
    )

    written = [u["progress"] for u in store.updates if u.get("progress") is not None]
    assert written == sorted(written), f"progress went backwards: {written}"
    assert written[-1] >= 0.84, "the revise peak must survive the quality revisit"


def test_resume_seeds_from_persisted_progress():
    """A resumed run must not restart the bar at 0.02 when the row already
    holds a high fraction."""
    store = FakeStore()
    store.progress = 0.72  # crashed mid-run after merge
    graph = FakeGraph([{"quality": {}}])

    _stream_graph_to_completion(
        store,
        graph,
        "job-x",
        None,  # resume path
        {"configurable": {"thread_id": "job-x"}},
    )

    written = [u["progress"] for u in store.updates if u.get("progress") is not None]
    assert all(p >= 0.72 for p in written), f"regressed: {written}"