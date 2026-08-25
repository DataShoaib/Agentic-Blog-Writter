from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from langsmith import traceable

from app.graph.nodes import (
    decide_images,
    fanout,
    generate_and_place_images,
    merge_content,
    planner_node,
    quality_gate,
    research_node,
    revise_content,
    route_after_router,
    route_quality,
    router_node,
    worker_node,
)
from app.graph.state import GraphState
from app.services.search import TransientSearchError


TRANSIENT_NODE_RETRY = RetryPolicy(
    max_attempts=2,
    retry_on=(TransientSearchError, TimeoutError, ConnectionError, OSError),
)


def traced(name: str, fn):
    """Expose every graph node as its own named LangSmith run.

    Without this, LangSmith mostly showed the root run's final input/output and
    nested LLM calls. Wrapping each node makes the whole workflow visible
    start-to-end: router, research, planner, workers, merge, quality, revise,
    images - each with its own inputs, outputs, timing and errors.
    """
    return traceable(name=name, run_type="chain")(fn)


def build_graph(checkpointer=None):
    """Compile the graph with a real checkpointer by default.

    InMemorySaver is ideal for local/demo deployment and tests. For a horizontally
    scaled production deployment, replace it with a database-backed checkpointer.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    graph = StateGraph(GraphState)
    graph.add_node("router", traced("router", router_node))
    graph.add_node("research", traced("research", research_node), retry_policy=TRANSIENT_NODE_RETRY)
    graph.add_node("planner", traced("planner", planner_node))
    graph.add_node("worker", traced("worker", worker_node))
    graph.add_node("merge", traced("merge", merge_content))
    graph.add_node("quality", traced("quality", quality_gate))
    graph.add_node("revise", traced("revise", revise_content))
    graph.add_node("images", traced("images", decide_images))
    graph.add_node("generate_images", traced("generate_images", generate_and_place_images))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"research": "research", "planner": "planner"},
    )
    graph.add_edge("research", "planner")
    graph.add_conditional_edges("planner", fanout, ["worker"])
    graph.add_edge("worker", "merge")
    graph.add_edge("merge", "quality")
    graph.add_conditional_edges(
        "quality",
        route_quality,
        {"revise": "revise", "images": "images"},
    )
    graph.add_edge("revise", "quality")
    graph.add_edge("images", "generate_images")
    graph.add_edge("generate_images", END)

    return graph.compile(checkpointer=checkpointer)
