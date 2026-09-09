from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from app.graph.checkpointer import get_default_checkpointer
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


def build_graph(checkpointer=None):
    """Compile the graph with a real checkpointer by default.

    Default selection (see app.graph.checkpointer.get_default_checkpointer):
    - InMemorySaver: local dev and hermetic tests.
    """
    if checkpointer is None:
        checkpointer = get_default_checkpointer()

    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("research", research_node, retry_policy=TRANSIENT_NODE_RETRY)
    graph.add_node("planner", planner_node)
    graph.add_node("worker", worker_node)
    graph.add_node("merge", merge_content)
    graph.add_node("quality", quality_gate)
    graph.add_node("revise", revise_content)
    graph.add_node("images", decide_images)
    graph.add_node("generate_images", generate_and_place_images)

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
