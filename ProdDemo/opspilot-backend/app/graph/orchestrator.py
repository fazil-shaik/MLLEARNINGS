"""
The supervisor graph:

    START -> classify_event -> (route) -> {sales|support|ops}_node -> persist_result -> END

Checkpointed against NeonDB, so a run can be inspected, resumed, or replayed
by thread_id — this is what gives OpsPilot durability instead of being a
fire-and-forget script.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from app.graph.state import OpsPilotState
from app.graph.nodes import (
    classify_event,
    run_sales_crew,
    run_support_crew,
    run_ops_crew,
    route_by_department,
    persist_result,
)
from app.database.connection import get_checkpointer_conninfo

_compiled_graph = None


def _build_graph(checkpointer) -> StateGraph:
    graph = StateGraph(OpsPilotState)

    graph.add_node("classify_event", classify_event)
    graph.add_node("sales_node", run_sales_crew)
    graph.add_node("support_node", run_support_crew)
    graph.add_node("ops_node", run_ops_crew)
    graph.add_node("persist_result", persist_result)

    graph.add_edge(START, "classify_event")
    graph.add_conditional_edges(
        "classify_event",
        route_by_department,
        {"sales": "sales_node", "support": "support_node", "ops": "ops_node"},
    )
    graph.add_edge("sales_node", "persist_result")
    graph.add_edge("support_node", "persist_result")
    graph.add_edge("ops_node", "persist_result")
    graph.add_edge("persist_result", END)

    return graph.compile(checkpointer=checkpointer)


def get_graph():
    """Lazily builds and caches the compiled graph with a live Postgres checkpointer."""
    global _compiled_graph
    if _compiled_graph is None:
        with PostgresSaver.from_conn_string(get_checkpointer_conninfo()) as checkpointer:
            checkpointer.setup()  # creates checkpoint tables on first run
        # Reopen a persistent connection for actual use (from_conn_string is a context manager)
        checkpointer = PostgresSaver.from_conn_string(get_checkpointer_conninfo()).__enter__()
        _compiled_graph = _build_graph(checkpointer)
    return _compiled_graph


async def run_event(thread_id: str, event_type: str, payload: dict) -> dict:
    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(
            {"thread_id": thread_id, "event_type": event_type, "payload": payload, "needs_human": False},
            config=config,
        )
        return result
    except NotImplementedError:
        # Checkpointer or persistent backend not available in this environment
        # Return a safe fallback so API endpoints remain usable for development.
        return {"crew_result": {"message": "graph unavailable in this environment"}, "needs_human": False}
    except Exception:
        # On unexpected errors, bubble up so the caller can observe the failure.
        raise
