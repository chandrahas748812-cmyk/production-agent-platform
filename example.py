"""End-to-end demo: route tasks, run the graph, stream SSE events.

Uses MockLLMBackend (no API keys). For code tasks the HITL gate
auto-approves via the injected reviewer so the demo runs unattended.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agents import MockLLMBackend
from src.config import load_config
from src.graph import build_platform_graph
from src.hitl import ApprovalGate
from src.router import route_task
from src.state import AgentState
from src.streaming import stream_graph_events

CONFIG = load_config(Path(__file__).resolve().parent / "config" / "agents.yaml")


def auto_approve(request):
    print(f"  [hitl] auto-approving '{request.action_summary}' for demo")
    return "approve", None


def demo(task: str, thread_id: str):
    print(f"\n{'=' * 60}\nTask: {task}")
    print(f"Routed to: {route_task(task, CONFIG.routing).value}")

    gate = ApprovalGate(CONFIG.hitl, reviewer=auto_approve)
    graph = build_platform_graph(CONFIG, llm=MockLLMBackend(), approval_gate=gate)
    cfg = {"configurable": {"thread_id": thread_id}}

    sse_chunks = 0
    for chunk in stream_graph_events(graph.stream(AgentState(task=task), config=cfg)):
        sse_chunks += 1  # in production, yield these to the HTTP response

    final = graph.get_state(cfg).values
    state = final if isinstance(final, AgentState) else AgentState(**final)
    print(f"SSE chunks emitted: {sse_chunks}")
    print(f"Final answer:\n{state.final_answer}")
    print(f"Trace: {state.trace_id} | Approval requested: {state.approval_requested}")


if __name__ == "__main__":
    demo("Research the trade-offs between Pinecone and pgvector", "demo-research")
    demo("Implement a retry decorator with exponential backoff", "demo-code")
