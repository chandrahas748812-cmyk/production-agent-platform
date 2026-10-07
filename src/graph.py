"""Supervisor-pattern LangGraph with SQLite checkpointing and HITL gates.

Flow:
    supervisor (classify + plan)
        ├─ research tasks → researcher → reviewer
        ├─ code tasks     → planner → coder → [HITL gate] → reviewer
        └─ review tasks   → reviewer
    All nodes checkpoint to SQLite; runs resume via thread_id.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - older langgraph
    SqliteSaver = None  # type: ignore

from .agents import MockLLMBackend, run_planner, run_reviewer, run_specialist
from .config import AgentConfig
from .hitl import ApprovalDecision, ApprovalGate, needs_approval
from .router import route_task
from .state import AgentState, RiskTier, TaskType
from .tracing import Tracer


def build_platform_graph(
    config: AgentConfig,
    llm=None,
    approval_gate: Optional[ApprovalGate] = None,
    checkpointer=None,
):
    """Build the compiled supervisor graph."""
    llm = llm or MockLLMBackend()
    gate = approval_gate or ApprovalGate(config.hitl)
    tracer = Tracer(config.tracing)
    definitions = config.agents

    # ---------- nodes ----------
    def supervisor(state: AgentState) -> Dict[str, Any]:
        task_type = route_task(state.task, config.routing)
        return {
            "task_type": task_type.value,
            "current_agent": "supervisor",
            "trace_id": state.trace_id or str(uuid.uuid4())[:8],
            "messages": [{"role": "supervisor", "content": f"classified as {task_type.value}"}],
        }

    def planner(state: AgentState) -> Dict[str, Any]:
        subtasks = run_planner(state.task, definitions["planner"], llm)
        return {
            "plan": [s.model_dump() for s in subtasks],
            "current_agent": "planner",
            "messages": [{"role": "planner", "content": f"planned {len(subtasks)} subtasks"}],
        }

    def researcher(state: AgentState) -> Dict[str, Any]:
        instruction = state.task if not state.plan else state.plan[0]["instruction"]
        result = run_specialist("researcher", instruction, definitions, llm)
        return {
            "current_agent": "researcher",
            "risk_tier": RiskTier.LOW.value,
            "messages": [{"role": "researcher", "content": result}],
        }

    def coder(state: AgentState) -> Dict[str, Any]:
        instruction = state.task if not state.plan else next(
            (s["instruction"] for s in state.plan if s["agent"] == "coder"), state.task
        )
        result = run_specialist("coder", instruction, definitions, llm)
        return {
            "current_agent": "coder",
            "risk_tier": RiskTier.HIGH.value,
            "messages": [{"role": "coder", "content": result}],
        }

    def reviewer(state: AgentState) -> Dict[str, Any]:
        last = state.messages[-1]["content"] if state.messages else ""
        verdict, escalated = run_reviewer(last, definitions, llm)
        tier = RiskTier.MEDIUM.value if escalated else RiskTier.LOW.value
        return {
            "current_agent": "reviewer",
            "risk_tier": tier,
            "messages": [{"role": "reviewer", "content": verdict}],
        }

    def hitl_gate(state: AgentState) -> Dict[str, Any]:
        """Human approval for high-risk outputs before they are applied."""
        last = state.messages[-1]["content"] if state.messages else ""
        tier = state.risk_tier
        escalated = tier == RiskTier.MEDIUM.value and "ESCALATE" in last.upper()
        if not needs_approval(tier, escalated, config.hitl):
            return {"approval_requested": False, "approval_decision": ApprovalDecision.APPROVE.value}
        result = gate.request(
            agent=state.current_agent,
            risk_tier=tier,
            action_summary=f"Apply {state.current_agent} output",
            payload_preview=last,
            escalated=escalated,
        )
        return {
            "approval_requested": True,
            "approval_decision": result.decision.value,
            "human_feedback": result.feedback,
            "messages": [{"role": "hitl", "content": f"decision={result.decision.value}"}],
        }

    def finalize(state: AgentState) -> Dict[str, Any]:
        if state.approval_requested and state.approval_decision != ApprovalDecision.APPROVE.value:
            answer = (
                f"Action NOT applied — human decision: {state.approval_decision}."
                + (f" Feedback: {state.human_feedback}" if state.human_feedback else "")
            )
        else:
            parts = [m["content"] for m in state.messages if m["role"] in {"researcher", "coder", "reviewer"}]
            answer = "\n\n".join(parts) if parts else "No output produced."
        return {"final_answer": answer, "current_agent": "supervisor"}

    # ---------- edges ----------
    def route_after_supervisor(state: AgentState) -> str:
        return {
            TaskType.RESEARCH.value: "researcher",
            TaskType.CODE.value: "planner",
            TaskType.REVIEW.value: "reviewer",
            TaskType.PLAN.value: "planner",
            TaskType.UNKNOWN.value: "researcher",
        }[state.task_type]

    builder = StateGraph(AgentState)
    for name, fn in [
        ("supervisor", supervisor),
        ("planner", planner),
        ("researcher", researcher),
        ("coder", coder),
        ("reviewer", reviewer),
        ("hitl_gate", hitl_gate),
        ("finalize", finalize),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_after_supervisor,
                                 {"researcher": "researcher", "planner": "planner", "reviewer": "reviewer"})
    # code path: plan -> code -> gate -> review -> finalize
    builder.add_edge("planner", "coder")
    builder.add_edge("coder", "hitl_gate")
    builder.add_edge("hitl_gate", "reviewer")
    # research path: research -> review -> finalize
    builder.add_edge("researcher", "reviewer")
    builder.add_edge("reviewer", "finalize")
    builder.add_edge("finalize", END)

    if checkpointer is None and SqliteSaver is not None:
        db_path = config.checkpointing.get("path", "./checkpoints.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)


def run_task(
    task: str,
    config: AgentConfig,
    thread_id: str = "default",
    llm=None,
    approval_reviewer: Optional[Callable] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream graph events for a task. Resumable via thread_id (SQLite)."""
    from .hitl import ApprovalGate

    gate = ApprovalGate(config.hitl, reviewer=approval_reviewer) if approval_reviewer else None
    graph = build_platform_graph(config, llm=llm, approval_gate=gate)
    cfg = {"configurable": {"thread_id": thread_id}}
    with Tracer(config.tracing).trace_run("run_task", {"task": task}):
        yield from graph.stream(AgentState(task=task), config=cfg)
