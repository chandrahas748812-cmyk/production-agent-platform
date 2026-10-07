"""Shared LangGraph state for the agent platform."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    RESEARCH = "research"
    CODE = "code"
    REVIEW = "review"
    PLAN = "plan"
    UNKNOWN = "unknown"


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _append(left: List[Any], right: List[Any] | Any) -> List[Any]:
    if isinstance(right, list):
        return left + right
    return left + [right]


class SubTask(BaseModel):
    id: int
    agent: str
    instruction: str
    status: Literal["pending", "running", "done", "failed", "awaiting_approval"] = "pending"
    result: Optional[str] = None


class AgentState(BaseModel):
    """State threaded through the supervisor graph.

    task_type / risk_tier are plain strings (not Enums) and plan items are
    plain dicts, so the SQLite checkpointer's msgpack serializer handles
    them without custom codecs. Use TaskType(...) / RiskTier(...) /
    SubTask(**d) to convert when needed.
    """

    task: str = ""
    task_type: str = TaskType.UNKNOWN.value
    plan: List[Dict[str, Any]] = Field(default_factory=list)
    messages: Annotated[List[Dict[str, str]], _append] = Field(default_factory=list)
    current_agent: str = "supervisor"
    risk_tier: str = RiskTier.LOW.value
    # HITL bookkeeping
    approval_requested: bool = False
    approval_decision: Optional[str] = None  # approve | reject | <feedback>
    human_feedback: Optional[str] = None
    # outputs
    final_answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    trace_id: Optional[str] = None
