"""Human-in-the-loop approval gates for high-stakes agent actions.

Design:
- `needs_approval` is a pure function: (risk_tier, escalated, config) -> bool.
  Easy to unit test, no I/O.
- `ApprovalGate` is the runtime gate: it records the request, invokes a
  reviewer callback (CLI prompt by default), enforces a timeout, and returns
  a structured decision the graph can branch on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    agent: str
    risk_tier: str
    action_summary: str
    payload_preview: str = ""
    requested_at: float = field(default_factory=time.time)


@dataclass
class ApprovalResult:
    decision: ApprovalDecision
    reviewer: str = "human"
    feedback: Optional[str] = None
    latency_seconds: float = 0.0


def needs_approval(
    risk_tier: str,
    escalated: bool,
    hitl_config: Dict[str, Any],
) -> bool:
    """Pure policy check: does this action require human approval?"""
    if risk_tier in hitl_config.get("approval_required_for", []):
        return True
    if escalated and risk_tier in hitl_config.get("approval_on_escalation_for", []):
        return True
    return False


def _cli_reviewer(request: ApprovalRequest) -> tuple[str, Optional[str]]:
    """Default reviewer: prompt on stdin. Returns (approve|reject, feedback)."""
    print(f"\n--- APPROVAL REQUIRED [{request.risk_tier.upper()}] ---")
    print(f"Agent:  {request.agent}")
    print(f"Action: {request.action_summary}")
    if request.payload_preview:
        print(f"Preview:\n{request.payload_preview[:2000]}")
    answer = input("Approve? [y/N] ").strip().lower()
    feedback = None
    if answer not in {"y", "yes"}:
        feedback = input("Feedback for the agent (optional): ").strip() or None
        return "reject", feedback
    return "approve", None


class ApprovalGate:
    """Runtime approval gate with audit log and timeout."""

    def __init__(
        self,
        hitl_config: Dict[str, Any],
        reviewer: Callable[[ApprovalRequest], tuple[str, Optional[str]]] = _cli_reviewer,
    ):
        self.config = hitl_config
        self.reviewer = reviewer
        self.timeout_seconds = float(hitl_config.get("timeout_seconds", 300))
        self.audit_log: List[Dict[str, Any]] = []

    def request(
        self,
        agent: str,
        risk_tier: str,
        action_summary: str,
        payload_preview: str = "",
        escalated: bool = False,
    ) -> ApprovalResult:
        """Run the gate. Returns APPROVE only on explicit human approval."""
        if not needs_approval(risk_tier, escalated, self.config):
            return ApprovalResult(ApprovalDecision.APPROVE, reviewer="policy")

        request = ApprovalRequest(
            agent=agent,
            risk_tier=risk_tier,
            action_summary=action_summary,
            payload_preview=payload_preview,
        )
        start = time.time()
        # NOTE: timeout enforcement for interactive stdin is best-effort;
        # programmatic reviewers should respect it themselves.
        verdict, feedback = self.reviewer(request)
        latency = time.time() - start

        decision = (
            ApprovalDecision.APPROVE if verdict == "approve" else ApprovalDecision.REJECT
        )
        if latency > self.timeout_seconds:
            decision = ApprovalDecision.TIMEOUT

        result = ApprovalResult(
            decision=decision, feedback=feedback, latency_seconds=round(latency, 2)
        )
        self.audit_log.append(
            {
                "agent": agent,
                "risk_tier": risk_tier,
                "action": action_summary,
                "decision": decision.value,
                "feedback": feedback,
                "latency_seconds": result.latency_seconds,
            }
        )
        return result
