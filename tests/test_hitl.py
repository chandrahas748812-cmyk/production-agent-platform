"""Tests for human-in-the-loop approval gates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hitl import ApprovalDecision, ApprovalGate, needs_approval

HITL = {
    "approval_required_for": ["high"],
    "approval_on_escalation_for": ["medium"],
    "timeout_seconds": 300,
}


def test_high_risk_always_needs_approval():
    assert needs_approval("high", escalated=False, hitl_config=HITL) is True


def test_low_risk_needs_no_approval():
    assert needs_approval("low", escalated=False, hitl_config=HITL) is False


def test_medium_needs_approval_only_on_escalation():
    assert needs_approval("medium", escalated=True, hitl_config=HITL) is True
    assert needs_approval("medium", escalated=False, hitl_config=HITL) is False


def _auto_approve(request):
    return "approve", None


def _auto_reject(request):
    return "reject", "not safe"


def test_gate_approves_via_policy_for_low_risk():
    gate = ApprovalGate(HITL, reviewer=_auto_reject)  # reviewer must NOT be called
    result = gate.request(agent="researcher", risk_tier="low", action_summary="read docs")
    assert result.decision == ApprovalDecision.APPROVE
    assert result.reviewer == "policy"
    assert gate.audit_log == []  # policy approvals are not audited as human decisions


def test_gate_approve_high_risk_with_human():
    gate = ApprovalGate(HITL, reviewer=_auto_approve)
    result = gate.request(agent="coder", risk_tier="high", action_summary="write file")
    assert result.decision == ApprovalDecision.APPROVE
    assert len(gate.audit_log) == 1
    assert gate.audit_log[0]["decision"] == "approve"


def test_gate_reject_high_risk_with_feedback():
    gate = ApprovalGate(HITL, reviewer=_auto_reject)
    result = gate.request(agent="coder", risk_tier="high", action_summary="write file")
    assert result.decision == ApprovalDecision.REJECT
    assert result.feedback == "not safe"
    assert gate.audit_log[0]["feedback"] == "not safe"


def test_gate_timeout_marks_timeout():
    gate = ApprovalGate({**HITL, "timeout_seconds": -1}, reviewer=_auto_approve)
    result = gate.request(agent="coder", risk_tier="high", action_summary="write file")
    assert result.decision == ApprovalDecision.TIMEOUT
