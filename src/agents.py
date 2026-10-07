"""Specialist agents with a pluggable LLM backend.

`LLMBackend` is the interface. `MockLLMBackend` is deterministic and runs
offline (used by tests and `example.py`). Wire in a real backend
(e.g. langchain ChatOpenAI) by implementing the two methods.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Protocol

from .config import AgentDefinition
from .state import SubTask


class LLMBackend(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


@dataclass
class MockLLMBackend:
    """Deterministic stand-in: keyword-driven canned responses, no network."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        lowered = user_prompt.lower()
        if "subtask" in system_prompt.lower() or "plan" in user_prompt.lower() and "json" in user_prompt.lower():
            return json.dumps(
                {
                    "subtasks": [
                        {"id": 1, "agent": "researcher", "instruction": f"Research: {user_prompt[:80]}"},
                        {"id": 2, "agent": "reviewer", "instruction": "Review the research for accuracy."},
                    ]
                }
            )
        if "research" in system_prompt.lower():
            return (
                "Finding 1: retrieved evidence supports the claim [1]. "
                "Finding 2: secondary source corroborates with caveats [2]."
            )
        if "coding" in system_prompt.lower():
            return (
                "```python\ndef solution():\n    return 'implemented'\n```\n"
                "Tests: 2 passed. No destructive operations performed."
            )
        if "reviewer" in system_prompt.lower():
            if any(w in lowered for w in ("delete", "drop table", "rm -rf", "exfiltrat")):
                return "ESCALATE: potentially destructive operation detected."
            return "APPROVE: no issues found."
        return f"[mock] processed: {user_prompt[:120]}"


def run_planner(task: str, definition: AgentDefinition, llm: LLMBackend) -> List[SubTask]:
    prompt = (
        f"Task: {task}\n"
        'Break this into subtasks. Output JSON: {"subtasks": '
        '[{"id": 1, "agent": "<researcher|coder|reviewer>", "instruction": "..."}]}'
    )
    raw = llm.complete(definition.system_prompt, prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group(0)) if match else {"subtasks": []}
    subtasks = []
    for item in data.get("subtasks", []):
        subtasks.append(
            SubTask(
                id=int(item.get("id", len(subtasks) + 1)),
                agent=str(item.get("agent", "researcher")),
                instruction=str(item.get("instruction", "")),
            )
        )
    if not subtasks:
        subtasks = [SubTask(id=1, agent="researcher", instruction=task)]
    return subtasks


def run_specialist(
    agent_name: str,
    instruction: str,
    definitions: Dict[str, AgentDefinition],
    llm: LLMBackend,
) -> str:
    definition = definitions[agent_name]
    return llm.complete(definition.system_prompt, instruction)


def run_reviewer(
    content: str, definitions: Dict[str, AgentDefinition], llm: LLMBackend
) -> tuple[str, bool]:
    """Returns (verdict_text, escalated)."""
    definition = definitions["reviewer"]
    verdict = llm.complete(definition.system_prompt, f"Review this:\n{content}")
    escalated = "ESCALATE" in verdict.upper()
    return verdict, escalated
