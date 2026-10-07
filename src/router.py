"""Dynamic agent routing based on task type.

Two-stage routing:
1. Fast heuristic keyword match (deterministic, zero LLM cost).
2. LLM classifier fallback when no keyword matches (injectable for tests).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .state import TaskType


class HeuristicRouter:
    """Keyword-based router driven by the YAML routing table."""

    def __init__(self, routing_table: Dict[str, List[str]]):
        # normalize: task_type -> set of lowercase keywords
        self.table: Dict[str, set] = {
            task: {kw.lower() for kw in kws} for task, kws in routing_table.items()
        }

    def route(self, task: str) -> Optional[TaskType]:
        lowered = task.lower()
        scores: Dict[str, int] = {}
        for task_type, keywords in self.table.items():
            hits = sum(1 for kw in keywords if kw in lowered)
            if hits:
                scores[task_type] = hits
        if not scores:
            return None
        best = max(scores, key=lambda k: (scores[k], k))
        try:
            return TaskType(best)
        except ValueError:
            return None


def _default_llm_classifier(task: str) -> TaskType:
    """Fallback classifier. Replace with a real LLM call in production.

    Without an LLM key we degrade gracefully to UNKNOWN instead of guessing.
    """
    return TaskType.UNKNOWN


def route_task(
    task: str,
    routing_table: Dict[str, List[str]],
    llm_classifier: Callable[[str], TaskType] = _default_llm_classifier,
) -> TaskType:
    """Route a task to a TaskType.

    Heuristic match wins; otherwise the LLM classifier decides; UNKNOWN is
    the honest fallback when neither can classify.
    """
    router = HeuristicRouter(routing_table)
    heuristic = router.route(task)
    if heuristic is not None:
        return heuristic
    return llm_classifier(task)
