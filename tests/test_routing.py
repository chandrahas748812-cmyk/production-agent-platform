"""Tests for dynamic agent routing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.router import HeuristicRouter, route_task
from src.state import TaskType

ROUTING_TABLE = {
    "research": ["research", "find", "what is", "explain"],
    "code": ["code", "implement", "fix", "bug", "refactor"],
    "review": ["review", "audit", "verify"],
    "plan": ["plan", "design", "architect"],
}


def test_heuristic_research():
    router = HeuristicRouter(ROUTING_TABLE)
    assert router.route("Can you research vector databases for me?") == TaskType.RESEARCH


def test_heuristic_code():
    router = HeuristicRouter(ROUTING_TABLE)
    assert router.route("Fix the bug in the retry logic") == TaskType.CODE


def test_heuristic_review():
    router = HeuristicRouter(ROUTING_TABLE)
    assert router.route("Please review this pull request") == TaskType.REVIEW


def test_heuristic_plan():
    router = HeuristicRouter(ROUTING_TABLE)
    assert router.route("Design the data pipeline architecture") == TaskType.PLAN


def test_no_match_returns_none():
    router = HeuristicRouter(ROUTING_TABLE)
    assert router.route("Hello there") is None


def test_route_task_prefers_heuristic_over_llm():
    def explosive_classifier(task: str) -> TaskType:
        raise AssertionError("LLM classifier should not be called on heuristic hit")

    assert route_task("implement the feature", ROUTING_TABLE, explosive_classifier) == TaskType.CODE


def test_route_task_falls_back_to_llm():
    def stub_classifier(task: str) -> TaskType:
        return TaskType.REVIEW

    assert route_task("something totally unrelated xyz", ROUTING_TABLE, stub_classifier) == TaskType.REVIEW


def test_route_task_unknown_when_nothing_matches():
    assert route_task("something totally unrelated xyz", ROUTING_TABLE) == TaskType.UNKNOWN


def test_config_routing_table_loads():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "agents.yaml")
    assert isinstance(cfg.routing, dict) and cfg.routing
    routed = route_task("research the latest embedding models", cfg.routing)
    assert routed == TaskType.RESEARCH
