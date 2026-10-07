"""Config-driven agent definitions loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class AgentDefinition:
    name: str
    description: str = ""
    model: str = "gpt-4o-mini"
    risk_tier: str = "low"
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    supervisor: Dict[str, Any]
    agents: Dict[str, AgentDefinition]
    routing: Dict[str, List[str]]
    hitl: Dict[str, Any]
    checkpointing: Dict[str, Any]
    streaming: Dict[str, Any]
    tracing: Dict[str, Any]

    def agent(self, name: str) -> AgentDefinition:
        if name not in self.agents:
            raise KeyError(f"Unknown agent: {name!r}. Known: {sorted(self.agents)}")
        return self.agents[name]

    def approval_tiers(self) -> List[str]:
        return list(self.hitl.get("approval_required_for", []))


def load_config(path: str | Path) -> AgentConfig:
    """Load and validate the platform YAML config."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    agents = {
        name: AgentDefinition(name=name, **spec)
        for name, spec in raw.get("agents", {}).items()
    }
    valid_tiers = {"low", "medium", "high"}
    for agent in agents.values():
        if agent.risk_tier not in valid_tiers:
            raise ValueError(
                f"Agent {agent.name!r} has invalid risk_tier {agent.risk_tier!r}; "
                f"must be one of {sorted(valid_tiers)}"
            )

    return AgentConfig(
        supervisor=raw.get("supervisor", {}),
        agents=agents,
        routing=raw.get("routing", {}),
        hitl=raw.get("hitl", {}),
        checkpointing=raw.get("checkpointing", {}),
        streaming=raw.get("streaming", {}),
        tracing=raw.get("tracing", {}),
    )
