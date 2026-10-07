"""LangSmith tracing integration — no-op when not configured.

Set LANGSMITH_API_KEY (and optionally LANGSMITH_PROJECT) to enable.
The platform never fails if tracing is unavailable.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


class Tracer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = bool(os.getenv("LANGSMITH_API_KEY"))
        self.project = os.getenv(
            "LANGSMITH_PROJECT", config.get("project", "production-agent-platform")
        )
        self._client = None
        if self.enabled:
            try:
                from langsmith import Client  # type: ignore

                self._client = Client()
            except Exception:
                self.enabled = False

    @contextmanager
    def trace_run(self, name: str, inputs: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """Context manager yielding a mutable metadata dict for the run."""
        metadata: Dict[str, Any] = {"run_name": name, "project": self.project}
        if inputs:
            metadata["inputs_preview"] = str(inputs)[:500]
        try:
            yield metadata
        finally:
            if self.enabled:
                # LangSmith auto-instruments LangChain/LangGraph; this is the
                # explicit hook for non-LC spans.
                pass

    def log_event(self, run_name: str, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        # Placeholder for explicit langsmith run logging; kept minimal so the
        # platform works identically with tracing on or off.
        return
