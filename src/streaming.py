"""Server-sent events (SSE) streaming helpers.

Framework-agnostic: `format_sse` turns graph events into SSE byte chunks.
`create_sse_app` optionally builds a FastAPI app when fastapi is installed.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, Optional


def format_sse(event: str, data: Dict[str, Any], event_id: Optional[str] = None) -> bytes:
    """Format a single SSE message."""
    lines = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data)}")
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def heartbeat(interval_seconds: int = 15) -> bytes:
    return b": heartbeat\n\n"


def stream_graph_events(
    graph_events: Iterator[Dict[str, Any]],
    heartbeat_seconds: int = 15,
) -> Iterator[bytes]:
    """Convert LangGraph stream events into an SSE byte stream."""
    last_beat = time.time()
    seq = 0
    for event in graph_events:
        seq += 1
        for node, payload in event.items():
            summary = str(payload)[:500] if not isinstance(payload, dict) else {
                k: str(v)[:200] for k, v in payload.items()
            }
            yield format_sse("node_update", {"node": node, "seq": seq, "summary": summary}, event_id=str(seq))
        if time.time() - last_beat >= heartbeat_seconds:
            yield heartbeat()
            last_beat = time.time()
    yield format_sse("done", {"seq": seq})


def create_sse_app(run_task_fn):
    """Build a FastAPI SSE app around a `run_task(task, thread_id)` callable.

    Raises ImportError if fastapi is not installed.
    """
    from fastapi import FastAPI  # type: ignore
    from fastapi.responses import StreamingResponse  # type: ignore

    app = FastAPI(title="Production Agent Platform")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/run")
    def run(task: str, thread_id: str = "default"):
        def gen():
            for chunk in stream_graph_events(run_task_fn(task, thread_id)):
                yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app
