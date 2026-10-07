# Production Agent Platform

A **production-grade multi-agent orchestration platform** built on LangGraph. A supervisor classifies each task and routes it to specialist agents (planner, researcher, coder, reviewer), with **human-in-the-loop approval gates** for high-stakes actions, **SQLite checkpointing** for resumable workflows, **SSE streaming** for real-time UX, and **LangSmith tracing** for observability. All agent behavior is **config-driven via YAML** — add or retune agents without touching code.

## Architecture

```
                          ┌─────────────┐
                          │ Supervisor  │  classify + plan
                          └──────┬──────┘
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │Researcher│ │ Planner  │ │ Reviewer │  specialists
              │  (low)   │ │  (low)   │ │(med/low) │
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            ▼            │
                   │      ┌──────────┐       │
                   │      │  Coder   │       │
                   │      │  (high)  │       │
                   │      └────┬─────┘       │
                   │           ▼             │
                   │    ┌─────────────┐      │
                   │    │  HITL Gate  │◀─────┘ (escalations)
                   │    │ approve /   │
                   │    │   reject    │
                   │    └──────┬──────┘
                   ▼           ▼
              ┌────────────────────────┐
              │       Reviewer         │  verify before finalize
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │       Finalize         │  merge into final answer
              └────────────────────────┘

        SQLite checkpointer ◀── every node persists state
        LangSmith tracer   ◀── every run traced (when configured)
        SSE stream         ◀── node updates pushed to clients live
```

**Risk tiers** (`low` / `medium` / `high`) are declared per agent in `config/agents.yaml`. The HITL gate (`src/hitl.py`) requires explicit human approval before `high`-risk outputs are applied, and before `medium`-risk outputs when the reviewer escalates. Every decision lands in an audit log.

## Quickstart

```bash
pip install -r requirements.txt
python example.py
```

No API keys needed — the demo runs on a deterministic mock LLM backend. For real models, implement the two-method `LLMBackend` protocol in `src/agents.py` (e.g. wrapping `langchain_openai.ChatOpenAI`):

```bash
export OPENAI_API_KEY=<redacted>
export LANGSMITH_API_KEY=<redacted>   # optional, enables tracing
```

## Example workflows

**1. Research task** (routes to researcher → reviewer):

```python
from src.config import load_config
from src.graph import run_task

config = load_config("config/agents.yaml")
for event in run_task("Research pgvector vs Pinecone trade-offs", config, thread_id="t1"):
    print(event)
```

**2. Code task** (routes to planner → coder → HITL gate → reviewer):

```python
def my_reviewer(request):          # programmatic approver
    return ("approve", None) if "test" in request.action_summary else ("reject", "needs tests")

for event in run_task("Implement exponential-backoff retry", config,
                      thread_id="t2", approval_reviewer=my_reviewer):
    print(event)
```

**3. Resume an interrupted run** (SQLite checkpointing):

```python
from src.graph import build_platform_graph
graph = build_platform_graph(config)
cfg = {"configurable": {"thread_id": "t2"}}
graph.update_state(cfg, {"human_feedback": "add jitter to the backoff"})
for event in graph.stream(None, config=cfg):
    print(event)
```

**4. SSE streaming** (e.g. behind FastAPI):

```python
from src.streaming import stream_graph_events, create_sse_app

def run(task, thread_id="web"):
    graph = build_platform_graph(config)
    return graph.stream(AgentState(task=task), config={"configurable": {"thread_id": thread_id}})

app = create_sse_app(run)   # GET /run?task=...&thread_id=... → text/event-stream
```

## Project layout

```
config/agents.yaml   # agent definitions, routing keywords, HITL policy, checkpoint/tracing settings
src/
  config.py          # YAML loading + validation
  state.py           # AgentState, TaskType, RiskTier
  router.py          # heuristic + LLM-fallback task routing
  agents.py          # planner/researcher/coder/reviewer + pluggable LLMBackend
  hitl.py            # ApprovalGate, needs_approval policy, audit log
  graph.py           # supervisor StateGraph, SQLite checkpointing, run_task
  streaming.py       # SSE formatting + optional FastAPI app
  tracing.py         # LangSmith integration (no-op when unconfigured)
tests/
  test_routing.py    # routing logic
  test_hitl.py       # approval-gate policy and decisions
example.py           # end-to-end demo (mock LLM, auto-approve HITL)
```

## Tests

```bash
pytest tests/ -v
```

## Config reference

| Section         | Purpose                                                        |
|---------------|----------------------------------------------------------------|
| `supervisor`  | model + prompt for the classifying supervisor                  |
| `agents.*`    | per-agent model, `risk_tier`, system prompt, tools             |
| `routing`     | keyword lists mapping to `research` / `code` / `review` / `plan` |
| `hitl`        | `approval_required_for`, `approval_on_escalation_for`, timeout  |
| `checkpointing` | backend (`sqlite`) and DB path                              |
| `streaming`   | transport (`sse`), heartbeat interval                          |
| `tracing`     | provider (`langsmith`), project name                           |
