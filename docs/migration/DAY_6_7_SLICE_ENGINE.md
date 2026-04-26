# Day 6-7 (slice) — Planner + Executor + Pipeline

> **Goal:** Stage 2 of the vertical-slice work. Wire the LLM-driven
> planner, the DAG executor, and the user-facing pipeline together.
> The dispatcher patch (Stage 3) and panel-slice factories (Stage 4)
> still come on top, but **after this day the system can already do
> a complete query → plan → execute → synthesize cycle for the topic-
> research and claim-tracking flows.**

## Files touched

| File | New / Modified | Lines | Why |
|---|---|---|---|
| `src/core/planner.py` | **NEW** | ~517 | Single LLM call → JSON Plan, with retry/repair on parse / schema / registry errors |
| `src/core/executor.py` | **NEW** | ~266 | Sequential v0 DAG executor; mutates the shared `Scratchpad` |
| `src/core/pipeline.py` | **NEW** | ~220 | Top-level orchestrator: query → plan → execute → emit synth output |
| `src/core/types.py` | **MODIFIED** | +2 | Relax `PlanStep.max_tool_calls` to `ge=0` (synth steps need 0) |
| `tests/test_planner.py` | **NEW** | ~392 | 19 tests: happy paths, JSON repair, schema retry, registry/policy retry, exhaustion |
| `tests/test_executor.py` | **NEW** | ~295 | 5 tests: topo order, status events, failed-step descendant skip, construction failures |
| `tests/test_pipeline_e2e.py` | **NEW** | ~311 | 4 tests: full happy path, planner failure, synth failure, missing-synth plan |

## Pipeline shape

```
                ┌──────────────────────────────────────────┐
   user query   │  src.core.pipeline.run_pipeline           │
   ──────────► │                                            │
   intent_flags │   Phase 2  → planner.plan(query, flags)   │
   (Phase 1)    │   Phase 3  → executor.execute(plan, ...)  │
                │   Phase 5  → emit synth StepResult.text   │
                └──────────────────────────────────────────┘
                                │
                                ▼  AsyncIterator[PanelEvent]
                       _status / text / error
```

Each phase is a separate module so they can be tested in isolation,
swapped out (e.g. a different planner backend), or replaced wholesale
when we add the joiner / replan loop (Phase 4 — deferred).

## What was added

### `src/core/planner.py` — Phase 2

Single async function `plan(query, *, intent_flags, registry,
history_summary, retries, timeout_seconds) -> Plan`. One LLM call
per attempt, up to `retries+1` attempts. Each attempt:

1. **Build the system prompt** — registry catalog block (per-agent
   description + role hint + tool subset + policy gates) + a short
   list of allowed agents + the active `intent_flags` + 2 few-shot
   plans.
2. **Build the user prompt** — the query, the intent flags, optional
   conversation summary.
3. **`await chat.ainvoke([SystemMessage, HumanMessage])`**.
4. **Parse**: `_try_repair_json(raw)` — strips ```json fences,
   single-quoted JSON, and prose preamble. JSONDecodeError → next
   attempt with the parse error echoed back.
5. **Validate against the Plan schema** (Pydantic). Validation errors
   → next attempt with errors echoed.
6. **Validate against the registry** — `REGISTRY.validate_plan(plan,
   intent_flags=...)`. Each step's agent must exist, its tools must
   be in the agent's allowed subset, and any policy-gated agent
   (e.g. `claim_agent`) must have its required intent flag set.
   Errors → next attempt with errors echoed.

If the loop exhausts retries, `PlannerError` is raised with the last
raw output and the validation error list attached for debugging.

The retry-with-error-echoback pattern is critical: an LLM that emitted
garbage on attempt 1 sees its own output and the parser/validator's
exact complaint on attempt 2, and almost always self-corrects.

### `src/core/executor.py` — Phase 3

Single async generator `execute(*, plan, scratchpad, intent_flags,
all_mcp_tools, registry, recursion_limit) -> AsyncIterator[PanelEvent]`.

Algorithm (the v0 is sequential):

```python
while not all_terminal:
    completed = scratchpad.completed_ids()      # only "complete"
    terminal = scratchpad.terminal_ids()        # complete ∪ failed ∪ skipped
    ready = [s for s in plan.steps
             if s.id not in terminal
             and all(dep in completed for dep in s.depends_on)]

    if not ready:
        # remaining steps are blocked by failed/skipped ancestors
        mark_remaining_skipped()
        break

    step = ready[0]                             # first ready in topo order
    yield from _run_one_step(step, ...)         # builds ScopedAgent → run() → write StepResult
```

Key invariant: **a step is ready iff every dep is `complete`**, not
just `terminal`. A failed dep means the step gets `skipped`, not
queued for execution. This way:

* The walk always terminates (every iteration either runs a step or
  marks remaining steps skipped).
* The scratchpad ends with one `StepResult` per `PlanStep`, each
  in `complete`, `failed`, or `skipped` status.
* The pipeline (Phase 5) can then look at the synth step's status
  and emit a partial result, an error, or the full report.

Per-step events are `_status` PanelEvents like:

```
Step 1: research_agent — Pull recent news on Indian IT sector...
Step 1 ✓ (3.4s, 2 tool calls)
```

Construction failures (`ScopedAgentError` from
`build_scoped_agent_for_step`) emit an additional `error` event and
write a `failed` StepResult so the descendant-skip logic kicks in.

### `src/core/pipeline.py` — Phase 5 + glue

The user-visible orchestrator. Single async generator
`run_pipeline(query, *, intent_flags, all_mcp_tools, registry,
history_summary) -> AsyncIterator[PanelEvent]`. Wires:

* **header** — `_status` event "Planning a multi-agent investigation
  for: _<query>_"
* **plan** — calls `planner.plan(...)`, catches `PlannerError`
  → emits `error` event, returns
* **plan summary** — `_status` event with the agent chain, e.g.
  `Plan ready: 3 step(s) (research_agent → claim_agent → synthesizer)`
* **execute** — yields all events from `executor.execute(plan,
  scratchpad, ...)`
* **synthesizer surface** — finds the highest-id `synthesizer` step
  in the plan; pulls its `StepResult` from the scratchpad; emits a
  `text` event with `persona="synthesizer"` carrying the report
* **footer** — `_status` event with total step count and elapsed
  seconds

Edge cases each emit a structured `error` event:

* No `synthesizer` step in the plan (planner bug)
* Synth step never ran (executor bug)
* Synth step `failed` or `skipped` — partial text surfaced if any,
  followed by an explanation of which step failed

### `src/core/types.py` change

```diff
     max_tool_calls: int = Field(
         default=20,
-        ge=1,
+        ge=0,
         le=200,
         description=(
             "Hard cap on the number of tool calls this step's agent "
             "may make. Enforced inside the ReAct loop. Most steps "
             "should be 5-15; long-running steps (e.g. paginating a "
-            "300-page Annual Report) may justify higher."
+            "300-page Annual Report) may justify higher. Use 0 for "
+            "pure-synthesis steps that only consume prior outputs."
         ),
     )
```

The synthesizer agent has `tools=()` by design — its sole job is to
write a markdown report from the scratchpad context, not to call
tools. The few-shot plans embedded in `planner.py`'s prompt already
use `max_tool_calls: 0` for synthesizer steps, so the schema needs
to accept that.

The `AgentDefinition.max_tool_calls_default` field in
`registry.py` keeps `ge=1` since it's a default for *operational*
agents.

## Why these specific failure modes

The retry loop in `planner.py` distinguishes 3 error classes
because they have different ergonomics:

| Failure | Recovery |
|---|---|
| JSON parse | Repair (strip fences / quote-fix), then echo `JSONDecodeError.msg` back to the LLM |
| Schema validation | Echo all validation errors (with locations) — Pydantic errors are extremely actionable |
| Registry validation | Echo the offending step ids + reason (unknown agent / unowned tool / policy-gated without flag) |

A 4th class (timeout) is mapped to `PlannerError` immediately
because the LLM provider's transient failure mode isn't typically
fixed by an immediate retry of the same prompt.

## Why sequential execution for v0

Three reasons:

1. **Simplicity** — the topological walk is 30 lines and obvious.
2. **The claim-tracker slice doesn't benefit much from parallelism.**
   Its 5 steps have data dependencies on prior steps anyway —
   `research_agent` (1) and `filings_agent` (2) are independent,
   then `claim_agent` (3) depends on (1), `claim_agent` (4) depends
   on (3) and (2), `synthesizer` (5) depends on (3) and (4). Only
   steps 1 and 2 are theoretically parallelisable.
3. **NIM serialises requests.** The single-tenant NIM endpoint we run
   on the demo box gives a small wall-clock improvement from
   parallelism even when the DAG allows it.

Day-7's enhancement is to swap the `step = ready[0]` line for
`asyncio.gather(*[_run_one_step(s, ...) for s in ready])`. The data
structures are already parallel-ready (the `Scratchpad` is
mutation-safe via `add()`).

## Test results

```
$ docker exec finai-api python -m unittest tests.test_planner tests.test_executor tests.test_pipeline_e2e -v
[...]
Ran 28 tests in 0.020s

OK
```

Plus full migration suite:

```
$ docker exec finai-api python -m unittest discover tests
[...]
Ran 164 tests in 0.426s

OK
```

By module:

| Module | Tests |
|---|---|
| `tests/test_types.py` | 46 |
| `tests/test_registry.py` | 40 |
| `tests/test_scoped_agent.py` | 31 |
| `tests/test_factories.py` | 19 |
| `tests/test_planner.py` | **19** (new) |
| `tests/test_executor.py` | **5** (new) |
| `tests/test_pipeline_e2e.py` | **4** (new) |
| **Total** | **164** |

## What this enables

After this day, the pipeline is a complete library — the dispatcher
just needs to import `run_pipeline` and forward its `PanelEvent`
stream like any other flow:

```python
# src/api/dispatcher.py (Stage 3)
from src.core.pipeline import run_pipeline

async def _flow_planner_pipeline(decision, ctx):
    async for ev in run_pipeline(
        decision.query,
        intent_flags=decision.intent_flags,
        all_mcp_tools=ctx.mcp_tools,
        history_summary=decision.history_summary,
    ):
        yield ev
```

The Stage 3 work is just plumbing: a `FINAI_PLANNER_ENABLED` env
var, a `/planner` slash-prefix, and the `if planner_enabled:` switch
in the dispatcher's intent-routing block.

## What's still missing (deferred to later stages)

* **Replan / joiner loop (Phase 4).** If the synth step fails, the
  pipeline emits an error and stops. A future iteration will detect
  unmet dependencies via the scratchpad's `unmet_dependencies` list
  and re-plan around them.
* **Parallel execution.** Independent ready steps still run
  sequentially. Day 7 enhancement.
* **Streaming the agent ReAct trajectory** during a step. The
  per-step PanelEvents are start/end summaries only; the actual
  tokens stream through the ScopedAgent's underlying ChatOpenAI but
  aren't forwarded into the executor's event stream. The user-
  visible synthesizer text DOES stream because the synthesizer's
  StepResult.output["text"] is emitted as a single text event the
  dispatcher/SSE handler renders incrementally.
* **Real-LLM smoke test.** Stage 2d covers mocked-LLM unit tests.
  An end-to-end test against the live NIM endpoint behind a feature
  flag will land after Stage 3 wires the dispatcher.
* **Panel-slice support.** `panel_agent`, `us_stock_agent`,
  `indian_stock_agent`, `portfolio_agent` don't have factories yet —
  Stage 4 (Day 4b).

## Snapshot

End-of-stage file content:

* `docs/migration/snapshots/day-6-7-slice-engine/src/core/planner.py`
* `docs/migration/snapshots/day-6-7-slice-engine/src/core/executor.py`
* `docs/migration/snapshots/day-6-7-slice-engine/src/core/pipeline.py`
* `docs/migration/snapshots/day-6-7-slice-engine/src/core/types.py`
* `docs/migration/snapshots/day-6-7-slice-engine/tests/test_planner.py`
* `docs/migration/snapshots/day-6-7-slice-engine/tests/test_executor.py`
* `docs/migration/snapshots/day-6-7-slice-engine/tests/test_pipeline_e2e.py`

To restore Stage 2's exact state:

```bash
git checkout migration/day-6-7-slice-engine
# or per-file:
cp -r docs/migration/snapshots/day-6-7-slice-engine/src/* src/
cp -r docs/migration/snapshots/day-6-7-slice-engine/tests/* tests/
```
