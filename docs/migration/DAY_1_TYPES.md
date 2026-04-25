# Day 1 — `src/core/types.py` (Plan, PlanStep, Scratchpad, etc.)

> **Goal:** Establish the shared Pydantic types that every other
> module in the planner-executor pipeline depends on. Nothing in this
> day touches runtime behaviour — it's pure data shape, validation,
> and JSON-schema generation for the planner LLM's `response_format`.

## Files touched

| File | New / Modified | Lines | Why |
|---|---|---|---|
| `src/core/types.py` | **NEW** | ~615 (initial; later grown to ~743 on Day 3) | Plan / PlanStep / Scratchpad / StepResult / ExecutionState / JoinDecision / variable-ref parser |
| `tests/test_types.py` | **NEW** | ~415 | 40 tests covering field validation, DAG checks, scoped context, JSON schema |

## What was added

### `src/core/types.py`

Eight Pydantic v2 models + helpers that define the shape of a plan
from creation through execution to joining:

* **`PlanStep`** — one node in the DAG. Fields: `id`, `description`,
  `agent` (the registry name), `tool_subset` (strict allow-list of
  MCP tool names), `depends_on` (list of prior step IDs), `inputs`
  (variable refs like `"#3.url"` resolved at runtime),
  `max_tool_calls`. `extra='forbid'` so any planner typo fails fast.
* **`Plan`** — top-level container. Fields: `goal`, `rationale`,
  `steps`, `schema_version`, `estimated_complexity`. Validators run
  Kahn's algorithm to detect cycles, ensure unique step IDs, and
  reject dependencies on non-existent steps.
* **`StepResult`** — what a step produces. `output: Any`, plus
  status / timing / error fields.
* **`Scratchpad`** — the shared blackboard. Holds every step's
  result, plus the `relevant_results_for_step()` API that returns
  ONLY a step's declared dependencies (Anthropic-style scoped
  context). Variable-ref resolver lives here too.
* **`ExecutionState`** — top-level run container with replan budget.
* **`JoinDecision`** — joiner's verdict: `finish` / `replan` /
  `abort`, with optional `additional_steps` for replans.

### Variable references

Plan steps use a LLMCompiler / ReWOO-style syntax in `inputs`:

```python
"#3"            # the entire output of step 3
"#3.url"        # the .url field of step 3's output
"#3.items[0]"   # first element of step 3's items list
```

`parse_var_ref(value)` parses; `Scratchpad.resolve_value(value)`
recursively replaces refs from the scratchpad's results. Helper
`_follow_path(base, path)` walks the dotted/indexed path.

### DAG validation (Kahn's algorithm)

`Plan` runs cycle detection at construction time. A plan with a
cycle, a self-loop, or a dependency on a non-existent step ID raises
`ValidationError` immediately — the planner LLM's output never
reaches the executor in a bad shape.

### `tests/test_types.py`

40 unittest tests organised into seven classes:

* `PlanStepValidationTests` — id ranges, description bounds,
  tool_subset rejects blanks, depends_on no duplicates, extra=forbid
* `PlanDAGValidationTests` — unique IDs, missing-dep, self-loop,
  cycle, valid topological order, ready-steps query
* `VariableRefTests` — bare / dotted / indexed parsing & resolution
* `ScratchpadScopedContextTests` — `relevant_results_for_step`
  returns only declared deps, handles missing dep gracefully
* `StepResultAndExecutionStateTests` — duration, terminal status,
  replan budget
* `JoinDecisionConsistencyTests` — replan ↔ additional_steps
  invariants
* `JSONSchemaTests` — pins the schema shape that the planner LLM
  receives in `response_format`

## Design decisions worth remembering

### Why Pydantic v2 (not dataclasses or TypedDict)?

The planner LLM is given `Plan.model_json_schema()` as its
OpenAI-compatible `response_format`. Pydantic v2 generates this
automatically with descriptions from `Field(...)`. Dataclasses can't
do this; TypedDict can with hacks. Pydantic also validates at parse
time, which we want — a malformed plan from the LLM should raise
immediately, not crash the executor mid-run.

### Why `extra='forbid'` everywhere?

Two reasons:
1. **Planner correctness check.** If the planner hallucinates a
   field (e.g. `agent: "research"` instead of `agent_name`), the
   parse fails immediately rather than silently dropping the field.
2. **Schema clarity.** `additionalProperties: false` in the JSON
   schema tells the LLM not to invent fields.

### Why Kahn's algorithm specifically?

Kahn's gives a topological order **and** detects cycles in a single
pass. The executor uses the order for parallel scheduling
(LLMCompiler-style). DFS-based cycle detection would also work but
would need a second pass to produce the topo order.

### Why is the scratchpad's "scoped context" only at the prompt level?

Per the architecture decision in `MULTI_AGENT_ARCHITECTURE.md`
Section 4: the scratchpad **stores everything** (Cognition's
Principle 1: full traces are necessary for the joiner and
synthesizer to reason about the run), but each step's prompt
**includes only its declared deps** (Anthropic-style scoping
prevents context bloat and cross-talk). This is the hybrid model.

## Test results

```
$ docker exec finai-api python -m unittest tests.test_types -v
[...]
Ran 40 tests in 0.004s
OK
```

All 40 tests passing.

## What this enables in later days

* **Day 2** can write the agent registry with confidence that
  `Plan` / `PlanStep` are stable inputs to `validate_plan()`.
* **Day 3** can build `ScopedAgent` knowing exactly what shape the
  scratchpad has and what `relevant_results_for_step` returns.
* **Day 6** (planner LLM) can use `Plan.model_json_schema()` as the
  `response_format` to get strict-schema output.
* **Future**: replay logs by deserialising stored Plans / Scratchpads
  back into Pydantic models.

## Snapshot

End-of-Day-1 file content:

* `docs/migration/snapshots/day-1/src/core/types.py`
* `docs/migration/snapshots/day-1/tests/test_types.py`

To restore Day 1's exact state:

```bash
cp docs/migration/snapshots/day-1/src/core/types.py src/core/types.py
cp docs/migration/snapshots/day-1/tests/test_types.py tests/test_types.py
```

Note: Day 3 added `UnmetDependency` to `types.py` and corresponding
tests. The Day 1 snapshot does NOT include those — it's the file as
it was at the end of Day 1, before Day 3's additions.
