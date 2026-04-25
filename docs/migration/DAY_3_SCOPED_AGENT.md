# Day 3 — `src/core/agents/_base.py` (`ScopedAgent`)

> **Goal:** Build the per-step runtime wrapper. A `ScopedAgent` takes
> exactly one `PlanStep` and produces exactly one `StepResult`, with
> a constrained tool surface and clean coordination primitives.

## Files touched

| File | New / Modified | Lines | Why |
|---|---|---|---|
| `src/core/types.py` | **MODIFIED** | +120 | Added `UnmetDependency` model + `Scratchpad.unmet_dependencies` field + `add_unmet_dependency()` helper |
| `src/core/agents/_base.py` | **NEW** | ~440 | `ScopedAgent` class, synthetic tools, system prompt assembly |
| `src/core/agents/__init__.py` | **MODIFIED** | +12 | Export new symbols |
| `tests/test_types.py` | **MODIFIED** | +90 | `UnmetDependencyTests` (6 tests) |
| `tests/test_scoped_agent.py` | **NEW** | ~520 | 31 tests: construction, gates, tool filtering, synthetic tools, prompt, run |

## What was added

### `ScopedAgent` — the unit of execution

The `ScopedAgent` is what the executor instantiates per step. Public
API:

```python
sa = ScopedAgent(
    step=plan_step,                # PlanStep
    scratchpad=run_scratchpad,     # Scratchpad (shared)
    all_mcp_tools=mcp_pool,        # 34-tool list from get_tools()
    model=chat_model,              # langchain BaseChatModel
    intent_flags={...},            # classifier output for gate re-check
    recursion_limit=25,            # ReAct loop cap
)
result: StepResult = await sa.run()
```

Construction-time work:

1. **Re-validate against the registry.** The executor *should* have
   validated already, but `ScopedAgent.__init__` calls
   `registry.validate_step(step, intent_flags)` defensively. Failure
   raises `ScopedAgentError` — this prevents anyone hand-constructing
   a step that bypasses the policy gates.
2. **Filter MCP tools to the strict allow-list.** Only tool names in
   `step.tool_subset` make it into the agent's tool list. A typo
   raises immediately rather than wasting LLM turns.
3. **Build synthetic tools** (see below).
4. **Build system prompt** — step description + scope rules + the
   full agent catalog for situational awareness.
5. **Compile the ReAct graph** via `langgraph.prebuilt.create_react_agent`.

Run-time work:

1. Send the step description to the model as a `HumanMessage`.
2. Let the ReAct loop run (up to `recursion_limit` turns).
3. Catch any exception and turn it into a `StepResult(status="failed")`
   — the executor expects a result for every step, never a bubble.
4. On success, parse the trajectory: extract the final AIMessage's
   content, collect distinct tool names called, return
   `StepResult(status="complete", output={"text": ...}, ...)`.

### The two synthetic tools

Every `ScopedAgent` exposes these in addition to its filtered MCP
tool list. They're closures over the step + scratchpad, rebuilt per
agent instance.

#### `get_prior_result(step_id: int) -> str`

Reads the result of a prior step from the shared scratchpad.
**Strict scope check**: refuses any step ID not in
`step.depends_on`. Out-of-scope reads return a JSON error with the
suggestion to call `request_assistance` instead.

Returns:

```json
{
  "step_id": 5,
  "status": "complete",
  "output": {...},
  "tools_used": ["research__search_news"]
}
```

Or one of these errors:

* `out_of_scope` — step_id is not in `depends_on`
* `not_yet_executed` — no result in scratchpad
* `step_not_complete` — dep failed or was skipped

#### `request_assistance(target_agent: str, reason: str) -> str`

Records an `UnmetDependency` on the scratchpad. **Does NOT call
another agent.** The orchestrator (joiner phase, future Day 8) reads
the unmet_dependencies list and decides whether to add follow-up
steps in a replan.

This is the "awareness without agency" pattern from the Day 3 design
discussion. Each agent knows the full agent directory (in its system
prompt), can flag a need, but cannot recruit. The planner stays in
charge of plan shape.

### `UnmetDependency` model

Added to `src/core/types.py`:

```python
class UnmetDependency(BaseModel):
    requested_by_step_id: int      # gt=0
    target_agent: str               # min_length=1
    reason: str                     # 10..500 chars
    raised_at: float                # auto-set timestamp
    model_config = ConfigDict(extra="forbid")
```

Length bounds on `reason` prevent buggy tools from writing garbage
the joiner then has to deal with.

### `Scratchpad.unmet_dependencies` + write helper

```python
class Scratchpad(BaseModel):
    # ...existing...
    unmet_dependencies: List[UnmetDependency] = Field(default_factory=list)

    def add_unmet_dependency(
        self,
        *,
        requested_by_step_id: int,
        target_agent: str,
        reason: str,
    ) -> UnmetDependency:
        ...
```

The helper constructs the model so call sites can't forget validation
or mutate the auto-set `raised_at`.

### System prompt structure

```text
You are the **{title}** ({name}) running step {id} of a multi-agent investigation.

### Your task
{step.description}

### Your scope
You are running ONE step of a larger plan. Stay strictly in your lane:
- You can ONLY call the MCP tools in your tool_subset: [...]
- You can read prior step outputs via `get_prior_result(step_id)`,
  but ONLY for step IDs in your depends_on list: [...]
- If you need help from another agent, call `request_assistance(...)`.
  **Do NOT try to do another agent's job yourself.**

### Agent directory (situational awareness only)
[full registry catalog text — all 8 agents listed]

### Output
[guidance on producing structured, dep-friendly output]
```

The "situational awareness" framing is deliberate: the agent sees
what other agents do (so it knows *when* to call `request_assistance`),
but not enough mechanism to actually call them.

### `tests/test_scoped_agent.py`

31 unittest tests across six classes:

* `ConstructionTests` — valid construction, unknown agent rejected,
  tool-not-owned rejected, gated agent blocked without flag, gated
  agent allowed with flag (5)
* `ToolFilteringTests` — empty subset, order preservation, missing
  tool raises (3)
* `GetPriorResultTests` — declared dep returns full output, undeclared
  step blocked, missing step result, failed dep, no-deps tool
  description (5)
* `RequestAssistanceTests` — writes to scratchpad, reason length
  validated, multiple calls each recorded (3)
* `SystemPromptTests` — contains step description, agent
  name+title, tool_subset, depends_on list, empty deps render
  cleanly, full catalog, "stay in your lane" warning, mentions both
  synthetic tools (8)
* `TrajectoryHelperTests` — `_extract_final_text` for str / list
  content, skips empty, no-AI-message case;
  `_collect_tools_used` distinct + ordered (5)
* `RunIntegrationTests` — end-to-end with a fake bind-tools-friendly
  model, captures crash as failed (2)

## Design decisions worth remembering

### Strict allow-list vs owned-tools default

`ScopedAgent` filters MCP tools by exact match against
`step.tool_subset`. If the planner forgot to declare a tool, the
agent doesn't get it — even if the agent's registry entry owns it.
This was the user's choice from the Day 3 design Q&A. Rationale:

* **Forces precise plans.** Vague plans fail at construction, not at
  the LLM's first wrong tool call.
* **Tightens the audit trail.** A step's `tool_subset` is a binding
  contract for what the agent could possibly have done.
* **Lets the planner do "narrow" sub-steps.** The same agent can be
  called twice with different tool subsets in different steps.

Cost: planner prompts have to be more careful. We accept this.

### Why a tool-injected getter for prior results?

User chose this over inline pre-rendering during the Day 3 design
Q&A. Pros:

* **Token-efficient.** If a dep produced 50 KB of structured data, the
  agent only loads it when it actually needs to look at it.
* **Composes with multi-step reasoning.** The agent can re-fetch on
  later turns if it needs to look at a different field.
* **Natural ReAct shape.** It's just another tool call.

Cost: one extra ReAct turn per dep used. Acceptable for the audit
clarity gain.

### Why not let `ScopedAgent` call other agents directly?

This was the central architectural decision. The user proposed:

> "each agent has its own list of tools, but it also knows what list
> of tools other agents also have, so, it can just say, that okay i
> need this info from that agent"

Three levels are theoretically possible:
1. **Awareness only** — the agent sees the catalog, can flag gaps,
   cannot call (CHOSEN)
2. Inline delegation — `delegate_to(agent, query)` spawns a sub-agent
3. Peer message bus — agents talk freely (rejected at architecture
   stage)

We picked level 1 because:

* It composes cleanly with the LLMCompiler-style DAG executor we
  already chose. The executor handles parallel + dependencies; the
  agents stay simple.
* Cognition's "Don't Build Multi-Agents" rebuttal explicitly warns
  against levels 2+: agents talking to agents lose context, produce
  diverging actions, and become impossible to debug.
* Anthropic's research multi-agent post — the system that produced
  90.2% better results than a single orchestrator — uses level 1:
  the lead agent delegates via _the planner_, sub-agents do not
  talk to each other.

### Why doesn't `request_assistance` actually trigger anything?

The orchestrator owns plan-shape decisions. If
`request_assistance` could trigger a real recruit, agents could
escalate work indefinitely and we'd lose the "single planner is in
charge" invariant. Instead, the unmet_dependency list is a signal
the joiner reads later — the joiner decides whether to replan.

This means: an agent that calls `request_assistance` should still
**finish its step with whatever data it has**. The system prompt
explicitly tells it this.

## Important moments during Day 3

### The fake-model bind_tools issue

`langchain_core.language_models.fake_chat_models.FakeMessagesListChatModel`
raises `NotImplementedError` from `bind_tools`. But
`langgraph.prebuilt.create_react_agent` always calls `bind_tools` at
compile time. So the integration tests crashed at agent construction.

Fix: a tiny `_BindableFakeModel` subclass in the test file that
overrides `bind_tools(...)` to return `self`. Since the responses are
pre-baked AIMessages, real tool binding is irrelevant — we just need
the call to succeed.

This let the 31 tests pass without firing a real LLM.

### Deprecation warning

`create_react_agent` emits:

```
LangGraphDeprecatedSinceV10: create_react_agent has been moved to
`langchain.agents`. Please update your import to `from langchain.agents
import create_agent`. Deprecated in LangGraph V1.0 to be removed in V2.0.
```

Not blocking, but we should migrate when the rest of the codebase
does (the persona system at `src/agents/personas/base.py` also uses
the old import). Add to a future "Day N: framework upgrade" task.

## Test results

```
$ docker exec finai-api python -m unittest tests.test_types tests.test_registry tests.test_scoped_agent
[...]
Ran 117 tests in 0.107s
OK
```

By module:

| Module | Tests |
|---|---|
| `tests/test_types.py` | 46 (40 base + 6 UnmetDependency) |
| `tests/test_registry.py` | 40 |
| `tests/test_scoped_agent.py` | 31 |
| **Total** | **117** |

## What this enables in later days

* **Day 4** — concrete scoped agents: `src/core/agents/{research,
  us_stock, indian_stock, filings, portfolio, synthesizer}.py`. These
  are thin factory functions that take a `PlanStep` and return a
  configured `ScopedAgent` (right model temp, max_tokens, etc.).
* **Day 5** — the gated `claim_agent` factory.
* **Day 6** — the planner LLM. Can use `REGISTRY.planner_catalog_text()`
  in its system prompt and `Plan.model_json_schema()` as its
  `response_format`.
* **Day 7** — the DAG executor. Iterates `Plan.ready_steps()`,
  instantiates a `ScopedAgent` per step, runs them with
  `asyncio.gather`, commits to scratchpad.
* **Day 8** — the joiner. Reads `scratchpad.unmet_dependencies` and
  decides finish / replan / abort.

## Snapshot

End-of-Day-3 file content (the current state):

* `docs/migration/snapshots/day-3/src/core/types.py` (with
  `UnmetDependency`)
* `docs/migration/snapshots/day-3/src/core/agents/__init__.py`
* `docs/migration/snapshots/day-3/src/core/agents/registry.py`
* `docs/migration/snapshots/day-3/src/core/agents/_base.py`
* `docs/migration/snapshots/day-3/tests/test_types.py`
* `docs/migration/snapshots/day-3/tests/test_registry.py`
* `docs/migration/snapshots/day-3/tests/test_scoped_agent.py`

To restore Day 3's exact state:

```bash
# Day 3 IS the current state, so this is a no-op now.
# After Day 4+ work, to roll back to Day 3:
cp docs/migration/snapshots/day-3/src/core/types.py src/core/types.py
cp docs/migration/snapshots/day-3/src/core/agents/_base.py src/core/agents/_base.py
cp docs/migration/snapshots/day-3/src/core/agents/__init__.py src/core/agents/__init__.py
cp docs/migration/snapshots/day-3/src/core/agents/registry.py src/core/agents/registry.py
cp docs/migration/snapshots/day-3/tests/test_types.py tests/test_types.py
cp docs/migration/snapshots/day-3/tests/test_registry.py tests/test_registry.py
cp docs/migration/snapshots/day-3/tests/test_scoped_agent.py tests/test_scoped_agent.py
# Plus remove anything later days created:
# rm src/core/agents/research_agent.py  # (Day 4 will create)
# etc.
```

The git tag `migration/day-3` (created when this folder was set up)
also points at this state for whole-tree rollback.
