# Day 4 (slice) — ScopedAgent factories

> **Goal:** Stage 1 of the vertical-slice work. Build factory functions
> that wrap a `PlanStep` into a configured `ScopedAgent` with
> agent-appropriate model parameters. This is the smallest piece of
> Day 4 needed for the claim-tracker slice; the panel-slice factories
> land in Stage 4 (Day 4b).

## Files touched

| File | New / Modified | Lines | Why |
|---|---|---|---|
| `src/core/agents/_base.py` | **MODIFIED** | +14 | Added `system_prompt_override` parameter so synthesizer can replace the default per-step framing |
| `src/core/agents/_factories.py` | **NEW** | ~310 | Per-agent factories + dispatch function |
| `src/core/agents/__init__.py` | **MODIFIED** | +14 | Re-export factory functions |
| `tests/test_factories.py` | **NEW** | ~340 | 19 tests covering construction, model params, gates, dispatcher |

## What was added

### Four agent factories

Each factory takes a `PlanStep` and returns a configured `ScopedAgent`.
They differ in:

| Factory | Temperature | Max tokens | System prompt | Key tools |
|---|---|---|---|---|
| `build_research_agent` | 0.3 | 1500 | default (per-step) | `research__search_*`, `research__get_*` |
| `build_filings_agent` | 0.1 | 3000 | default (per-step) | `research__get_sec_*`, `research__fetch_*`, `research__*_indian_*` |
| `build_claim_agent` | 0.1 | 2000 | default (per-step) | `research__extract_forward_claims`, `research__compare_claim_to_reality` |
| `build_synthesizer` | 0.3 | **4000** | **custom override** | none (LLM-only) |

Why the choices:

* **research_agent** — slight creative latitude (T=0.3) for fluent
  news/source summarisation, modest token budget.
* **filings_agent** — analytical extraction (T=0.1), large token
  budget because 10-Ks and concall transcripts are dense.
* **claim_agent** — structured outputs (T=0.1, label + verdict +
  evidence cite). Gated by the registry's `wants_claim_tracking` flag
  — that enforcement happens automatically via `ScopedAgent.__init__`.
* **synthesizer** — the user-visible final report. **Largest token
  budget of any agent.** Custom system prompt because the default
  ScopedAgent prompt frames the LLM as "you are running ONE step of a
  larger plan; your output feeds downstream steps" — wrong for the
  synthesizer (its output IS the user-visible response, not feedstock
  for downstream steps).

### `build_scoped_agent_for_step(step, ...)` — the dispatch

Used by the executor (Stage 2). Looks up the right factory by
`step.agent` and forwards all arguments. Raises `ScopedAgentError`
if no factory is registered — so a planner that emits a step naming
e.g. `us_stock_agent` (which has no factory yet in Stage 1) fails
loudly with a list of the agents we DO have, instead of silently
running with a broken setup.

### `ScopedAgent.system_prompt_override` parameter

A small extension to the Day 3 `ScopedAgent`. New optional kwarg:

```python
ScopedAgent(
    step=...,
    scratchpad=...,
    all_mcp_tools=...,
    model=...,
    system_prompt_override="You are the FinAI Synthesizer...",  # NEW
)
```

If provided, the agent uses this string as its system prompt instead
of building one from the per-step template. The default behaviour
(`None`) is unchanged, so all 31 existing `tests/test_scoped_agent.py`
tests still pass. The override is currently used only by
`build_synthesizer`.

### Synthesizer prompt highlights

The synthesizer's custom prompt explicitly:

* Frames the LLM as the **final** agent producing the user-visible
  report (not an intermediate step).
* Tells it to call `get_prior_result(step_id)` for each declared
  dependency to fetch the data.
* Lists hard rules: `DO NOT fabricate numbers`, `DO NOT recommend
  buying or selling`, `DO NOT include a regulatory disclaimer` (the
  dispatcher adds the disclaimer for finance flows).
* Asks for a `Bottom line` at the end.

## Why factories instead of a switch in the executor

The executor (Stage 2) will iterate `plan.ready_steps()` and call
`build_scoped_agent_for_step` for each step. Keeping the
agent-specific knowledge (model params, system prompt) in the
factories means:

* The executor stays agent-agnostic — it doesn't need a giant
  `if step.agent == "synthesizer": ...` switch.
* New agents are added by adding to `_FACTORY_MAP` here, not by
  editing the executor.
* Factory tests can run in isolation with no LLM / no MCP — see
  `tests/test_factories.py`.

## Test results

```
$ docker exec finai-api python -m unittest tests.test_factories -v
[...]
Ran 19 tests in 0.316s
OK
```

Plus full migration suite:

```
$ docker exec finai-api python -m unittest tests.test_types tests.test_registry tests.test_scoped_agent tests.test_factories
[...]
Ran 136 tests in 0.438s
OK
```

By module:

| Module | Tests |
|---|---|
| `tests/test_types.py` | 46 |
| `tests/test_registry.py` | 40 |
| `tests/test_scoped_agent.py` | 31 |
| `tests/test_factories.py` | **19** (new) |
| **Total** | **136** |

## What this enables

The next stage (planner.py + executor.py + pipeline.py) can now:

* Iterate `plan.steps` in topological order and call
  `build_scoped_agent_for_step(step, ...)` to get a runnable agent.
* Trust that each agent has the right model parameters for its job
  without per-call configuration.
* Trust that the synthesizer step produces user-visible output
  rather than feedstock for further steps.

## What's still missing (deferred to later stages)

* **Factories for the panel slice's 4 other agents** (`us_stock_agent`,
  `indian_stock_agent`, `portfolio_agent`, `panel_agent`). These land
  in Stage 4 of the slice work (Day 4b).
* **Real LLM tests.** The current tests mock `build_chat_model` to
  return a fake. Once the planner and executor are in place, an
  integration test will exercise the full chain end-to-end with the
  real NIM model behind a feature flag. (Stage 2d.)
* **Streaming integration.** Factories set `streaming=True` on the
  underlying `ChatOpenAI`, but the streaming events don't yet flow
  through to the dispatcher's `PanelEvent` stream. The pipeline.py
  in Stage 2c handles that wiring.

## Snapshot

End-of-stage file content:

* `docs/migration/snapshots/day-4-slice-factories/src/core/agents/_base.py`
* `docs/migration/snapshots/day-4-slice-factories/src/core/agents/_factories.py`
* `docs/migration/snapshots/day-4-slice-factories/src/core/agents/__init__.py`
* `docs/migration/snapshots/day-4-slice-factories/tests/test_factories.py`

To restore Stage 1's exact state:

```bash
git checkout migration/day-4-slice-factories
# or per-file:
cp -r docs/migration/snapshots/day-4-slice-factories/src/* src/
cp docs/migration/snapshots/day-4-slice-factories/tests/test_factories.py tests/
```
