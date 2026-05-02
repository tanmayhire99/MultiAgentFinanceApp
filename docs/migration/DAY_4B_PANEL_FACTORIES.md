# Day 4b (slice) — Panel-slice ScopedAgent factories

> **Goal:** Stage 4 of the vertical-slice work. Add the four
> remaining `ScopedAgent` factories so the planner can emit plans
> naming **any** of the 8 agents in the registry without crashing the
> executor at construction time. The four agents covered here are
> the ones the **panel slice** needs:
>
> * `us_stock_agent`
> * `indian_stock_agent`
> * `portfolio_agent`
> * `panel_agent` *(special-cased — runs a multi-round persona debate
>   instead of a ReAct loop)*
>
> After this stage **every registry agent has a factory**, so the
> only remaining piece for the panel slice is the dispatcher wiring
> in Stage 5 — a one-line tweak to make `/planner` panel queries
> actually reach this layer end-to-end.

## Files touched

| File | New / Modified | Lines | Why |
|---|---|---|---|
| `src/core/agents/_panel_agent.py` | **NEW** | ~545 | `PanelScopedAgent` subclass — overrides `run()` to drive the multi-round persona debate via `src.core.debate` + closing-brief synthesis |
| `src/core/agents/_factories.py` | **MODIFIED** | +130 | 4 new factory functions; `_FACTORY_MAP` grows from 4 → 8 entries; module docstring updated to summarise model params per agent |
| `src/core/agents/__init__.py` | **MODIFIED** | +9 / -3 | Re-export new factories + `PanelScopedAgent`; refresh module docstring |
| `tests/test_factories.py` | **MODIFIED** | +316 / -28 | 15 new tests across 5 new test classes; update 2 existing tests that previously asserted Stage 4 was *missing* |

## Files NOT touched

* `src/core/agents/_base.py` — `ScopedAgent` base class is unchanged;
  Day 4 already added the `system_prompt_override` hook the
  synthesizer uses, and that hook is sufficient for the new factories
  too.
* `src/core/agents/registry.py` — the catalog already had all 8
  agents; only the factory layer was incomplete.
* `src/core/planner.py` / `src/core/executor.py` /
  `src/core/pipeline.py` — these consume the factory dispatcher
  (`build_scoped_agent_for_step`) and are agent-agnostic, so adding
  new factories doesn't change them. Verified by re-running the
  full migration suite — 204 tests OK after the change.
* `src/core/dispatcher.py` — Stage 5 is where this changes (panel
  queries route through `planner_pipeline`). Stage 4 is purely the
  agent layer.

## Architecture: why the panel agent is special

Every other agent in the registry uses the standard ReAct loop
(`langgraph.prebuilt.create_react_agent`) with a constrained tool
subset. The panel agent does **not**:

* It owns **zero MCP tools** at the planner level (`tools=()` in
  `registry.PANEL_AGENT`). Its work is delegated to three persona
  sub-agents (Buffett / Wood / Graham) orchestrated by the existing
  `src.core.panel` + `src.core.debate` machinery.
* It is **policy-gated** — the registry rejects any plan that names
  `panel_agent` unless the classifier set
  `intent_flags["wants_panel_debate"] = True`.
* Running the standard `ScopedAgent.run()` for it would compile a
  ReAct graph with no tools and rely on the LLM to *narrate* a
  hypothetical panel — exactly the failure mode we're trying to
  avoid.

To preserve the architecture (`ScopedAgent` is the unit of execution
the executor consumes) while delegating the actual debate to the
existing machinery, Day 4b introduces a thin subclass:

```python
class PanelScopedAgent(ScopedAgent):
    """ScopedAgent variant that orchestrates the multi-round panel debate.

    Inherits the full ScopedAgent constructor (registry validation,
    policy-gate enforcement, system-prompt assembly) so audits show
    a panel_agent step the same as any other. Only run() is
    overridden.
    """
```

The base constructor still runs (registry validation, gate check,
synthetic-tool wiring), so a panel step shows up identically to any
other in tests, logs, and the `Scratchpad`. The override only kicks
in at execution time.

### What `PanelScopedAgent.run()` does

1. **Build a portfolio context.** Walks `step.depends_on` and looks
   for prior step outputs that match known shapes:
   * a `portfolio_agent` step → adopt its `holdings`, `summary`,
     `allocation`, `risks`, `score` directly into a
     `PortfolioContext`.
   * a `us_stock_agent` / `indian_stock_agent` step → record its
     `fundamentals` / `growth_metrics` / `defensive_metrics` /
     `moat_signals` under the relevant ticker in the context.
   * a `research_agent` step → record `catalysts` per ticker.

   If at least one source was found, populate the context. Otherwise
   the debate runs **ungrounded** (still valid — degrades gracefully
   from "panel reasons over portfolio data" to "panel reasons over
   the user's question").

2. **Run the multi-round sequential debate** via
   `src.core.debate.run_debate_loop`. Drains the `AsyncIterator` of
   `PanelEvent` dicts, accumulating renderable `header` / `text`
   events into a transcript-markdown buffer and capturing the final
   `PanelScratchpad` from the terminal `_debate_done` sentinel.

3. **Synthesize the closing brief** with one moderator-voice LLM
   call. Reuses `_DEBATE_SYNTH_SYSTEM` and
   `_format_scratchpad_for_moderator` from
   `src.core.flows.portfolio_analysis` so output style matches the
   existing static-flow path. Falls back to an inlined system prompt
   if that import fails (keeps tests light).

4. **Return a single** `StepResult` with:

   ```python
   StepResult(
       step_id=...,
       status="complete",
       output={
           "text":  "## Investor Panel Debate\n\n...\n\n## Closing Brief\n\n...",
           "verdicts":            [...],   # final-round per-persona verdicts
           "consensus_round":     int|None,
           "stance_evolution_md": "...",
           "rounds":              int,
       },
       tools_used=["panel_debate_loop", "moderator_synthesis"],
       ...
   )
   ```

Every failure mode is captured at the boundary of each phase: a
crashed debate loop becomes a `failed` StepResult, a synthesis
failure surfaces as an apologetic placeholder while preserving the
transcript. The executor never sees a propagated exception from a
panel step.

## What was added

### `src/core/agents/_panel_agent.py` (new)

Defines `PanelScopedAgent`. The class is ~150 lines of code + ~250
lines of docstring (extensive because the rationale for diverging
from the base class deserves to be discoverable from the source).

Helper methods (all on the class) factor the orchestration:

| Method | Purpose |
|---|---|
| `_panel_query()` | Pick `step.description` if present, else `Scratchpad.query` |
| `_build_portfolio_context(PortfolioContext)` | Heuristic recovery of a `PortfolioContext` from prior step outputs |
| `_run_debate(...)` | Drain `run_debate_loop` into transcript markdown + scratchpad |
| `_write_closing_brief(...)` | One-shot moderator-voice synthesis |
| `_format_scratchpad_inline(scratchpad)` | Fallback transcript formatter |
| `_render_full_text(...)` | Combine transcript + closing brief into the user-visible markdown |
| `_extract_verdicts(scratchpad)` | Pull the latest per-persona stance/one-liner/confidence |
| `_consensus_round(scratchpad)` | Detect the round at which all three personas converged on a stance (or `None`) |

### `src/core/agents/_factories.py` (modified)

Three straightforward factories follow the same shape as
`build_filings_agent`:

```python
def build_us_stock_agent(*, step, scratchpad, all_mcp_tools, ...) -> ScopedAgent:
    model = build_chat_model(temperature=0.1, max_tokens=1500, streaming=True, ...)
    return ScopedAgent(...)
```

Per-agent model parameters:

| Factory | Temperature | Max tokens | Why |
|---|---|---|---|
| `build_us_stock_agent` | **0.1** | **1500** | Numeric extraction; LLM should repeat tool numbers verbatim, not paraphrase |
| `build_indian_stock_agent` | **0.1** | **1500** | Mirror of US stock; same constraint |
| `build_portfolio_agent` | **0.1** | **1500** | Deterministic Python summaries are the source of truth; no extrapolation wanted |
| `build_panel_agent` | **0.2** | **1100** | Matches moderator-synthesis params in `portfolio_analysis._run_panel_branch` for output-style parity |

The fourth factory (`build_panel_agent`) is intentionally a thin
shell that picks moderator-style params and instantiates the
`PanelScopedAgent` subclass. The constructor's chat model is
**ceremonial** for a `PanelScopedAgent` — its `run()` builds its
own chat model for the closing-brief LLM call (it does not run a
ReAct loop). We still pass one for parity with every other factory
so a future refactor that consolidates model selection has a
sensible starting point.

### `_FACTORY_MAP` grows to 8 entries

```python
_FACTORY_MAP: Dict[str, _FactoryFn] = {
    "research_agent":     build_research_agent,
    "filings_agent":      build_filings_agent,
    "us_stock_agent":     build_us_stock_agent,      # NEW (Day 4b)
    "indian_stock_agent": build_indian_stock_agent,  # NEW (Day 4b)
    "portfolio_agent":    build_portfolio_agent,     # NEW (Day 4b)
    "claim_agent":        build_claim_agent,
    "synthesizer":        build_synthesizer,
    "panel_agent":        build_panel_agent,         # NEW (Day 4b)
}
```

A new cross-cutting test asserts
`set(_FACTORY_MAP.keys()) == {a.name for a in REGISTRY}` so any
9th-agent registration without a matching factory surfaces
immediately in CI.

### `tests/test_factories.py` updates

| Existing test | What changed |
|---|---|
| `test_unknown_agent_raises_clean_error` | Previously asserted `us_stock_agent` was missing from the factory map. After Stage 4 every registry agent has a factory, so the test now uses an obviously-not-real `ghost_agent_404` name to exercise the same guard. The assertion list also grew: the error message now lists ALL 8 canonical agents the planner could correct to. |
| `CrossCuttingTests.test_all_stage1_agents_in_factory_map` | Renamed to `test_factory_map_covers_every_registered_agent` and asserts the dispatcher map is **exactly** `{a.name for a in REGISTRY}` (the canonical 8). Plus a sibling test pinning the literal set so a stray addition surfaces explicitly. |

| New test class | Tests | Coverage |
|---|---|---|
| `BuildUsStockAgentTests` | 3 | Construction; T=0.1, max=1500; works with `intent_flags=None` |
| `BuildIndianStockAgentTests` | 2 | Construction; same model params as US |
| `BuildPortfolioAgentTests` | 2 | Construction; deterministic-Python-style model params |
| `BuildPanelAgentTests` | 4 | Gate enforcement (no flag → fails); returns `PanelScopedAgent`; no MCP tools; moderator-style params |
| `PanelScopedAgentRunTests` | 3 | `run()` delegation with mocked debate loop + chat model; failure-mode coverage (crashed loop → `failed` StepResult; missing `_debate_done` → apologetic placeholder) |

The `PanelScopedAgentRunTests` class uses
`unittest.IsolatedAsyncioTestCase` because the override is
`async def run`. Two tiny ad-hoc duck-typed classes
(`_FakeScratchpadEntry`, `_FakePanelScratchpad`) stand in for
`src.core.debate.PanelScratchpad` — only the attributes the panel
agent reads (`entries`, `stance_evolution_md`, etc.).

## Test results

```
$ docker exec finai-api python -m unittest tests.test_factories -v
[... 34 dots ...]
Ran 34 tests in 0.647s

OK
```

Plus full migration suite:

```
$ docker exec finai-api python -m unittest discover tests
[... ]
Ran 204 tests in 0.652s

OK
```

By module:

| Module | Tests | Δ vs. Day 10 |
|---|---|---|
| `tests/test_types.py` | 46 | — |
| `tests/test_registry.py` | 40 | — |
| `tests/test_scoped_agent.py` | 31 | — |
| `tests/test_factories.py` | **34** | **+15** (Day 4b) |
| `tests/test_planner.py` | 19 | — |
| `tests/test_executor.py` | 5 | — |
| `tests/test_pipeline_e2e.py` | 4 | — |
| `tests/test_dispatcher_planner_routing.py` | 25 | — (was 20 at Day 10; +5 since landing) |
| **Total** | **204** | **+15** |

## What this enables

* The planner can now emit plans naming **any** of the 8 registry
  agents and the executor will construct each one cleanly. The
  Day-10 limitation (panel / portfolio queries failed at the factory
  layer) is gone.
* The Stage 5 work is now purely **dispatcher wiring** — the panel
  query path is one routing branch away from being end-to-end
  reachable. There's no remaining factory work.
* `PanelScopedAgent` is general enough to handle any
  panel-debate plan shape (portfolio-grounded, ticker-grounded,
  ungrounded). Stage 5 doesn't need to add planner few-shots
  specifically about the panel — the catalog description suffices —
  though we may want to add one for shape clarity.

## Smoke test (in-process; requires no API restart)

The `/planner` route exercises the factories at runtime — but the
running uvicorn process holds cached imports. We verify the new
factories construct correctly via an in-process Python import:

```python
$ docker exec finai-api python -c "
from src.core.agents._factories import build_scoped_agent_for_step
from src.core.agents._panel_agent import PanelScopedAgent
from src.core.types import KNOWN_INTENT_FLAGS, PlanStep, Scratchpad

step = PlanStep(
    id=4, description='Run the investor panel.', agent='panel_agent',
    tool_subset=[], depends_on=[1, 2],
)
flags = {f: False for f in KNOWN_INTENT_FLAGS}
flags['wants_panel_debate'] = True
flags['wants_portfolio_data'] = True

agent = build_scoped_agent_for_step(
    step=step, scratchpad=Scratchpad(query='Run the panel'),
    all_mcp_tools=[], intent_flags=flags,
)
assert isinstance(agent, PanelScopedAgent)
assert agent.mcp_tools == []
assert len(agent.synthetic_tools) == 2
print('panel_agent construction OK')
"
panel_agent construction OK
```

Plus a sweep across **every** registered agent confirms the
factory map is wired correctly:

```
✓ research_agent            (tools picked: 2)
✓ us_stock_agent            (tools picked: 2)
✓ indian_stock_agent        (tools picked: 2)
✓ filings_agent             (tools picked: 2)
✓ portfolio_agent           (tools picked: 2)
✓ synthesizer               (tools picked: 0)
✓ claim_agent               (tools picked: 2)
✓ panel_agent               (tools picked: 0)
Total factories registered: 8
```

A live end-to-end smoke test against the running API
(`/planner ... investor panel ...`) is deferred to **Stage 5**
because the running uvicorn process has cached imports of the
pre-Day-4b agent layer. Stage 5 covers the dispatcher tweak that
makes the panel route exercise this layer end-to-end.

## What's still missing (deferred to Stage 5)

* **Live end-to-end demo of the panel via `/planner`.** Today the
  factories work, the executor consumes them happily in unit tests,
  and the pipeline already understands a `panel_agent` step's
  output (it goes into the synthesizer's `get_prior_result`). What's
  missing is a planner few-shot example showing the canonical shape
  of a panel plan, plus a real-LLM smoke test demonstrating that the
  classifier sets `wants_panel_debate=True` for the user's phrasing.
* **Refining `PortfolioContext` recovery from prior step outputs.**
  The Stage-4 heuristics (key-shape matching) are deliberately
  conservative. Stage 5 + later iterations can add more agents'
  shapes (filings, claim verdicts) so the panel personas reason over
  every available source the planner gathered.
* **Streaming the persona round-by-round through the executor.** The
  current `PanelScopedAgent.run()` accumulates the transcript
  internally and surfaces it as one `StepResult`. The user sees the
  panel as a wall of markdown after ~60-180s. To restore the
  round-by-round streaming the static flow has, the executor would
  need to forward the `panel_agent` step's intermediate events into
  its `PanelEvent` stream — a Day-7 / Day-11 enhancement.

## Snapshot

End-of-stage file content:

* `docs/migration/snapshots/day-4b-panel-factories/src/core/agents/_factories.py`
* `docs/migration/snapshots/day-4b-panel-factories/src/core/agents/_panel_agent.py`
* `docs/migration/snapshots/day-4b-panel-factories/src/core/agents/__init__.py`
* `docs/migration/snapshots/day-4b-panel-factories/tests/test_factories.py`

To restore Stage 4's exact state:

```bash
git checkout migration/day-4b-panel-factories
# or per-file:
cp -r docs/migration/snapshots/day-4b-panel-factories/src/* src/
cp docs/migration/snapshots/day-4b-panel-factories/tests/test_factories.py tests/
```
