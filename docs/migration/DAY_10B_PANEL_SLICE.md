# Day 10b (slice) — panel queries route through the planner pipeline

> **Goal:** Stage 5 (final stage) of the vertical slice. Make a
> `/planner` panel query run end-to-end through the planner-first
> pipeline: the planner emits a `panel_agent`-terminal plan, the
> executor streams the multi-round debate **live**, and the pipeline
> surfaces it without a redundant synthesizer pass.
>
> Builds on Day 4c (per-agent factory files) and Day 4b (the
> `PanelScopedAgent`). Depends on the recovery commit
> `migration: day 4c` being green first.

## Design decisions (confirmed with product owner)

| Question | Decision |
|---|---|
| Panel ↔ synthesizer | **Panel is terminal.** `panel_agent` writes its own debate transcript + closing brief, so a panel plan ends with `panel_agent` and needs no `synthesizer` step. Avoids double-summary / flattening the debate. |
| Live streaming | **Stream the debate live**, since it's the panel's signature UX — done cleanly via a `run_streaming()` async-generator the executor forwards, with `run()` preserved for buffered/test callers. Never at the cost of architecture/quality. |

## Files touched

| File | New / Modified | Why |
|---|---|---|
| `src/core/agents/panel_agent.py` | **MODIFIED** | Add `PanelScopedAgent.run_streaming()` (async generator: live debate + closing-brief events, then a `{"type":"_step_result","result":StepResult}` sentinel). `run()` now drains `run_streaming()` so the two paths can't diverge. Removed the now-redundant `_run_debate` buffering helper. |
| `src/core/executor.py` | **MODIFIED** | `_run_one_step` forwards events from any agent whose **class** exposes an async-gen `run_streaming` (MagicMock-proof via `inspect.isasyncgenfunction`), committing the sentinel's `StepResult`. Every other agent keeps the buffered `run()`. |
| `src/core/pipeline.py` | **MODIFIED** | `_find_synth_step` → `_find_report_step` returning `(step, kind)`: prefers `synthesizer`, else a terminal `panel_agent`. Phase 5 emits the synthesizer's buffered text as before, but for a panel terminal emits **nothing** (it already streamed live) — no duplication. |
| `src/core/planner.py` | **MODIFIED** | Hard-rule #5 relaxed: the last step is `synthesizer` **unless** `wants_panel_debate` is True, in which case it's `panel_agent`. Added worked **Example C** (portfolio_agent → panel_agent terminal). Fixed pre-existing invalid `estimated_complexity` enum values in Examples A/B (`moderate`/`heavy` → `medium`/`high`) that were teaching the LLM schema-invalid output. |
| `src/core/agents/registry.py` | **MODIFIED** | `panel_agent` description notes it is the TERMINAL step (no synthesizer after it); `synthesizer` role-hint notes the panel exception. |
| `tests/test_pipeline_e2e.py` | **MODIFIED** | New `PanelTerminalTests`: a real (non-MagicMock) streaming fake panel agent + a panel plan; asserts the debate streams live, Phase 5 does not re-emit, and a panel failure surfaces a clean error. |
| `tests/test_planner.py` | **MODIFIED** | Asserts Example C is present; validates every worked example against the schema; checks the panel example is registry-valid and panel-terminal. |

## The streaming protocol (how a step emits live events)

```
PanelScopedAgent.run_streaming()  ──yields──▶  header / text events  (live debate)
                                  ──yields──▶  {"type":"_step_result","result":StepResult}  (terminal sentinel)

executor._run_one_step():
    if inspect.isasyncgenfunction(type(agent).run_streaming):   # MagicMock-proof
        async for ev in agent.run_streaming():
            if ev["type"] == "_step_result": result = ev["result"]   # commit to scratchpad
            else:                            yield ev                 # forward live
    else:
        result = await agent.run()                                   # buffered (all other agents)
```

The `_step_result` sentinel never escapes the executor, so it's never
rendered. `run()` uses the same drain logic, so the buffered path
(tests, non-streaming callers) produces a byte-identical `StepResult`.

## Plan shape for a panel query

```json
{
  "goal": "Run the investor panel over the user's portfolio",
  "estimated_complexity": "high",
  "steps": [
    {"id": 1, "agent": "portfolio_agent",
     "tool_subset": ["portfolio__get_holdings", "portfolio__get_concentration_risks", ...],
     "depends_on": []},
    {"id": 2, "agent": "panel_agent", "tool_subset": [], "depends_on": [1]}
  ]
}
```

`panel_agent` is the **last** step — no synthesizer. The dispatcher's
`_derive_intent_flags` already maps the classifier's
`portfolio_analysis` intent (and `want_panel`) to
`wants_panel_debate=True`, which satisfies `panel_agent`'s policy gate.

## Test results

```
$ .venv/bin/python -m unittest discover tests
Ran 209 tests   OK
```

By module (deltas vs Day 4c's 204):

| Module | Tests |
|---|---|
| `tests/test_planner.py` | 21 (+2: worked-example schema + panel registry-validity) |
| `tests/test_pipeline_e2e.py` | 6 (+2: panel streams live / panel failure surfaces) |
| `tests/test_planner.py::…panel_example` | +1 (Example C present) |
| **Total** | **209** (+5) |

Plus a production-path check: `inspect.isasyncgenfunction(
PanelScopedAgent.run_streaming)` is True (the executor streams it live)
while `run()` is preserved.

## What this enables / closes out

* A `/planner` panel query (e.g. "what would the investor panel make of
  my portfolio?") now runs through the planner-first pipeline
  end-to-end, streaming the Buffett/Wood/Graham debate live — closing
  the last open stage of the vertical slice.
* The static `portfolio_analysis` / `stock_research` flows are
  unchanged; they remain the default path. `/planner` stays opt-in.

## Known limitations / next steps

* The closing brief is emitted as one block (the moderator-synthesis
  call is non-streaming). Token-level streaming of the brief is a
  future polish.
* Auto-routing panel intents to the planner (without the `/planner`
  prefix) is still gated behind the post-demo classifier upgrade.
* Phase 4 (joiner / re-plan) remains deferred, as in Day 6-7.

## How to roll back

```bash
# whole-tree at the end of this stage
git checkout migration/day-10b-panel-slice

# or per file from the snapshot
cp -r docs/migration/snapshots/day-10b-panel-slice/src/* src/
cp docs/migration/snapshots/day-10b-panel-slice/tests/* tests/
```
