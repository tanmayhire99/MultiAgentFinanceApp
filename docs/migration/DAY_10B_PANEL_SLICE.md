# Day 10b — Panel-slice planner examples + enable `/planner` prefix

> **Goal:** Stage 5 of the vertical-slice work. The planner-first pipeline
> can now handle panel debates (both portfolio-level and single-ticker)
> end-to-end. The `/planner` slash-command is enabled by default.

## Files touched

| File | New / Modified | Lines | Why |
|---|---|---|---|
| `src/core/planner.py` | **MODIFIED** | +105 | Add `_EXAMPLE_PANEL_PORTFOLIO` and `_EXAMPLE_PANEL_STOCK` few-shot plans; wire them into the system prompt template and `_build_system_prompt` |
| `docs/migration/README.md` | **MODIFIED** | +1 | Mark Stage 5 as complete |

## What was added

### `_EXAMPLE_PANEL_PORTFOLIO` (planner.py)

A 3-step plan that shows the canonical shape for a portfolio-level panel debate:

```
portfolio_agent → panel_agent → synthesizer
```

- Step 1: `portfolio_agent` — pulls holdings, sector allocation, concentration risks, diversification score
- Step 2: `panel_agent` — runs the multi-round Buffett/Wood/Graham debate (depends on step 1)
- Step 3: `synthesizer` — writes the moderator closing brief (depends on step 2)

### `_EXAMPLE_PANEL_STOCK` (planner.py)

A 4-step plan for a single-ticker panel debate:

```
us_stock_agent + research_agent → panel_agent → synthesizer
```

- Step 1: `us_stock_agent` — live fundamentals, growth, defensive metrics
- Step 2: `research_agent` — recent catalysts + company brief (parallel with step 1)
- Step 3: `panel_agent` — multi-round debate on the fundamentals + catalysts (depends on [1, 2])
- Step 4: `synthesizer` — moderator closing brief (depends on [3])

### System prompt updates

Two new example sections (`### Example C` and `### Example D`) in the
planner's system prompt, mirroring the existing Examples A/B format. Each
shows the query, the intent flags, and the JSON plan body.

### `FINAI_PLANNER_PREFIX` default changed to `1`

The docker-compose.override.yml now defaults `FINAI_PLANNER_PREFIX=1`
(instead of `0`). Stage 5 completes the factory coverage for all 8
registry agents, so the planner pipeline is safe to use for any query
type. Users can still set `FINAI_PLANNER_PREFIX=0` to disable the
`/planner` prefix recognition.

## Smoke tests (real LLM + real MCP tools)

### `/planner run a panel debate on NVDA`

```
Routing through the planner-first pipeline (/planner opt-in).
Plan ready: 4 step(s) (us_stock_agent → research_agent → panel_agent → synthesizer)
Step 1: us_stock_agent — Fetch NVDA live fundamentals... ✓ (5.8s, 3 tool calls)
Step 2: research_agent — Pull recent catalysts... ✓ (6.0s, 2 tool calls)
Step 3: panel_agent — Run the full multi-round debate... ✓ (69.7s, 2 tool calls)
Step 4: synthesizer — Write the moderator closing brief... ✓ (5.9s, 1 tool call)
Plan execution complete: 4 ok, 0 failed, 0 skipped, in 87.3s

[Closing Brief with stance evolution table, persona-specific analysis,
bottom-line verdict, disclaimer]
```

### `/planner run a panel debate on my portfolio`

```
Routing through the planner-first pipeline (/planner opt-in).
Plan ready: 3 step(s) (portfolio_agent → panel_agent → synthesizer)
Step 1: portfolio_agent — Pull holdings, sector allocation, risks... ✓ (19.8s, 4 tools)
Step 2: panel_agent — Run multi-round debate on the portfolio... ✓ (71.9s, 2 tools)
Step 3: synthesizer — Write closing brief... ✓ (7.0s, 1 tool call)
Plan execution complete: 3 ok, 0 failed, 0 skipped, in 98.7s

[Moderator Closing Brief with portfolio-specific analysis, stance
evolution, bottom-line verdict, disclaimer]
```

### Static flows unaffected

```
$ curl ... -d '{"messages": [{"role": "user", "content": "what is EBITDA?"}]}'
[normal educational flow output, no planner routing header]
```

## Test results

```
$ docker exec finai-api python -m unittest discover tests -q
Ran 204 tests in 1.541s
OK
```

All 204 tests pass. No new tests needed — the planner's worked examples
are pure data that the existing test infrastructure validates (the
`_build_system_prompt` call exercises the format strings, and the
registry's `validate_plan` would catch any schema violations in the
example data).

## What this completes

Stage 5 was the last item in the migration slice table. The planner-first
multi-agent architecture is now feature-complete for all 8 registry agents:

- **Ungated (5):** research_agent, us_stock_agent, indian_stock_agent,
  filings_agent, portfolio_agent
- **Gated (2):** claim_agent (requires `wants_claim_tracking`),
  panel_agent (requires `wants_panel_debate`)
- **Ceremonial (1):** synthesizer (final step in every plan)

The `/planner` prefix routes any query through the planner → executor →
synthesizer pipeline, and the static flows handle regular traffic
unchanged. The migration from the demo-era flow-based architecture to
the planner-first architecture is **complete**.

## Migration log entry

- **Git tag:** `migration/day-10b-panel-slice`
- **Commit:** `migration: day 10b - panel-slice planner examples + enable /planner prefix`