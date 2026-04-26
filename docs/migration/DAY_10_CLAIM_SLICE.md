# Day 10 (slice) — `/planner` opt-in dispatcher wiring

> **Goal:** Stage 3 of the vertical-slice work. Make the new
> planner-first pipeline reachable from a real user query for the
> first time, without disturbing any of the existing static flows.
> The `/planner <query>` slash-command is the only entry point —
> auto-routing per intent (FINAI_PLANNER_ENABLED env var) is held
> off until after the demo so the planner is exercised manually only.

## Files touched

| File | New / Modified | Lines | Why |
|---|---|---|---|
| `src/core/flows/planner_pipeline.py` | **NEW** | ~191 | Bridge between dispatcher and `pipeline.run_pipeline`; converts `_status` events to chat lines, optional artifact wrap |
| `src/core/flows/__init__.py` | **MODIFIED** | +4 | Export the new flow |
| `src/core/dispatcher.py` | **MODIFIED** | +51 | `_strip_planner_prefix` helper + flow-selection branch when `force_planner` is set |
| `tests/test_dispatcher_planner_routing.py` | **NEW** | ~316 | 20 tests covering prefix detection, intent-flag derivation, and 4 end-to-end routing cases |

## What was added

### `src/core/flows/planner_pipeline.py`

A flow whose `run(query, decision, user_id)` signature matches every
other flow's, so the dispatcher can drop it into `_FLOW_MAP`-shaped
slots without special-casing. Internally it:

1. Calls `_derive_intent_flags(decision)` to translate the
   classifier's coarse `intent` enum + `want_panel` boolean into the
   new 6-flag `intent_flags` vocabulary the registry's policy gates
   consume. This is a deterministic mapping; the long-term plan is
   for the upgraded classifier to emit `intent_flags` directly, at
   which point this helper becomes a one-line passthrough.
2. Loads the cached MCP tool list via `mcp_servers.get_tools()`.
3. Emits a one-line "_Routing through the planner-first pipeline
   (`/planner` opt-in)._" header so demo audiences can see they're
   on the new path.
4. Iterates `pipeline.run_pipeline(query, intent_flags=..., ...)`
   and rewrites events:
   * `_status` → italic chat line via `artifacts.status(text)`
     (always shown, per UX decision)
   * `text` from the synthesizer → either inline or wrapped in a
     LibreChat artifact block (lazy-open on first emission, depending
     on `decision["wants_artifact"]`)
   * `error` and any unrecognised types → forwarded as-is so the
     dispatcher's existing rendering kicks in.
5. Closes the artifact block if it was opened.

### `_derive_intent_flags` mapping

| Classifier intent | Flags set |
|---|---|
| `deep_stock_research` | `wants_claim_tracking`, `wants_filings`, `wants_historical_news`, `wants_deep_research` |
| `portfolio_analysis` | `wants_portfolio_data`, `wants_panel_debate` |
| `stock_research` | none (unless `want_panel=True`) |
| `topic_research`, `educational`, `meta_help`, `smalltalk` | none |
| _(any intent)_ + `want_panel=True` | + `wants_panel_debate` |

Unknown / future intents get all-false flags — the planner falls back
to the unrestricted set of agents (registry's policy gates only fire
when a flag is required).

### Dispatcher patch — `_strip_planner_prefix`

Mirrors the existing `_strip_trace_prefix` and
`_strip_artifact_prefix` helpers:

```python
_PLANNER_PREFIXES = ("/planner ", "/planner\t")
_PLANNER_BARE = "/planner"


def _strip_planner_prefix(query: str) -> Tuple[str, bool]:
    q = query.lstrip()
    lower = q.lower()
    for prefix in _PLANNER_PREFIXES:
        if lower.startswith(prefix):
            return q[len(prefix):].lstrip(), True
    if lower.rstrip() == _PLANNER_BARE:
        return "", True
    return query, False
```

* Case-insensitive (`/Planner`, `/PLANNER` work).
* Requires a trailing space or tab — `/plannerhello` is NOT the
  prefix, so a hypothetical user message starting with `/plan...`
  isn't accidentally hijacked.
* Strips it BEFORE the classifier sees the query, so the intent
  classification is unaffected by the meta-instruction.

### Dispatcher routing branch

```python
intent = decision.get("intent", "educational")
if force_planner:
    flow = planner_pipeline.run
    log.info("dispatcher: /planner prefix -> routing through planner_pipeline (intent=%s)", intent)
else:
    flow = _FLOW_MAP.get(intent) or _FLOW_MAP["educational"]
```

The disclaimer footer logic still uses the **classifier's** intent
(not "planner_pipeline"), so the Day 10 wiring doesn't change which
flows trigger the regulatory disclaimer.

## Smoke test (real LLM + real MCP tools)

```
$ curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "finai-default",
         "messages": [{"role": "user",
            "content": "/planner explain the impact of US-China trade tensions on global semiconductor supply chains in 2026"}]}'
```

Returns a 75-second, 3-step run:

```
Routing through the planner-first pipeline (/planner opt-in).
Planning a multi-agent investigation for: explain the impact of US-China trade tensions...
Plan ready: 3 step(s) (research_agent → research_agent → synthesizer)
Step 1: research_agent — Gather recent news... ✓ (27.0s, 2 tool calls)
Step 2: research_agent — Collect analyst takes and key catalysts... ✓ (24.4s, 3 tool calls)
Step 3: synthesizer — Synthesize a concise explanation... ✓ (12.8s, 1 tool call)
Plan execution complete: 3 ok, 0 failed, 0 skipped, in 64.2s

[multi-source markdown report with sourced citations and a "Bottom line:" closing]

⚠️ Disclaimer — This response is an educational, multi-agent analysis...
```

Static-flow path is unaffected:

```
$ curl ... -d '{"messages": [{"role": "user", "content": "what is EBITDA?"}]}'
[normal educational flow output, no "Routing through the planner-first pipeline" header]
```

## Test results

```
$ docker exec finai-api python -m unittest tests.test_dispatcher_planner_routing -v
[...]
Ran 20 tests in 0.009s

OK
```

Plus full migration suite:

```
$ docker exec finai-api python -m unittest discover tests
[...]
Ran 184 tests in 0.485s

OK
```

By module:

| Module | Tests |
|---|---|
| `tests/test_types.py` | 46 |
| `tests/test_registry.py` | 40 |
| `tests/test_scoped_agent.py` | 31 |
| `tests/test_factories.py` | 19 |
| `tests/test_planner.py` | 19 |
| `tests/test_executor.py` | 5 |
| `tests/test_pipeline_e2e.py` | 4 |
| `tests/test_dispatcher_planner_routing.py` | **20** (new) |
| **Total** | **184** |

## What this enables

* The planner-first pipeline is now a real, manually-reachable code
  path against the running container. Demo: `/planner <any query>`.
* Existing demos and the LibreChat user experience are
  byte-identical for messages without the prefix — zero regression
  risk.
* Future work can enable the env-var auto-route table without
  touching any flow code (it's a one-line change in the dispatcher's
  flow-selection branch).

## Known limitations (deferred to Stage 4 / 5)

The planner is happy to pick `indian_stock_agent`, `us_stock_agent`,
`portfolio_agent`, or `panel_agent` — but those agents have no
factory yet, so any plan that uses them will fail with:

```
Step N construction failed: No factory registered for agent
'indian_stock_agent'. Available factories: ['claim_agent',
'filings_agent', 'research_agent', 'synthesizer'].
```

This is the **intended** Stage 3 behaviour — failing fast keeps the
factory map honest. Stage 4 ships the four missing factories; Stage
5 wires panel debates into the planner's catalog.

For the demo today, queries that route to the four available agents
work end-to-end:

* Pure topic queries (research_agent only)
* Topic + claim-tracking queries (research_agent + filings_agent +
  claim_agent + synthesizer)

Anything that needs market data on a specific ticker or a
portfolio-shaped panel run will surface the factory-missing error
to the user; they can either re-run without `/planner` (and get the
existing static flow) or wait for Stage 4.

## Snapshot

End-of-stage file content:

* `docs/migration/snapshots/day-10-claim-slice/src/core/dispatcher.py`
* `docs/migration/snapshots/day-10-claim-slice/src/core/flows/__init__.py`
* `docs/migration/snapshots/day-10-claim-slice/src/core/flows/planner_pipeline.py`
* `docs/migration/snapshots/day-10-claim-slice/tests/test_dispatcher_planner_routing.py`

To restore Stage 3's exact state:

```bash
git checkout migration/day-10-claim-slice
# or per-file:
cp -r docs/migration/snapshots/day-10-claim-slice/src/* src/
cp docs/migration/snapshots/day-10-claim-slice/tests/test_dispatcher_planner_routing.py tests/
```
