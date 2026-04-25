# Day 0 — Baseline (pre-migration)

> **TL;DR:** The codebase that existed *before* the planner-first
> multi-agent refactor began. This is the demo-era architecture that
> Phases 11–14 produced — flow-based routing, hard-coded execution
> graphs per intent, and the persona panel. None of `src/core/types.py`,
> `src/core/agents/`, or `tests/` existed yet.

## What the system did

The Day 0 system was a working FinAI demo with:

* **4 hand-coded flows** picked by a router LLM:
  - `stock_research` — single-stock deep dive (us / india)
  - `portfolio_analysis` — full portfolio + Buffett/Wood/Graham panel
  - `topic_research` — broad market / sector queries
  - `deep_stock_research` — `deepagents`-driven SEC + claim-tracking
    (added in Phase 12)
* **Investor panel** (`src/core/panel.py`, 1392 lines) — multi-round
  Buffett / Wood / Graham debate with moderator + convergence detection.
* **34 namespaced MCP tools** across 4 worker servers (portfolio /
  us_stock / indian_stock / research). Tools loaded lazily via
  `MultiServerMCPClient`.
* **Resilience layer** — `src/core/resilient_stream.py` wraps every
  LLM stream with retry + cache-replay. `src/core/cache.py` caches
  full responses keyed by stream signature.
* **LibreChat front-end** — Docker compose stack with custom plugins
  routing user queries to the FastAPI app.

## File inventory at Day 0

```
src/
├── app.py                                 FastAPI entrypoint
├── config/
│   └── mcp_servers.py                     MultiServerMCPClient lifecycle
├── core/
│   ├── dispatcher.py                      router → flow → stream events
│   ├── router.py                          intent classifier (4 intents)
│   ├── panel.py                           Buffett/Wood/Graham panel
│   ├── debate.py                          panel transcript helpers
│   ├── streaming.py                       SSE event types
│   ├── cache.py                           response cache
│   ├── resilient_stream.py                retry + replay wrapper
│   └── flows/
│       ├── stock_research.py
│       ├── portfolio_analysis.py
│       ├── topic_research.py
│       ├── deep_stock_research.py         deepagents harness
│       └── educational.py
├── agents/
│   ├── personas/
│   │   ├── base.py                        ChatOpenAI + ReAct factory
│   │   ├── buffett.py
│   │   ├── wood.py
│   │   ├── graham.py
│   │   └── moderator.py
│   └── workers/
│       ├── portfolio_mcp.py               6 tools
│       ├── us_stock_mcp.py                6 tools
│       ├── indian_stock_mcp.py            6 tools
│       ├── research_mcp.py                16 tools (incl. SEC + Indian)
│       ├── _research.py                   research helpers (Tavily, DDG)
│       ├── _live.py                       Yahoo Finance live data
│       ├── _claims.py                     forward-claim extraction
│       ├── _indian_filings.py             BSE / Screener / NSE
│       ├── _fixtures.py                   demo fallbacks
│       └── _live.py
└── legacy/                                user-pasted reference code
    ├── planner.py
    ├── orchestrator.py
    └── router.py
```

## What was wrong with this architecture

The Day 0 system worked but had three structural problems that motivated
the refactor:

### 1. Flows are static — no runtime planning

Each flow is a hand-written async generator that decides at code-time
which tools to call, in what order, and with what error handling. New
intents require a new flow, which means writing both the routing logic
**and** the execution graph. There's no LLM-level reasoning about
"which agents should I call to answer this?" The router only picks
the bucket.

### 2. No agent specialisation — flows just pass tools to a single ReAct loop

`deep_stock_research.py` builds a single `create_deep_agent` with the
**full 34-tool surface** and a long prompt. The agent has no concept
of "I am the Research Agent" vs "I am the Filings Agent" — it just
sees all tools and picks. This is exactly the failure mode Anthropic
warns about in their multi-agent post: orchestrators with too-broad
tool access produce 90.2% worse results than a supervisor + scoped
sub-agents.

### 3. Policy gates are implicit (and broken)

The expensive operations — `claim_agent` (~1 LLM call per claim, 1 per
verdict) and the `panel` (~60-180 s per run) — were "gated" only by
which flow the router picked. But `deep_stock_research` would always
run claim-tracking, even for queries that didn't ask for it. There
was no first-class policy check at the agent / tool level.

## The refactor's goals

The migration from Day 1 onwards introduces:

* A **`Plan`-based execution model** (Day 1: `src/core/types.py`) with
  per-step `tool_subset` declarations and DAG dependencies.
* An **agent registry** (Day 2: `src/core/agents/registry.py`) that
  declares tool ownership and policy gates as Pydantic data, so a bad
  plan fails validation before it ever runs.
* A **scoped agent runtime** (Day 3: `src/core/agents/_base.py`) that
  enforces the planner's tool subset, gives each agent only its
  declared dependencies via a synthetic `get_prior_result` tool, and
  emits structured `unmet_dependencies` instead of trying to do other
  agents' jobs.

By Day N (target: ~10 days of work), the planner replaces the static
flow router for all but the most performance-critical paths.

## Where Day 0 code still lives

**The Day 0 architecture is not deleted.** All four flows still work,
the panel still runs, the MCP workers still serve 34 tools, and the
dispatcher still routes by intent. The new planner-first stack lives
**alongside** the existing system in `src/core/types.py` +
`src/core/agents/` + `tests/`.

The migration is opt-in: only when the dispatcher decides "this query
needs LLM-driven planning, not a static flow" does it instantiate the
new pipeline. The boring portfolio / panel paths keep using the
existing flows.

This means: **rolling Day 0 back is trivial** — delete `src/core/types.py`,
delete `src/core/agents/`, delete `tests/`, and the demo still runs
exactly as before. See [README.md](README.md#how-to-roll-back) for
the exact commands.

## Key Day 0 files referenced by later days

When the planner-first system needs to integrate with the existing
runtime, it touches these Day 0 modules:

* `src/config/mcp_servers.py` — `get_tools()` returns the 34-tool
  pool that ScopedAgents filter into per-step subsets.
* `src/core/router.py` — currently classifies into 4 intents; will
  grow `intent_flags: Dict[str, bool]` in a future day to feed the
  registry's policy gates.
* `src/core/dispatcher.py` — the integration point where the new
  planner pipeline will plug in next to the existing flow-router.
* `src/agents/personas/base.py` — `build_chat_model()` + the NIM
  api-key pool. ScopedAgents reuse this rather than reinventing it.
