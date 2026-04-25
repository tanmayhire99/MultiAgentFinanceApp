# Multi-Agent Architecture — Planner-First Redesign

A research-grounded proposal to fix the "deep web research also runs claim
analysis" problem by reintroducing a real planner with scoped tool subsets
per step. Companion to `FROM_DEMO_TO_PRODUCT.md`; this document is
demo-scoped (not product-scoped).

> **TL;DR.** The current architecture (intent-router → single flow →
> deepagents with all 34 tools) is the **augmented LLM pattern with a fat
> toolset**, which Anthropic's "Building Effective Agents" explicitly warns
> reliably misuses tools. The fix is the **orchestrator-workers pattern**:
> a planner LLM emits a typed DAG of steps where each step declares the
> exact subset of tools it's allowed to call. Claim-extraction tools simply
> aren't in any step's tool_subset unless the user's query explicitly
> requested claim tracking. The principle: **tools are scoped per-step,
> not global**.

---

## 1. The problem in concrete terms

Current architecture:

```
intent_router  →  ONE flow (e.g. deep_stock_research)
              →  deepagents ReAct loop with ALL 34 tools available
              →  LLM picks tools opportunistically
```

Per Anthropic (Dec 2024): *"agents continuing when they already had
sufficient results, using overly verbose search queries, or selecting
incorrect tools"* — the canonical fat-toolset failure mode.

Concrete demo evidence: a "deep research on TCS" query triggers
`extract_forward_claims` and `compare_claim_to_reality` calls because they
exist in the agent's toolbox, even though the user didn't ask for claim
tracking.

## 2. Strategic decisions made (2026-04-25)

| Decision | Choice |
|---|---|
| **Planner depth** | DAG with parallel execution (LLMCompiler-style) — Plan emits a DAG of steps with `depends_on[]`; executor runs independent steps concurrently |
| **Claim isolation** | **Both** — plan-gated AND hard policy gate. Claim agent only included when planner adds it AND query passes a phrase whitelist (`claim`, `promised vs delivered`, `verify guidance`, etc.) |
| **Context model** | Hybrid — shared scratchpad readable by all steps; each step's *prompt* only includes its declared dependencies |
| **Flow migration** | Keep existing flows as production fast-paths; planner LLM only kicks in for novel/ambiguous queries (lowest demo-breakage risk) |

## 3. State-of-the-art reference (verified upstream)

### 3.1 The six authoritative sources

| Source | Date | Key takeaway |
|---|---|---|
| **Anthropic — Building Effective Agents** ([blog](https://www.anthropic.com/engineering/building-effective-agents)) | Dec 2024 | Six composable patterns: augmented LLM, prompt chaining, routing, parallelization, **orchestrator-workers**, evaluator-optimizer. Workflows (predefined paths) vs Agents (dynamic). |
| **Anthropic — Multi-Agent Research System** ([blog](https://www.anthropic.com/engineering/built-multi-agent-research-system)) | Jun 2025 | Production case: lead orchestrator + parallel subagents = **90.2% better** than single Opus 4. Token usage explains 80% of variance. **15× more tokens than chat**. Best for breadth-first parallel research. |
| **Cognition — Don't Build Multi-Agents** ([blog](https://cognition.ai/blog/dont-build-multi-agents)) | Jun 2025 | Counter-perspective. Two principles: (1) **share context + full traces**, (2) **actions carry implicit decisions**. Recommends single-threaded linear agents + memory compression. Critiques OpenAI Swarm and AutoGen. |
| **LangChain — Plan-and-Execute Agents** ([blog](https://blog.langchain.com/planning-agents/)) | Feb 2024 | Three planning patterns: Plan-and-Execute (serial), ReWOO (variable assignment), **LLMCompiler** (DAG, parallel execution, joiner). |
| **LangChain — Multi-Agent Workflows** ([blog](https://blog.langchain.com/langgraph-multi-agent-workflows/)) | Jan 2024 | Three LangGraph patterns: collaboration (shared scratchpad), supervisor (own scratchpads + supervisor routes), hierarchical teams. |
| **Microsoft Magentic-One** ([arxiv 2411.04468](https://arxiv.org/abs/2411.04468)) | Nov 2024 | SOTA on GAIA + AssistantBench + WebArena. Lead Orchestrator with **Task Ledger + Progress Ledger**, re-plans on errors. Modular agents added/removed without prompt tuning. |

### 3.2 The convergence (where everyone agrees)

- **Plan first, execute second.** Either explicit (Anthropic orchestrator,
  Magentic-One) or via inferred DAG (LLMCompiler). Don't let the LLM dynamically
  choose tools across the entire flow.
- **Share context across the system but scope it per-step.** Cognition's
  Principle 1; the hybrid pattern this document adopts.
- **DAG > linear plan when there's any parallelism.** LLMCompiler claims
  3.6× speedup on parallelizable tasks.
- **Modular agents with declared capabilities.** Magentic-One is the
  reference here.
- **Re-planning is mandatory for production.** Both Anthropic's research
  system and Magentic-One re-plan when results expose gaps.

### 3.3 The disagreement (and how to resolve it)

Anthropic says multi-agent works (90.2% improvement). Cognition says it
fails (context fragmentation). They're both right and they're talking about
different things:

- **Anthropic's "multi-agent"** = parallel subagents on **independent**
  research topics (compress findings → return to lead). No coordination
  needed.
- **Cognition's "multi-agent"** = parallel subagents that need to produce a
  **coordinated** output (e.g. building a Flappy Bird clone). Heavy
  coordination needed.

For us, finance research is mostly the Anthropic case (parallel filings
fetch, parallel news fetch, parallel metric fetch — independent), with the
Cognition case appearing only at synthesis time (panel debate, final
report). The chosen architecture (hybrid context model + one-shot
synthesis) handles both.

## 4. The proposed architecture for FinAI v2

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 1 — CLASSIFY (~1s, ~500 tokens)             │
│  Query → light LLM → { intent, complexity, ticker_hints, depth }    │
│  Reuses existing src/core/router.py logic                           │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│                   PHASE 2 — PLAN (~3s, ~2000 tokens)                │
│  IF intent matches a known fast-path flow (e.g. educational,        │
│      portfolio_panel, single_ticker_research):                      │
│      use that flow's deterministic Plan template                    │
│  ELSE:                                                               │
│      planner LLM with full agent/tool catalog as a JSON schema      │
│      Output: typed Plan = { goal, rationale, steps[] }              │
│      Each step:                                                      │
│        { id, description, agent, tool_subset[], inputs,             │
│          depends_on[] }                                              │
│  Hard policy gate: claim_agent only added if query passes the       │
│      claim_phrase whitelist                                         │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│             PHASE 3 — EXECUTE (DAG, parallel where possible)        │
│  LLMCompiler-style DAG executor:                                    │
│    • Topologically sort steps by depends_on                         │
│    • Run independent steps concurrently (asyncio.gather)            │
│    • Each step gets a constrained agent: ReAct loop with ONLY       │
│      its declared tool_subset, never the full 34 tools              │
│    • Variable references: step3 can read step1.output via #1        │
│    • Step results saved to a shared Scratchpad                      │
│    • Each step's PROMPT only includes its declared dependencies     │
│      (Cognition principle); but it CAN query the scratchpad if      │
│      it needs extra context                                         │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│                   PHASE 4 — JOIN / RE-PLAN                          │
│  Joiner LLM looks at all step outputs + original goal               │
│  Decides:                                                            │
│    a) Sufficient → render final report                              │
│    b) Gaps remain → re-plan with new steps appended (Magentic-One)  │
│    c) Failure → graceful degrade with partial answer                │
│  Re-plan limit: max 2 re-plan rounds, then synthesise what we have  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│                   PHASE 5 — SYNTHESIZE                              │
│  Single-shot LLM call (Open Deep Research finding: one-shot         │
│  writing > parallel section writing for coherent reports)           │
│  Renders markdown to existing PanelEvent SSE stream                 │
└─────────────────────────────────────────────────────────────────────┘
```

## 5. Agent + tool catalog (what the planner sees)

```yaml
agents:
  research_agent:
    description: "Web search, company briefs, news (recent and historical)"
    tools: [search_news, search_historical_news, search_web, get_company_brief]

  us_stock_agent:
    description: "US-listed equity metrics, fundamentals, defensive ratios"
    tools: [get_quote, get_fundamentals, get_growth_metrics,
            get_defensive_metrics, get_moat_signals]

  indian_stock_agent:
    description: "NSE/BSE equity metrics; same shape as us_stock"
    tools: [get_quote, get_fundamentals, get_growth_metrics,
            get_defensive_metrics, get_moat_signals]

  filings_agent:
    description: "SEC EDGAR + Indian BSE/NSE filings; PDF + HTML extraction"
    tools: [get_sec_filings, fetch_sec_document, get_indian_filings,
            fetch_indian_document, get_screener_snapshot,
            get_indian_concall_urls, get_indian_annual_reports]

  claim_agent:
    description: "Forward-claim extraction + verdict against actuals"
    tools: [extract_forward_claims, compare_claim_to_reality]
    policy_gate:
      required_phrases:
        - "claim"
        - "claims vs reality"
        - "promised vs delivered"
        - "verify guidance"
        - "did they deliver"
        - "stood on their claims"
        - "track record of guidance"
      hard_block_unless_match: true

  portfolio_agent:
    description: "User's holdings, sector allocation, concentration"
    tools: [get_holdings, get_portfolio_summary, get_sector_allocation,
            get_concentration_risks, get_diversification_score]

  panel_agent:
    description: "Buffett/Wood/Graham debate w/ shared scratchpad"
    tools: []  # uses sub-personas internally
    policy_gate:
      required_phrases: ["panel", "debate", "experts", "buffett", "wood",
                         "graham", "personas", "investor view"]

  synthesizer:
    description: "Final markdown report from collected step outputs"
    tools: []
```

## 6. Plan examples for the queries that motivated this redesign

### "Deep research on TCS" — claim agent NOT in plan

```json
{
  "goal": "Deep research on TCS",
  "rationale": "User asked for research, not claim tracking",
  "steps": [
    {"id": 1, "agent": "indian_stock_agent",
     "tool_subset": ["get_quote", "get_fundamentals"], "depends_on": []},
    {"id": 2, "agent": "research_agent",
     "tool_subset": ["search_news", "get_company_brief"], "depends_on": []},
    {"id": 3, "agent": "filings_agent",
     "tool_subset": ["get_screener_snapshot"], "depends_on": []},
    {"id": 4, "agent": "synthesizer", "tool_subset": [],
     "depends_on": [1, 2, 3]}
  ]
}
```

Steps 1, 2, 3 run in parallel (no dependencies). Step 4 waits for all three.

### "Claim tracking on NVDA" — claim agent IS in plan

```json
{
  "goal": "Track NVDA's past claims vs actuals",
  "rationale": "User explicitly asked for claim tracking, query passed policy_gate",
  "steps": [
    {"id": 1, "agent": "filings_agent",
     "tool_subset": ["get_sec_filings", "fetch_sec_document"], "depends_on": []},
    {"id": 2, "agent": "research_agent",
     "tool_subset": ["search_historical_news"], "depends_on": []},
    {"id": 3, "agent": "claim_agent",
     "tool_subset": ["extract_forward_claims"], "depends_on": [1, 2]},
    {"id": 4, "agent": "us_stock_agent",
     "tool_subset": ["get_fundamentals", "get_quote"], "depends_on": []},
    {"id": 5, "agent": "claim_agent",
     "tool_subset": ["compare_claim_to_reality"], "depends_on": [3, 4]},
    {"id": 6, "agent": "synthesizer", "tool_subset": [], "depends_on": [5]}
  ]
}
```

DAG execution: {1, 2, 4} in parallel → 3 → 5 → 6.

## 7. What we explicitly avoid (anti-patterns from research)

### Per Cognition's warnings

- **No parallel sub-agents with private LLM contexts** unless the task is
  provably breadth-first independent (claim extraction across multiple
  filings is — claim verdict synthesis is not).
- **No shared mutable scratchpad mutated by parallel writers** — each
  step's output is appended atomically; no in-place edits.
- **No invisible re-planning** — every plan revision emits a visible event
  so the UI shows "the planner decided we needed more SEC filings; adding 2
  more steps".

### Per Anthropic's warnings

- **No shipping without an evaluation harness** — every planner-prompt tweak
  can regress queries it used to handle. Wire up FinBen + custom Indian
  benchmark before going deep.
- **Don't let agents run unbounded** — every step has a max-tool-calls
  budget, set proportional to the planner's stated complexity score.
- **Don't have agents talk to each other in real time** — they pass
  outputs through the executor, not through agent-to-agent chat.

### Per LangChain's warnings

- **Plan-and-execute beats ReAct for multi-step tasks**, but not for
  one-shot QA — for very simple queries, skip the planner and use the
  augmented LLM directly. (Hence the fast-path templates for educational
  and trivial queries.)

## 8. Migration plan from current architecture

### Current state

```
src/core/router.py  →  src/core/dispatcher.py  →  src/core/flows/{...}.py
                                                  └─ create_deep_agent(tools=ALL)
```

### Proposed state

```
src/core/router.py        ─┐
                            ├──► src/core/dispatcher.py (rewired)
src/core/planner.py  (new) ─┤
                            │      │
src/core/executor.py (new) ─┘      ▼
                                  src/core/agents/{research,stock,...}.py (new)
                                    └─ each is a thin LangGraph create_react_agent
                                       with tools=ONLY_THIS_STEP'S_SUBSET
                            ▲
                            │
                            └─ existing flows/ stay as fast-path templates
                               (deterministic Plan{...} for known intents)
```

### Phased migration (~2-3 weeks)

| Week | Day | Deliverable |
|---|---|---|
| **1** | 1 | `src/core/types.py` — port `Plan` + `PlanStep` from `legacy/planner.py`, add `tool_subset` and `depends_on` |
| | 2 | `src/core/agents/registry.py` — agent catalog with declared capabilities, tool ownership map, policy_gate config |
| | 3 | `src/core/agents/_base.py` — `ScopedAgent` wrapper that takes a tool_subset and produces a `create_react_agent` with only those tools |
| | 4 | `src/core/agents/{research,us_stock,indian_stock,filings,portfolio,synthesizer}.py` — concrete scoped agents |
| | 5 | `src/core/agents/claim.py` — claim agent with policy_gate enforcement |
| **2** | 1 | `src/core/planner.py` (fresh) — single LLM call with strict JSON schema, agent catalog as system prompt |
| | 2 | `src/core/policy_gate.py` — phrase whitelist enforcement; rejects plans containing claim_agent without matching phrases |
| | 3 | `src/core/executor.py` — DAG executor: topological sort + asyncio.gather for parallel steps + scratchpad accumulation |
| | 4 | `src/core/joiner.py` — sufficiency check + re-plan trigger |
| | 5 | Wire dispatcher to: classify → fast-path-template-or-plan → execute → join → synthesize |
| **3** | 1 | Convert each existing `flows/*.py` into a **plan template** (deterministic plan for the common case) |
| | 2 | Move `flows/deep_stock_research.py` (deepagents-based) to `legacy/` — keep for reference but no longer in `_FLOW_MAP` |
| | 3 | Add structured tool-call streaming so the UI shows the plan + per-step progress + parallel execution |
| | 4 | End-to-end smoke test: TCS deep research (claim tools should NOT appear), NVDA claim tracking (claim tools SHOULD appear), portfolio panel |
| | 5 | FinBen baseline run on the new architecture; compare to pre-migration scorecard |

## 9. Tools that stay, agents that change

The 34 MCP tools are unchanged. What changes is **which agent owns which
subset, and how the planner picks subsets per query**:

| MCP tool | Owning agent | Included by default for "deep research"? |
|---|---|---|
| `search_news` | research_agent | ✅ |
| `search_historical_news` | research_agent | ⚠ only if claim tracking |
| `search_web` | research_agent | ✅ |
| `get_company_brief` | research_agent | ✅ |
| `get_key_catalysts` | research_agent | ✅ |
| `get_analyst_takes` | research_agent | ✅ |
| `get_quote`, `get_fundamentals` | us_stock or indian_stock | ✅ |
| `get_growth_metrics`, `get_defensive_metrics`, `get_moat_signals` | us/indian_stock | ✅ |
| `get_sec_filings`, `fetch_sec_document` | filings_agent | ⚠ only if filings explicitly relevant |
| `get_indian_filings`, `fetch_indian_document` | filings_agent | ⚠ same |
| `get_screener_snapshot` | filings_agent | ✅ for Indian tickers |
| `get_indian_concall_urls`, `get_indian_annual_reports` | filings_agent | ⚠ only if claim tracking |
| **`extract_forward_claims`** | **claim_agent** | **❌ never unless explicit** |
| **`compare_claim_to_reality`** | **claim_agent** | **❌ never unless explicit** |
| `get_holdings`, `get_portfolio_summary`, etc. | portfolio_agent | only for portfolio queries |

## 10. Success criteria for the migration

The migration is "done" when all of these are true:

1. **Behaviour test**: "Deep research on TCS" produces a Final Report with
   zero `extract_forward_claims` or `compare_claim_to_reality` calls in the
   tool-call trace. (Currently fails; this is the headline fix.)
2. **Behaviour test**: "Claim tracking on NVDA" produces a Final Report
   with at least one `extract_forward_claims` and one
   `compare_claim_to_reality` call.
3. **Performance**: Average end-to-end latency unchanged or better
   (parallel execution should compensate for planner overhead).
4. **Determinism**: The same query produces the same plan structure
   ≥90% of the time. (Plan-template fast paths help here.)
5. **Graceful degradation**: A failing tool inside a step does not abort
   the whole plan — the joiner sees the gap and either re-plans or
   synthesises with the partial result.
6. **Visibility**: The UI streams the plan upfront ("Here's what I'm going
   to do: 4 steps in parallel, then synthesise") and shows per-step
   progress.
7. **Eval**: FinBen + custom Indian benchmark scores do not regress vs
   pre-migration baseline.

## 11. Citations (verified upstream 2026-04-25)

1. Anthropic. *Building Effective Agents*. Dec 2024.
   <https://www.anthropic.com/engineering/building-effective-agents>
2. Anthropic. *How we built our multi-agent research system*. Jun 2025.
   <https://www.anthropic.com/engineering/built-multi-agent-research-system>
3. Cognition. *Don't Build Multi-Agents*. Walden Yan, Jun 2025.
   <https://cognition.ai/blog/dont-build-multi-agents>
4. LangChain. *Plan-and-Execute Agents*. Feb 2024.
   <https://blog.langchain.com/planning-agents/>
5. LangChain. *LangGraph: Multi-Agent Workflows*. Jan 2024.
   <https://blog.langchain.com/langgraph-multi-agent-workflows/>
6. Fourney et al. *Magentic-One: A Generalist Multi-Agent System for Solving
   Complex Tasks*. arXiv:2411.04468, Nov 2024.
   <https://arxiv.org/abs/2411.04468>
7. *(Underlying papers for LangChain's planning patterns:)*
   Wang et al., *Plan-and-Solve Prompting* (2023);
   Xu et al., *ReWOO* (2023);
   Kim et al., *LLMCompiler* (2024).

## Appendix — Files preserved in `src/legacy/`

The user's previously drafted planner code is moved (not deleted) to
`src/legacy/`:

- `legacy/planner.py` — `PlannerAgent` with `guided_json` schema. Concept
  is correct; will be ported to `src/core/planner.py` with extensions
  (`tool_subset`, `depends_on`).
- `legacy/router.py` — regex-first intent matcher. Subsumed by the
  existing `src/core/router.py` (more sophisticated).
- `legacy/orchestrator.py` — LangGraph-based plan-execute loop. The
  topology is right; the new `src/core/executor.py` will follow the
  same shape but add DAG / parallel execution.

See `src/legacy/README.md` for the full provenance.
