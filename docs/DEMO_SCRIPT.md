# FinAI Demo Script

A curated walkthrough that exercises every shipping flow with the
right query, the right setup, and a talking-point cheat sheet so
nothing's improvised.

## Pre-flight checklist

1. Confirm the container is running:
   ```bash
   curl -s http://localhost:8000/health | python3 -m json.tool
   ```
   Expected: `"status": "healthy"`.

2. Confirm the demo env vars are in effect:
   ```bash
   docker exec finai-api env | grep FINAI_
   ```
   Expected:
   ```
   FINAI_VERBOSE_TRACE=0
   FINAI_PLANNER_PREFIX=0
   ```

3. Confirm the LibreChat web UI loads at http://localhost:3080
   (or wherever LibreChat is configured).

4. **Avoid these queries during the demo** — they'll surface a
   friendly error but it's not a great showcase:
   - `did Tesla deliver on FSD?` (or any deep-claim-tracking query
     that triggers `deep_stock_research`) — context-window issue
     in the deepagents harness; flagged as a known limitation in
     the migration docs and is being addressed post-demo.

## Flow 1 — Smalltalk (warm-up)

**Query:** `hi`

**What to expect:** ~1-second reply, single short paragraph
acknowledging the user and listing capability bullets.

**Talking point:**
> "FinAI doesn't pretend to be a generic chatbot — even a casual
> 'hi' is routed through the classifier, but it short-circuits to
> a deterministic friendly opener so we don't waste an LLM call on
> social pleasantries."

## Flow 2 — Capabilities discovery

**Query:** `what can you do?`

**What to expect:** ~4kb static markdown listing the 7 supported
flows + supported tickers + example queries. Zero LLM calls.

**Talking point:**
> "When users ask the system to introduce itself, we serve a
> curated capability sheet rather than letting an LLM hallucinate
> features we don't have. This is the `meta_help` flow — entirely
> deterministic, sub-100ms response."

## Flow 3 — Educational (concept explanation)

**Query:** `what is EBITDA?`

**What to expect:** Markdown definition + numbered "How it
works" steps + worked example. ~5-second LLM stream.

**Talking point:**
> "Concept questions don't need tools or panels — just a focused
> LLM call with a system prompt that enforces structure. Notice
> we don't add a regulatory disclaimer here because there's no
> investment claim being made."

## Flow 4 — Stock research (single-ticker live data)

**Query:** `tell me about Western Digital`

**What to expect:** Italic status lines streaming as the agents
work (`Resolving market...`, `Pulling fundamentals...`, etc.)
followed by a structured report:
- **Company Overview** with sources
- **Live Fundamentals** table (price, P/E, P/B, P/S, ROE, ROA,
  margins, debt/equity, revenue)
- **Recent Catalysts** — 5 latest news items with URLs
- **Analyst Synthesis** — investment thesis, what's working, risks,
  what to watch

Total time: ~50-90s. Disclaimer footer at the end.

**Talking points:**
> "This is the bread-and-butter flow. Five MCP workers — research,
> us_stock, indian_stock, portfolio, filings — feed live data into
> a synthesis agent. Each italic status line is a real tool call
> happening right now; the audience can watch the agent graph
> work in real time."

> "Notice the 'sources' on the company overview — every claim is
> grounded in a real URL. The fundamentals are live yfinance
> data, not pre-cached fixtures."

**Optional:** Try `tell me about TCS` for an Indian stock — the
flow auto-detects the market and converts INR to USD.

## Flow 5 — Topic research (open-ended sector / macro)

**Query:** `outlook for Indian IT services in 2026`

**What to expect:** Top web-search hits with full URLs +
"Analyst Briefing" — a structured LLM summary that synthesises
the search results into bullets. Includes inline citations
(`【url】`-style brackets).

Total time: ~25-40s.

**Talking points:**
> "When the user asks an open-ended sector or macro question
> there's no specific ticker to pull data for, so we run a single
> Research Agent web search and ask the LLM to synthesise. Each
> bullet cites a specific URL inline — the audience can verify
> any claim by clicking through."

## Flow 6 — Portfolio analysis (multi-ticker, no panel)

**Query:** `analyze my portfolio for risk and opportunities`

**What to expect:** Long, structured report (~25kb):
- **Portfolio at a Glance** — 11-row holdings table with weights,
  P&L per holding, sector allocation, concentration flags
- **Market Snapshot** — live fundamentals for every holding
- **Recent Catalysts** — news per ticker
- **Qualitative Enrichment** — moat signals, growth, defensive
  metrics tables
- **Analyst Summary** — synthesised commentary with explicit
  "Suggested next questions" pointing at the panel

Total time: ~90-150s.

**Talking points:**
> "This is what most demos lead with. The portfolio is sample
> data; the market data is live. The orchestrator runs a focused
> set of agents (no panel debate yet) and produces a report that
> explicitly flags concentration risks (red dot for >15% in one
> stock, yellow for >30% in one sector)."

> "Notice the 'Suggested next questions' at the end — those are
> deliberate prompts to lead the audience into the panel debate
> for the next demo."

## Flow 7 — Panel debate (the headliner)

**Query:** `run a panel debate on my portfolio`

**What to expect:** Same structural skeleton as Flow 6, but the
synthesis section becomes a 5-round, 3-persona debate that
short-circuits the moment all three personas reach the same
stance label (true consensus). Typical runs land in Round 3 or
Round 4; genuinely divergent panels (e.g. bullish/cautious/bearish
across the three personas) run all 5 rounds and end in an
explicit "agree to disagree".

The five rounds each have a distinct character — the personas
don't just repeat themselves:
- **Round 1 — Opening** — Buffett (long-term moat / quality),
  Wood (innovation / TAM / disruption), Graham (margin of safety /
  defensive metrics) each open with their independent read
- **Round 2 — Rebuttal** — each persona must cite a specific
  claim from another panelist and either AGREE, CHALLENGE, or
  REFINE it
- **Round 3 — Reconsideration (steel-man)** — only runs if no
  consensus yet; each persona articulates the strongest argument
  *against* their own view, then decides honestly whether it
  changes their evaluation
- **Round 4 — Bridge-Building** — only runs if still no
  consensus; each persona names at least one point of genuine
  agreement with each of the other two panelists before
  restating where they still differ
- **Round 5 — Final Position** — closing round; explicitly
  flags any irreducible "agree to disagree"
- **Final Verdict** — moderator synthesis + chat-pane stance
  summary line

Total time: ~3-6 minutes depending on how many rounds run.

**Talking points:**
> "The panel is what differentiates FinAI from a glorified
> chatbot. Each persona has its own system prompt, model
> parameters, and tool subset. Round 1 is parallelised — all
> three personas stream simultaneously."

> "Watch how the rounds escalate. Round 2 is rebuttal — they're
> challenging each other. Round 3 is *steel-manning* — each
> persona literally articulates the strongest opposing argument
> as that opponent would put it, then decides honestly if it
> changes their mind. That's a quality of debate you almost
> never see in stock-research output."

> "Round 4 is bridge-building — the personas pivot from arguing
> to finding common ground. They name specific points of
> agreement with each other before flagging where they still
> differ. The panel typically converges here, on consensus
> that's been *earned* through the prior rounds — not because
> they all happened to start with the same view."

> "Notice the stance summary line in the chat. Whether the
> debate ends in consensus (`🟢 Buffett bullish · 🟢 Wood bullish ·
> 🟢 Graham bullish — converged Round 3`) or in standing
> divergence (`🟢 Wood bullish · 🟡 Buffett cautious · 🔴 Graham
> bearish — divergent after 5 rounds`), the user gets the
> verdict at a glance."

**If time permits:**
- Try `/artifact run a panel debate on my portfolio` — same
  flow but the heavy markdown lands in LibreChat's side pane,
  Claude.ai-style.

## Flow 8 — Hidden / advanced (optional, audience-dependent)

These are NOT on the main demo path but exist if the audience
asks "can it do X?":

| Trigger | Flow | Note |
|---|---|---|
| `/trace <query>` | any flow + classification card | Shows the routing card so the audience can see how the classifier picked the flow |
| `/artifact <query>` | any heavy flow | Routes the report into LibreChat's side pane instead of inline |
| Indian-stock query (`tell me about RELIANCE`) | stock_research | Same flow, different agent (indian_stock) — exercises the multi-jurisdiction code path |

## Common Q&A

**Q: "Why are some metrics blank in the table?"**
> Live data tools occasionally return partial responses (e.g.
> HDFCBANK doesn't disclose debt/equity through yfinance). We
> render an em-dash rather than fabricating a number.

**Q: "What model is this running on?"**
> NVIDIA NIM-hosted GPT-OSS-120B for most flows; the panel
> personas are pinned to per-persona NIM API keys so all three
> can stream simultaneously without rate-limit collisions.

**Q: "What about the planner / deep agent stuff in the docs?"**
> The planner-first multi-agent architecture is shipping in
> stages. Stages 2 and 3 are landed — Stage 4 (factories for the
> remaining four agents) and Stage 5 (full panel slice) are the
> next two milestones. The `/planner` slash-command is gated
> behind `FINAI_PLANNER_PREFIX=1` and turned off in the demo
> environment until the panel-slice factories ship.

## Recovery scripts (if something goes wrong)

* **Stale container after a code change:**
  ```bash
  docker restart finai-api && sleep 10 && \
    curl -s http://localhost:8000/health
  ```

* **MCP tools not loading:**
  Check `docker logs finai-api --tail 50 | grep "Loaded.*MCP tools"`.
  Should show `Loaded 34 MCP tools`.

* **Panel run hangs:**
  The flow has a 7-minute hard cap. If it's clearly stuck (no
  output for >60s), kill the request and try a simpler query
  like `tell me about WDC` to confirm the basic stack still
  works.
