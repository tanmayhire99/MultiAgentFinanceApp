# FinAI — From Demo to Product

A research-grounded strategy memo + executable 12-month product plan, with all the
upstream-verified citations needed to defend the architecture choices.

> **TL;DR (one paragraph).** Your current demo is genuinely ahead of what most
> 2024-2025 academic papers describe (real multi-agent debate loop, working
> claim-tracking pipeline, SEC + BSE + Screener + NSE coverage, planner-first
> deep-research pipeline with replan loop). Turning it into a product means closing four gaps the literature
> identifies as critical: persistent **memory**, reliable **numerical reasoning**,
> domain-fine-tuned **models**, and **persona fidelity**. The recommended path:
> build an Indian-native financial LLM ("Ganga-LLM"), layer fine-tuned investor
> personas (Buffett, Wood, Graham, Marks, Lynch) on top via LoRA, ship through a
> B2C retail app whose primary engagement loop is event-driven alerts on
> portfolio holdings, and continuously evaluate every release on FinBen +
> persona-fidelity benchmarks.

---

## Strategic decisions made (2026-04, updated 2026-05)

These are the four product directions chosen at the end of the research:

| Decision | Choice |
|---|---|
| **Product wedge** | B2C India retail (freemium SaaS targeting the ~5M active retail investors) |
| **Critical-path feature** | Event-driven alerts on portfolio holdings (the engagement multiplier) |
| **Persona fine-tuning depth** | Option C — full domain pretraining + persona LoRAs (6 months, $50-200k) |
| **Evaluation infrastructure** | Wire up FinBen + persona-fidelity + numerical-accuracy benchmarks this quarter |

Everything below assumes these four choices.

---

# Part I — Research-Grounded Memo

## 1. The four gaps separating demo from product

The 2023-2026 literature converges on four problem areas. Every serious financial-AI
paper addresses at least one of them.

1. **Memory** — agents that know your portfolio history, your risk tolerance,
   your past reasoning, across sessions and years (FinMem, FinCon).
2. **Reliable numerical reasoning** — the frontier's single biggest gap; GPT-4
   gets only **58%** on data-based reasoning (QRData, ACL 2024). The fix is
   hybrid: LLM for narrative, code/tools for numbers.
3. **Domain-fine-tuned models** — open-source evidence (FinGPT, PIXIU/FinMA,
   InvestLM) shows LoRA + carefully curated instruction data beats a generic
   frontier model on financial tasks at 1-2% of the cost of pretraining from
   scratch (which is what BloombergGPT spent $10M+ doing).
4. **Persona fidelity** — Character-LLM (EMNLP 2023) shows you can train an
   LLM to act as a specific individual from their writings alone. Buffett has
   **48 public shareholder letters (1977-2024)** freely available — ~2-3M
   words of him writing in his own voice — which is more training material
   than Character-LLM had for Beethoven.

## 2. Academic state-of-the-art (2023-2026)

All papers below verified upstream on arxiv.org.

### 2.1 Foundation models for finance

| Paper | Year | What it establishes |
|---|---|---|
| **BloombergGPT** ([2303.17564](https://arxiv.org/abs/2303.17564)) | Mar 2023 | 50B params, trained on 363B Bloomberg tokens + 345B general. The gold standard, but **proprietary** — no release, locked inside Terminal. |
| **FinGPT** ([2306.06031](https://arxiv.org/abs/2306.06031)) | Jun 2023 → Nov 2025 | Open-source alternative. **Data-centric + LoRA** approach. Proves you don't need 50B params or 363B tokens to beat generic LLMs on financial tasks. |
| **PIXIU / FinMA** ([2306.05443](https://arxiv.org/abs/2306.05443)) | Jun 2023 | First open FinLLM + 136K-sample instruction dataset + benchmark of 5 tasks / 9 datasets. |
| **InvestLM** ([2309.13064](https://arxiv.org/abs/2309.13064)) | Sep 2023 | **Superficial Alignment Hypothesis for finance**: a small (few-thousand) well-curated instruction set is enough to make a base LLaMA-65B comparable to GPT-3.5/4 on investment Q&A per hedge-fund-manager ratings. **Huge implication for fine-tuning plan.** |
| **Plutus** ([2502.18772](https://arxiv.org/abs/2502.18772)) | Feb 2025 | Demonstrates a viable template for **country-specific** financial LLMs (Greek). Directly transferable: India could have its own "Ganga-LLM" fine-tuned on Indian filings + business media + concall transcripts. |

### 2.2 Multi-agent architectures

| Paper | Year | Key contribution |
|---|---|---|
| **FinMem** ([2311.13743](https://arxiv.org/abs/2311.13743)) | Nov 2023 | LLM trading agent with **Profiling + Layered Memory + Decision-making** modules. Memory mirrors human trader cognition — layered (short/mid/long-term), with adjustable perceptual span. Character design explicitly part of the architecture. **Panel personas should inherit from this memory pattern.** |
| **FinCon** ([2407.06567](https://arxiv.org/abs/2407.06567), NeurIPS 2024) | Jul 2024 → Nov 2024 | **Manager-analyst hierarchy** (like real investment firms), synchronized cross-functional collaboration, **risk-control self-critique** that periodically updates "conceptualized beliefs" propagated only to agents that need them. Outperforms single-agent baselines on stock trading + portfolio management. |
| **TradingAgents** ([2412.20138](https://arxiv.org/abs/2412.20138), v7 Jun 2025) | Dec 2024 → Jun 2025 | Fundamental analyst + Sentiment analyst + Technical analyst + Bull/Bear researchers + **Risk management team** + Traders w/ varied risk profiles. Reported improvements on Sharpe + max drawdown vs baselines. **[Open-source on GitHub](https://github.com/TauricResearch/TradingAgents)** — study their architecture. |
| **FinRobot** ([2405.14767](https://arxiv.org/abs/2405.14767)) | May 2024 | Open-source 4-layer platform: AI Agents → LLM Algorithms → LLMOps+DataOps → Foundation Models. Template for a production-grade architecture. |

### 2.3 Persona / character training

| Paper | Year | Relevance |
|---|---|---|
| **Character-LLM** ([2310.10158](https://arxiv.org/abs/2310.10158), EMNLP 2023) | Oct 2023 | **The canonical method**: edit profiles as "experiences of a certain character" → train LLM as personal simulacrum. Validated on Beethoven, Cleopatra, Julius Caesar. Evaluation = interview the trained agent and check it "remembers" its life. Directly applicable to Buffett/Wood/Graham. |
| **Anthropic's Character Training** (Claude constitution blog, 2024) | 2024 | Uses a constitution-based method to instill character traits. Less direct than Character-LLM but worth noting. |

### 2.4 Benchmarks & evaluation

| Benchmark | Year | What it measures |
|---|---|---|
| **FinBen** ([2402.12659](https://arxiv.org/abs/2402.12659)) | Feb 2024 | **36 datasets, 24 tasks, 7 aspects**: IE, textual analysis, QA, text generation, risk management, forecasting, decision-making. First to evaluate agent + RAG on finance + **stock trading**. The product needs to benchmark here for credibility. |
| **FinPT / FinBench** ([2308.00065](https://arxiv.org/abs/2308.00065)) | Jul 2023 | Profile Tuning for tabular risk prediction (default, fraud, churn). Useful for a *credit/underwriting* product extension. |
| **QRData** ([2402.17644](https://arxiv.org/abs/2402.17644), ACL 2024) | Feb 2024 | GPT-4 gets **58% on data-based reasoning**, Deepseek-coder 37%. **The single biggest warning sign for pure-LLM financial analysis** — the numbers bite back. |

### 2.5 Reasoning & planning (2025 frontier)

| Paper | Year | Relevance |
|---|---|---|
| **LLM Test-Time Compute Survey** ([2501.10069](https://arxiv.org/abs/2501.10069), TMLR) | Jan 2025 → Apr 2025 | Comprehensive review of search-based inference (ToT, LATS, Best-of-N, MCTS-style). A framework-level understanding of how to get more out of a fixed base model via inference-time compute. |

## 3. Commercial landscape — what's already shipped

| Product | Target | What it does | Price | Gap to attack |
|---|---|---|---|---|
| **Bloomberg Terminal + GPT** | Institutional | 50B proprietary LLM inside terminal | ~$25k/user/year | Retail locked out; India poorly covered |
| **AlphaSense** | PM, research | AI search over brokers' research, transcripts | $10-25k/user/year | Retail / small RIAs locked out |
| **Hebbia** | Asset managers | Agentic research over their documents | Enterprise | No Indian document corpus |
| **Rogo** | Investment bankers | "AI research analyst" — models + memos | Enterprise | Sell-side focus, not buy-side |
| **Tegus / AlphaSense Calls** | Analyst research | Expert network + AI over transcripts | $500+/mo | US-only experts |
| **Kensho / S&P** | Institutional | Finance-specific LLM inside S&P data | Enterprise | Locked to S&P customers |
| **Morgan Stanley AI @ Work** | MS advisors only | GPT-4 over MS research corpus | Internal | Employees only |
| **Magnifi** | US retail | AI chat for robo-advisor | Freemium + AUM | US ETFs, basic reasoning |
| **Harvest / AdvisorAI** | RIAs | Copilot for US financial advisors | SaaS | US-focused, no agent depth |
| **Trendlyne / Smallcase** | India retail | Screens, model portfolios | Freemium | No LLM reasoning, no agent |
| **Tijori / Ticker** | India retail | Company analytics | Freemium | No LLM, no advisor-style debate |

**The gap:** nobody's shipped a serious retail / small-RIA-grade agentic
financial research product for the Indian market, nor a product built around
the "investor panel" UX. That's the product-market fit slice worth pursuing.

## 4. Where FinAI sits today — honest gap analysis

What the demo does well:

- Multi-agent debate with convergence detection (matches FinCon / TradingAgents architecture)
- Planner-first pipeline: LLM planner generates DAG of ScopedAgent steps, executor runs independent steps in parallel, joiner decides finish/replan/abort
- Claim-tracking pipeline with SEC EDGAR + BSE + Screener + NSE + PDF extraction
- Retrieval post-processing: semantic re-ranking (all-MiniLM-L6-v2), dedup (cosine θ=0.90), freshness filtering, date-window filtering — wired into all search functions
- `run_python` synthetic tool — any agent can execute verified Python
- API key cycling via `_CyclingChatOpenAI` — transparent rotation on rate-limit
- Search backend chain: Tavily (agent-optimized) → DDG (free) → fixture, with Tavily singleton client + negative caching
- LibreChat-compatible SSE streaming UI
- FinBen baseline: 71.1% overall (professional_accounting 26.7%, econometrics 60%)

What it lacks vs a real product:

- **No persistent user memory.** Every session is cold. No idea who the user is between runs.
- **No personalised risk profile.** Buffett / Wood / Graham give the same advice to a 25-year-old engineer as to a 55-year-old retiree.
- **Numerical reasoning is weak.** `run_python` synthetic tool ships (any agent can execute verified Python), but no `verify_numbers` post-processing pass yet — the synthesizer still renders unchecked LLM-computed numbers. This is the QRData problem.
- **No learning loop.** The system doesn't improve from its past verdicts.
- **No real evaluation.** FinBen baseline recorded (71.1% overall), but no CI gate blocks regressions on model/prompt changes.
- **No regulatory framing.** Zero visible SEBI/SEC disclaimers, no compliance audit trail, no "this is not advice" enforcement beyond a footer.
- **No multi-modal output.** Everything is markdown text. No dashboards, charts, alerts, voice, scheduled reports, or PDF exports.
- **No broker / portfolio-sync integration.** Portfolio is a static fixture.
- **No real-time events.** Agents react to what the user asks, not to what's happening in the market.
- **Personas are system-prompted, not fine-tuned.** Buffett's language is generic "value investor" talk, not his actual writing style.
- **No API authentication.** Endpoints are open — no auth, no rate limiting, CORS allows all origins.
- **No per-step timeout.** A stuck agent blocks the entire pipeline indefinitely.
- **No cache eviction.** Disk-backed response cache grows without bound.
- **Python version mismatch.** Dockerfile=3.11, pyproject.toml=3.12, conda=3.13 — must pin to 3.12 everywhere.

## 5. The persona fine-tuning question — yes, and here's how

### 5.1 The data is there

Publicly available writings per persona (all verified upstream):

| Persona | Primary source | Volume | Additional |
|---|---|---|---|
| **Warren Buffett** | [48 Berkshire letters 1977-2024](https://www.berkshirehathaway.com/letters/letters.html) | ~2-3M words | Annual meeting transcripts (CNBC, Yahoo Finance), book *The Essays of Warren Buffett* (Cunningham), interview transcripts |
| **Cathie Wood** | ARK "Big Ideas" annual reports 2017-2026, ARK weekly research | ~800k-1.2M words | CNBC/Bloomberg interview transcripts, ARK podcasts |
| **Benjamin Graham** | *The Intelligent Investor* (1949, partially public domain), *Security Analysis* (1934, public domain), Columbia lectures | ~400-600k words | Limited to historical corpus but stylistically distinctive |
| **Peter Lynch** (bonus persona) | *One Up On Wall Street*, *Beating the Street*, Magellan fund letters | ~300k words | "Invest in what you know" is a distinctive stance for GARP |
| **Seth Klarman** (bonus) | *Margin of Safety* (1991, out of print but pirated copies ubiquitous), Baupost letters | ~200k words | Defensive value complement |
| **Howard Marks** (bonus) | [Oaktree memos](https://www.oaktreecapital.com/insights/memos) (free, since 1990) | ~1M words | Market-cycle focus, great counter-voice to Wood |

**Total available training corpus across ~6 personas: roughly 5-6 million words.**
Character-LLM trained on much less and it worked.

### 5.2 Three options, ordered by effort

**Option A — Style-only LoRA per persona (1-2 weeks per persona, $500-2k per persona)**

- Fine-tune LoRA adapters on top of the base model (LLaMA-3.1-70B,
  Qwen-2.5-72B, or gpt-oss-120b if weights are accessible)
- Supervised fine-tuning (SFT) on passages from the persona's own writing
- Each persona gets its own adapter; swap at inference time
- Produces: language style + distinctive phrases + signature arguments
- Does NOT give them new knowledge beyond the base model
- Runs on a single H100 with Axolotl / Unsloth / LLaMA-Factory

**Option B — Character-LLM method (~4 weeks per persona, $2-10k per persona)**

- Convert each persona's writings into **scene-formatted training data**:
  `[Question from interviewer] → [Buffett's response in his voice, with his
  reasoning]`
- Supplement with synthetic Q&A generated by a frontier model (GPT-5 / Claude)
  in Buffett's style, grounded in actual Berkshire letters
- SFT + DPO (direct preference optimization) on the dataset
- Produces: durable in-character behavior AND factual knowledge of the
  persona's past decisions
- This is what the Character-LLM paper validated; it works

**Option C — Full domain pretraining + persona LoRAs (~3-6 months, $50-200k)**
**[CHOSEN]**

- Follow the **BloombergGPT / Plutus** recipe: take an open base model
  (LLaMA-3.1-70B) and continue-pretrain on 50-100B tokens of financial text
  (BSE + NSE + concall transcripts + business media + SEC filings)
- Then layer persona LoRAs on top of the domain-pretrained base
- Produces: better finance reasoning + in-character personas
- Expensive but this is the moat — nobody else has done this for Indian finance

### 5.3 Critical legal note

Using someone's writings to train an LLM-persona of them has a real-but-unresolved legal status:

- **Buffett's letters**: copyrighted but "reproduced with permission" per the
  Berkshire site. Training on them is arguably fair use but untested in court.
- **Graham**: *Security Analysis* is public domain (pre-1964, but later
  editions are copyrighted); *Intelligent Investor* is under copyright in
  modern editions.
- **ARK / Wood**: most research is explicitly "no reproduction without
  permission" per ARK's terms.

For a commercial product, consider:

- Calling the personas "Value Investor (inspired by Buffett)", "Disruption
  Investor (inspired by Wood)", "Defensive Investor (inspired by Graham)" —
  similar to how Character-LLM frames theirs as "trainable agents for
  role-playing"
- Getting counsel before shipping any fine-tuned persona to paying users
- Using only out-of-copyright passages for direct training where possible

## 6. Ten big product extensions (research-paper-backed)

Each has real research validating it. Ordered by priority for the product.

### 1. Persistent user memory with preference learning [FinMem, Character-LLM]

**What**: Per-user memory store that tracks portfolio history, expressed risk
preferences, past queries, past verdicts the user agreed/disagreed with.
Implemented via LangGraph memory store + a vector store keyed by user_id.

**Why**: The #1 reason B2C AI apps feel like toys is amnesia. A real product remembers.

**Effort**: 2-3 weeks for v1 (LangGraph memory), 2 more months for preference-learning RLHF-lite.

### 2. Risk-profile-aware panel moderator [new]

**What**: A user onboarding flow (5 questions: age, horizon, income,
loss-tolerance, goals) produces a risk profile. The panel moderator weights
each persona's voice by the profile — a 55-year-old retiree hears more Graham
and less Wood.

**Why**: Generic advice is worthless. Weighted advice is defensible. SEBI/SEC
approved "suitability" check.

**Effort**: 3-4 weeks.

### 3. Hybrid numerical reasoning [QRData, FinBen]

**What**: Every numeric claim the LLM makes gets verified in Python. The `run_python` synthetic tool is shipped (agents can execute code), but the `verify_numbers_in_text(draft_text, known_data)` post-processing pass that the synthesizer calls before rendering is not yet implemented. Catches hallucinated numbers.

**Why**: GPT-4 only gets 58% on QRData. Cannot ship a financial product that
hallucinates numbers.

**Effort**: 3-4 weeks.

### 4. Persona fine-tuning (Phase 1 — LoRA style transfer)

**What**: 3 persona LoRAs for the base model, using Character-LLM method on
Berkshire letters / ARK research / Graham's books.

**Why**: The #2 reason this feels like a demo is that Buffett doesn't sound
like Buffett. Fine-tuning fixes that in a measurable, benchmarkable way.

**Effort**: 4-6 weeks end-to-end (data prep + 1 weekend of training per
persona + evaluation).

### 5. Scheduled / event-driven agent runs [new]

**What**: Agents run on a schedule (e.g. "every Monday morning, deep-research
my portfolio and alert me to any claim-verdict flips") or on events ("when
NVDA files an 8-K Item 2.02, re-run the claim-tracking and notify me if any
past claim just became verifiable").

**Why**: A product that only runs when you open it is a feature. A product
that watches your portfolio while you sleep is valuable.

**Effort**: 3-4 weeks (cron + webhook + LLM summarization of alert-worthiness).

### 6. Quantitative strategy sandbox [TradingAgents, FinMem]

**What**: Let the user backtest the panel's verdicts. "If I had followed
Buffett's panel recommendation on NVDA in 2023, what would my return be
today?" Requires historical price data, historical panel runs (cache them!),
and a backtesting engine.

**Why**: Proves the panel's edge. Also exposes weak personas you can then improve.

**Effort**: 2 months (historical backtests are tricky).

### 7. Real retail broker integration [new]

**What**: Read-only OAuth integration with Zerodha Kite, Upstox, Groww
(India), plus Robinhood / Fidelity / Schwab (US). Portfolio is live, not
fixture.

**Why**: The moment portfolio data is live, engagement doubles. Users check in more often.

**Effort**: 1-2 months per broker (OAuth + data mapping + compliance).

### 8. Structured compliance + audit trail [SEBI LODR, MiFID II, SEC Reg BI]

**What**: Every LLM output is logged with (query, retrieval context, tools
called, personas' verdicts, final text, user acknowledgement). Built-in
disclaimers that adapt to the user's jurisdiction and product tier.
"Educational content" vs "personalised advice" badge on every output.

**Why**: Mandatory for monetisation in regulated markets. Also a moat — most
AI products ship without this and can't scale past hobby tier.

**Effort**: 4-6 weeks.

### 9. Benchmark suite + continuous evaluation [FinBen, PIXIU, QRData]

**What**: Run every system change through FinBen (24 tasks) + QRData
(numerical reasoning) + a custom "persona fidelity" benchmark (the version of
Character-LLM's interview test). Track weekly; block releases that regress.

**Why**: Without this, every prompt tweak is a roll of the dice. With it, ship with confidence.

**Effort**: 3-4 weeks to wire up, then amortised.

### 10. Country-specific domain LLM (Phase 3) [BloombergGPT, Plutus]

**What**: Continue-pretrain LLaMA-3.1-70B on 50B tokens of Indian financial
text (BSE announcements 2015+, NSE, Screener pages, Moneycontrol, Livemint,
concall transcripts, 10+ years of Indian business news). Call it "Ganga-LLM"
or similar. License it.

**Why**: This is the defensible moat. Nobody has an Indian-native financial
LLM. BloombergGPT is US-first; Plutus proved country-specific LLMs are
viable; you have the distribution.

**Effort**: 6 months + significant compute budget ($50-200k).

## 7. Architecture for a production system

```
┌─────────────────────────────── USER LAYER ───────────────────────────────┐
│  Web (React) · iOS · Android · Slack · WhatsApp · Scheduled Reports      │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌─────────────────────────── API GATEWAY ──────────────────────────────────┐
│  FastAPI · Auth (OAuth + SAML) · Rate-limit · Compliance Interceptor     │
└──────────────────────────────┬───────────────────────────────────────────┘
                               │
┌────── ORCHESTRATION (LangGraph + planner-first pipeline) ─────────────────┐
│ • Intent Router (existing) → Planner → DAG of ScopedAgent steps │
│ • Joiner (rule-based v0) → finish / replan / abort (max_replans=2) │
│ • Persistent User Context (memory store + preferences + portfolio) │
│ • Risk-Profile-Aware Moderator │
│ • Deep Research / Panel / Quick / Educational (via planner DAG) │
│ • Parallel step execution + replan on synth failure │
│ • NEW: Event-Driven Agent (watchers on SEC 8-K, BSE Reg 30, news) │
│ • NEW: Scheduled Runner (cron + Celery) │
└──────┬───────────────────────────────────────────────────────────────────┘
       │
┌──── AGENT POOL ──────────────────────────────────────────────────────────┐
│  Buffett LoRA · Wood LoRA · Graham LoRA · Marks LoRA · Lynch LoRA ·      │
│  Moderator LoRA · Analyst LoRA  (base = Ganga-LLM with adapter swap)     │
└──────┬───────────────────────────────────────────────────────────────────┘
       │
┌──── TOOL POOL (via MCP) ─────────────────────────────────────────────────┐
│  • Market data: yfinance, Polygon, Alpha Vantage, Alpha Vantage India    │
│  • Filings: SEC EDGAR (existing), BSE (existing), NSE (existing)         │
│ • Research: Tavily, DDG, Screener.in (existing) │
│ • Retrieval pipeline: re-rank + dedup + freshness + date-window (shipped) │
│ • Claim tools: extract, compare (existing) │
│ • Retrieval pipeline: semantic re-ranking + dedup + freshness + date-window (shipped) │
│ • NEW: Python sandbox for verified numerical reasoning (`run_python` shipped) │
│  • NEW: Backtester (historical prices + strategy runner)                 │
│  • NEW: Broker read (Zerodha, Upstox, Robinhood, Schwab OAuth)           │
│  • NEW: Alerts (email, WhatsApp via Twilio, Slack, push)                 │
└──────┬───────────────────────────────────────────────────────────────────┘
       │
┌──── STATE LAYER ─────────────────────────────────────────────────────────┐
│  PostgreSQL: users, portfolios, verdicts, compliance audit trail         │
│  Vector DB (pgvector or Qdrant): semantic memory, persona knowledge      │
│  Redis: hot cache, session state, rate-limit counters                    │
│  S3 / R2: raw SEC + BSE PDFs, concall transcripts, generated reports     │
└──────┬───────────────────────────────────────────────────────────────────┘
       │
┌──── OBSERVABILITY & EVAL ────────────────────────────────────────────────┐
│  LangSmith traces · Prometheus metrics · FinBen eval harness             │
│  Persona-fidelity benchmark · QRData-style numerical accuracy eval       │
│  Compliance audit dashboard · A/B testing framework                      │
└──────────────────────────────────────────────────────────────────────────┘
```

## 8. Monetisation paths & moat

### Three viable paths, in order of defensibility

**(a) B2C retail — Freemium SaaS [CHOSEN]**

- **Free**: 5 queries/month, portfolio read-only, educational content
- **Pro** ($20/mo): unlimited queries, 1-stock deep research + panel per day,
  scheduled reports, 1 broker integration
- **Premium** ($99/mo): unlimited deep research, backtester, multi-portfolio,
  WhatsApp alerts, persona switching, priority compute
- **TAM India**: 300M demat accounts, of which ~5M active retail traders
  would pay for a tool.
- **Realistic**: 10k paying users × avg $25/mo = $250k MRR within 12-18 months
  of product launch.

**(b) B2B2C — White-label to brokers / wealth-tech**

- License the panel + deep-research to Zerodha, Groww, Upstox as a "research
  tab". They pay $10-50 per active user per year, branded as their product.
- Cuts out the distribution problem. Brokers already have the users.
- Risk: brokers build it themselves, or squeeze margins over time.

**(c) B2B — RIAs / small hedge funds / family offices**

- Enterprise tier ($500-2000/seat/mo). Compliance + audit trail + private data
  isolation + fine-tuned-on-their-house-views personas.
- Higher ACV, stickier, but needs a full sales motion and enterprise security
  certifications (SOC 2, ISO 27001).

### Moat components (ranked by durability)

1. **Fine-tuned personas** — hard to replicate without the training pipeline +
   eval harness. **3-6 month lead** over a competitor who notices.
2. **Indian filings + news corpus** at scale — ingesting 10+ years of BSE +
   NSE + Screener + concall transcripts is a data moat. **1-2 year lead** if
   done properly.
3. **Claim-tracking benchmark dataset** — once 1000+ claim→reality pairs for
   Indian and US stocks are annotated, that's proprietary. Nobody else has it.
4. **User portfolios + preference history** — classic product moat.
   **Permanent** for retained users.
5. **Compliance certifications** — SEBI research-analyst registration or
   equivalent is a regulatory moat in India.
6. **Broker integrations** — each OAuth+data mapping is 1-2 months and a
   business-dev investment.

Weakest moats: the LLM orchestration, the planner-first pipeline, the MCP
tooling — all open-source commodities. Don't rely on these as moats.

---

# Part II — Product Plan: "FinAI Pro"

Indian Retail · Domain-LLM-Powered · Alerts-First.

## 1. The thesis in one paragraph

Build the first **Indian-native financial LLM** ("Ganga-LLM") by
continue-pretraining LLaMA-3.1-70B on 50-80B tokens of Indian filings +
concall transcripts + business media, layer **fine-tuned investor personas**
(Buffett, Wood, Graham, Lynch, Marks) on top via LoRA, expose it through a
B2C **retail app for Indian investors** whose primary engagement loop is
**event-driven alerts on portfolio holdings** ("NVDA just filed an 8-K Item
2.02 — your panel re-ran claim-tracking, here's what changed"). Ship every
release through a continuous **FinBen + persona-fidelity evaluation harness**
so quality only goes up.

## 2. The corpus for Option C (full domain pretraining)

The Indian-native LLM needs **50-80 billion tokens** of high-quality
finance-relevant text. The math: BloombergGPT used 363B Bloomberg tokens;
Plutus used much less for Greek and got publishable results. For India-focused,
50B is a defensible target.

| Source | Volume estimate | Acquisition |
|---|---|---|
| **BSE corporate filings 2010-2025** (announcements + Q-results + audited annuals) | ~8B tokens | Scrape via `src/mcp/_indian_filings.py` + extend back to 2010 |
| **NSE corporate filings** (overlap with BSE but unique press releases) | ~3B tokens | curl_cffi-based scraper (existing pattern) |
| **Concall transcripts 2014-2025** (top 500 NSE companies × ~40 quarters) | ~5B tokens | Screener.in scrape + IR-page direct + Motley Fool India |
| **Annual Reports 2010-2025** (top 1000 NSE × 15 years × 250pp) | ~12B tokens | BSE-hosted PDFs via `fetch_pdf_text` (existing) |
| **Indian business media** (Moneycontrol, Livemint, Economic Times, Business Standard, Mint, BloombergQuint) 2015-2025 | ~10B tokens | Crawl via Common Crawl + targeted scrapers; check robots.txt |
| **SEBI consultation papers + circulars + orders** | ~1B tokens | sebi.gov.in (free, public) |
| **MCA filings + judgements** | ~2B tokens | mca.gov.in scrape + court records |
| **US SEC filings 2015-2025** (for cross-market comparability of Indian ADRs + global names) | ~5B tokens | `src/mcp/_sec_edgar.py` + bulk SEC EDGAR archive download |
| **Buffett / Graham / Marks / Lynch source corpus** (for persona LoRAs, NOT the base) | ~50M tokens | Berkshire letters + Oaktree memos + scanned books |
| **CFA curriculum + Indian textbooks (NISM, ICSI)** | ~2B tokens | Some free PDFs; some require licensing |
| **Reddit r/IndianStockMarket / r/IndiaInvestments + Twitter financial India** | ~3B tokens | Pushshift + Twitter scrape (note: ToS) |
| **Total** | ~50-55B tokens | |

> **This data ingestion is itself a 2-3 month project.** Don't underestimate
> it. Most of the Plutus team's time was spent on data, not training.

## 3. Compute budget for Option C, broken down

Continue-pretrain LLaMA-3.1-70B on 50B tokens.

| Phase | Cost | Compute |
|---|---|---|
| Data pipeline + cleaning + dedup + tokenization | $5-15k | A few CPU instances + storage |
| **Continue-pretraining** (1 epoch on 50B tokens, LR ~5e-5) | **$80-180k** | 8x H100 SXM5 cluster for ~3-4 weeks via Lambda / Together / Voltage Park |
| Evaluation suite (FinBen + custom Indian benchmark) wired up | $2-5k | Mostly inference compute on the resulting checkpoint |
| Persona LoRA fine-tunes (5 personas × 1 day H100 each) | $3-5k | Single H100 for each persona |
| Inference hosting (post-training, for product) | $5-15k/month | 2x H100 minimum for low-latency serving + spillover to NIM gpt-oss-120b |

**Total one-time: $90-205k.** Then $5-15k/month inference until past ~5k DAU,
when scaling kicks in.

Realistic mid-point: **$130k for the training run + 6 months of $10k/mo
inference = $190k**.

## 4. Event-driven alerts — the critical-path feature

Architecture for what makes the product sticky:

```
┌── INGEST LAYER (continuous) ────────────────────────────────┐
│ • SEC EDGAR feed (poll every 15 min)                        │
│ • BSE announcements API (poll every 5 min)                  │
│ • NSE announcements (every 15 min via curl_cffi)            │
│ • Tavily news webhook (push)                                │
│ • Yahoo Finance regularMarket events (price triggers)       │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌── EVENT CLASSIFIER (small fast LLM, gpt-oss-120b prompt) ───┐
│ Input: raw filing/news headline + ticker                    │
│ Output: { material: bool, category: ..., urgency: 0-3 }     │
│ Categories: guidance_change, earnings, m&a, mgmt_change,    │
│             rating, lawsuit, product, macro                 │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌── USER MATCHING (Postgres + Redis) ─────────────────────────┐
│ Match the event to users who:                               │
│ • Hold the ticker (broker-sync portfolio)                   │
│ • Have alerts on for this ticker / category                 │
│ • Match the urgency threshold for their tier                │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌── SUMMARIZER + DECIDER (deep agent, scheduled) ─────────────┐
│ For each (user, event) pair:                                │
│ • Run deep-research flow (cached if recent)                 │
│ • Re-run claim-tracking on past commitments related to      │
│   the event topic                                           │
│ • Compose 2-paragraph summary in user's preferred persona's │
│   voice (e.g. "Your Buffett-LoRA panel says…")              │
│ • Score: should we wake the user? (notification budget)     │
└───────────────────┬─────────────────────────────────────────┘
                    │
┌── DELIVERY ─────────────────────────────────────────────────┐
│ WhatsApp (Twilio) · Push · Email · In-app                   │
│ Cool-down: max 2 alerts per ticker per 24h, max 5 per day   │
│ Per-tier: Free=1/wk, Pro=daily, Premium=real-time           │
└─────────────────────────────────────────────────────────────┘
```

**Implementation effort:** ~6-8 weeks for v1.

**The KPI to optimise:** Daily Active Users (DAU) / Weekly Active Users (WAU).
Free tools have ~10% D/W; with portfolio sync + meaningful alerts, this can
reach ~50% — the difference between "abandoned tab" and "checked daily."

## 5. FinBen evaluation infrastructure (this quarter)

Concrete plan to ship by end of Q1.

**Week 1-2: clone + adapt FinBen harness**

- Repo: <https://github.com/The-FinAI/PIXIU> (FinBen + PIXIU live here)
- Wire it as a CI job that runs on every model swap or major prompt change
- Track per-task scores (24 tasks): forecasting, IE, QA, etc.
- Baseline: gpt-oss-120b (current). After fine-tuning: report delta.

**Week 3-4: build a custom Indian financial benchmark**

- ~500 hand-curated Indian-specific Q&A pairs across: BSE/NSE conventions,
  SEBI regulation, INR/USD, rupee crore/lakh, fiscal-year arithmetic, Indian
  sector taxonomy
- Annotate with 3 financial advisors / CFAs in India
- This is the moat for showing "we beat GPT-5 on Indian finance" in marketing

**Week 5-6: persona fidelity benchmark (Character-LLM method)**

- 50-100 questions per persona, evaluated by 3 raters per response
- Compare: system-prompted Buffett vs Buffett LoRA vs Buffett base-LLM
- Report: "Buffett LoRA beats system-prompted Buffett by 22% on persona
  fidelity rating, p<0.01" — this becomes a marketing claim

**Week 7-8: numerical-reasoning audit (QRData-style)**

- 200 finance-specific numerical reasoning Q&A: "Given this 10-Q, what was
  YoY revenue growth?", "Compute Graham number from these inputs"
- Two configurations: pure-LLM vs LLM+Python-sandbox
- Sets the baseline for the hybrid-numerical-reasoning feature shipping in Q2

## 6. First 90 days, week-by-week

The most useful thing in this document. Cumulative effort assumes 1-2 senior
engineers + product lead.

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Data pipeline scaffold | Indian-corpus ingest crew: BSE archive 2010+, NSE archive, Annual Reports back to 2010 |
| 2 | Data pipeline (cont.) | Concall transcripts crawler (Screener + Motley Fool India) |
| 3 | FinBen wiring | Run baseline on gpt-oss-120b, publish internal scorecard (**baseline recorded: 71.1% overall**; wire as CI gate next) |
| 4 | User auth + memory v1 | Auth (Clerk/Supabase), per-user portfolio + preferences in PG, vector store for chat memory |
| 5 | Persona dataset construction | Berkshire letters → ~5000 instruction pairs synthesised via Claude |
| 6 | Persona LoRA training (first attempt) | Buffett LoRA on gpt-oss-120b via Unsloth on rented H100 |
| 7 | Persona fidelity benchmark | Set up 50-question eval, 3 raters; compare system-prompted vs LoRA Buffett |
| 8 | Hybrid numerical reasoning | `verify_numbers` post-processing pass (`run_python` tool already shipped) |
| 9 | Alerts ingest layer | BSE/NSE/SEC poll loops + classifier prompt |
| 10 | Alerts user matching | Per-user holdings table, alert preferences, cool-downs |
| 11 | Alerts summarizer + delivery | WhatsApp/email/push integration; first end-to-end alert delivered |
| 12 | Soft launch | Invite-only beta with 50 friends-and-family Indian investors. Instrument funnel: signup → portfolio sync → first query → first alert → return. |
| 13 | **Begin domain LLM pretraining** | Provision 8x H100 cluster, kick off continue-pretraining with the cleaned corpus from weeks 1-2 |

By week 13, working private beta + the domain LLM training kicking off.
That's a credible 90-day product story.

## 7. The 3-line investor pitch (for fundraising)

> FinAI is the AI investment research product for Indian retail investors.
> We've trained an Indian-native financial LLM and a panel of fine-tuned
> investor personas — Buffett, Wood, Graham, Marks — that debate every claim a
> company makes against what they actually delivered, alerting you the moment
> your portfolio is affected. Built on 15+ years of Indian filings and concall
> transcripts, evaluated rigorously against FinBen (the academic standard),
> and shipped through WhatsApp.

## 8. Top risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Domain LLM training fails to beat the base model** | Medium | Stage gates: after 5B tokens, evaluate vs base. Kill the run if no gain at 10B. Fallback: ship LoRA-on-LLaMA-3.1-70B-instruct without continue-pretraining |
| **SEBI compliance friction (research analyst registration, advice vs education)** | High | Engage SEBI-compliant counsel by week 4. Start as "educational" tier; register as Investment Adviser when revenue justifies (~₹5L/yr threshold) |
| **Persona copyright / IP claims (esp. ARK, Buffett)** | Medium | Frame as "value investor inspired by ___"; avoid direct quotes longer than fair-use snippets in outputs; train on summaries rather than direct text where possible |
| **Compute cost overruns** | Medium | Run a small test (LLaMA-3.1-8B continue-pretrained on 5B tokens, $5-10k) to validate the recipe before committing to 70B |
| **Hallucinated numbers in production** | High → Medium (post-mitigation) | `run_python` tool shipped (agents can compute in Python). Still need: `verify_numbers` post-processing pass that blocks rendering of unchecked numerical claims (Q2, week 8) |
| **Indian retail isn't actually willing to pay** | High | Validate in beta: does ₹500-2000/mo retain 30%+ at month 3? If not, pivot to B2B2C white-label |
| **Bigger players notice and clone** | Medium | The Indian-corpus moat is real (12-18 month lead). The persona-fidelity benchmark moat is real. The compliance certification moat is real. Move fast on these. |

## 9. Suggested team

Bare minimum for a 12-month execution of the above:

| Role | Why |
|---|---|
| **You** | Product + system architecture |
| **1 ML engineer** specialising in fine-tuning + data pipelines | Owns Ganga-LLM training, FinBen eval, persona LoRAs |
| **1 backend engineer** | Owns user/auth, portfolio sync, broker integrations, alerts, deliverability |
| **1 CFA-track domain analyst** (consultant or part-time) | Owns persona dataset curation, benchmark Q&A, content quality |
| **1 designer** (part-time) | Owns the alert UX, panel UI, onboarding flow |
| **Compliance counsel** (consultant) | Owns SEBI registration path, IP risk on personas |

Annual burn: roughly ₹1.5-2.5 cr ($180-300k) for India-based team + $200k for
compute + miscellaneous = **~$400-500k for the first 12 months**. That's the
realistic ask if raising a seed round to ship this.

## 10. What to do this week

1. **Day 1-2**: Set up a separate Git repo for the `ganga-llm` training
   pipeline (clean separation from the demo). Use an existing template like
   Axolotl or LLaMA-Factory.
2. **Day 3-4**: Order the data: kick off the BSE 2010-2025 announcement
   scraper (existing module + back-fill loop). It'll take days to run; let
   it cook.
3. **Day 5**: Spin up the FinBen evaluation harness against the current
   gpt-oss-120b setup. Get a baseline scorecard before any changes.
4. **Day 6-7**: Write the first 100-question Indian benchmark by hand. (This
   alone can be done in a weekend; it's the highest-leverage item to ship.)
   Ground every question in a real BSE filing.

That's a productive, decisive week. By the end of it:

- A baseline FinBen scorecard
- 100 ground-truth Indian Q&A pairs
- A growing Indian-corpus crawl in flight
- A clear separation between "the demo" and "the product"

## 11. Two non-negotiable commitments

You picked the path that has the most defensible moat and the highest
credibility ceiling. It's also the slowest path to first revenue. Two
operating commitments to lock in upfront:

1. **A weekly metric scorecard** (FinBen score, persona fidelity, beta
   DAU/WAU) shared with whoever you're accountable to (advisors, investors,
   yourself). This forces honesty.
2. **A 90-day kill-or-continue checkpoint** on the domain LLM. If by week 13
   the corpus is < 30B tokens and beta retention is < 20%, scope down to
   LoRA-only personas and revisit Option C in a year. Don't pour $200k of
   compute into something the product hasn't earned.

Everything else is execution.

---

## Appendix A — Full citation list

All papers referenced in this memo, verified upstream on arxiv.org as of
April 2026.

1. **BloombergGPT: A Large Language Model for Finance** — Wu et al., 2023.
   <https://arxiv.org/abs/2303.17564>
2. **FinGPT: Open-Source Financial Large Language Models** — Yang et al.,
   2023 (v2 Nov 2025). <https://arxiv.org/abs/2306.06031>
3. **PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark
   for Finance** — Xie et al., 2023. <https://arxiv.org/abs/2306.05443>
4. **InvestLM: A Large Language Model for Investment using Financial Domain
   Instruction Tuning** — Yang, Tang, Tam, 2023.
   <https://arxiv.org/abs/2309.13064>
5. **FinPT: Financial Risk Prediction with Profile Tuning on Pretrained
   Foundation Models** — Yin et al., 2023. <https://arxiv.org/abs/2308.00065>
6. **FinBen: A Holistic Financial Benchmark for Large Language Models** —
   Xie et al., 2024. <https://arxiv.org/abs/2402.12659>
7. **FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory
   and Character Design** — Yu et al., 2023.
   <https://arxiv.org/abs/2311.13743>
8. **FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal
   Reinforcement for Enhanced Financial Decision Making** — Yu et al., 2024
   (NeurIPS 2024). <https://arxiv.org/abs/2407.06567>
9. **TradingAgents: Multi-Agents LLM Financial Trading Framework** — Xiao et
   al., 2024 (v7 Jun 2025). <https://arxiv.org/abs/2412.20138>
10. **FinRobot: An Open-Source AI Agent Platform for Financial Applications
    using Large Language Models** — Yang et al., 2024.
    <https://arxiv.org/abs/2405.14767>
11. **Character-LLM: A Trainable Agent for Role-Playing** — Shao et al., 2023
    (EMNLP 2023). <https://arxiv.org/abs/2310.10158>
12. **Are LLMs Capable of Data-based Statistical and Causal Reasoning?
    Benchmarking Advanced Quantitative Reasoning with Data (QRData)** — Liu
    et al., 2024 (ACL 2024). <https://arxiv.org/abs/2402.17644>
13. **A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling,
    Search Algorithms, and Relevant Frameworks** — Li, 2025 (TMLR).
    <https://arxiv.org/abs/2501.10069>
14. **Plutus: Benchmarking Large Language Models in Low-Resource Greek
    Finance** — Peng et al., 2025. <https://arxiv.org/abs/2502.18772>

## Appendix B — Primary data sources for persona corpus

- **Warren Buffett**: Berkshire Hathaway shareholder letters 1977-2024 —
  <https://www.berkshirehathaway.com/letters/letters.html>
- **Cathie Wood / ARK**: Big Ideas annual reports 2017-2026 + research
  archive — <https://www.ark-invest.com/big-ideas-2026>
- **Howard Marks**: Oaktree Memos (1990-present) —
  <https://www.oaktreecapital.com/insights/memos>
- **Benjamin Graham**: *Security Analysis* (1934, public domain via Internet
  Archive); *The Intelligent Investor* (1949, partially public domain)
- **Peter Lynch**: *One Up On Wall Street* (1989); *Beating the Street* (1993)
- **Seth Klarman**: *Margin of Safety* (1991, out of print)

## Appendix C — Cross-reference to existing FinAI codebase

What's already built that supports this plan:

- `src/mcp/_sec_edgar.py` — US filings ingest, ready for the
  SEC corpus chunk
- `src/mcp/_indian_filings.py` — BSE + NSE + Screener.in scraper;
  the foundation of the Indian corpus pipeline
- `src/mcp/_claims.py` — claim extraction + comparison; reusable
  in the alerts summarizer
- `src/core/flows/planner_pipeline.py` — planner-first pipeline;
  generates DAG of ScopedAgent steps per query; replan loop with
  max_replans=2; replaces old deepagents-based flow
- `src/mcp/_research.py` — search backend chain (Tavily → DDG →
  fixture) with retrieval post-processing; Tavily client singleton
- `src/mcp/_retrieval.py` — retrieval post-processing pipeline
  (semantic re-ranking via all-MiniLM-L6-v2, dedup, freshness
  filtering, date-window filtering); wired into all 4 search
  functions; graceful no-op when sentence-transformers absent
- `src/core/joiner.py` — rule-based joiner (v0): decides
  finish/replan/abort after step execution; replans on synth
  failure
- `src/core/executor.py` — parallel DAG executor; runs
  independent steps concurrently
- `src/core/dispatcher.py` — intent routing; `_FAST_PATH` for
  smalltalk/meta_help, all other intents → planner pipeline
- `src/core/router.py` — intent classifier
- `src/core/resilient_stream.py` — retry + cache fallback; production-grade
  resilience pattern already in place
- `src/core/agents/_factories.py` — 7 agent factories using
  `_CyclingChatOpenAI` with automatic API key cycling
- `src/personas/base.py` — persona system with key pool cycling;
  panel agents use pinned slots (no cycling) for concurrent
  streaming
- `src/core/types.py` — Plan, PlanStep, Scratchpad,
  ExecutionState, JoinDecision, JoinAction, UnmetDependency
- `data/response_cache/` — disk-backed response cache; extend for per-user
  alert deduplication

### Shipped features not in original plan

- **`run_python` synthetic tool** — any agent can execute verified
  Python code via `exec()` in a local namespace; suppressed from
  streaming output
- **Retrieval post-processing pipeline** — semantic re-ranking,
  dedup (cosine θ=0.90), freshness filtering (min_year gate),
  date-window filtering (closes DDG `enforced=False` gap);
  24 unit tests in `tests/test_retrieval.py`
- **API key cycling** — `_CyclingChatOpenAI` transparently rotates
  through key pool on rate-limit; all 7 factories use `cycle_keys=True`
- **Parallel step execution** — independent DAG steps run
  concurrently; events collected per-step then yielded in step-id
  order
- **Replan loop** — joiner decides replan on synth failure; pipeline
  retries with same dependency graph; max 2 replans then abort
- **Tavily search integration** — agent-optimized search with
  pre-summarized results and `answer` field; singleton client
  pattern with negative caching
- **FinBen baseline** — 71.1% overall (professional_accounting 26.7%,
  econometrics 60%); establishes pre-fine-tuning benchmark

### Known infrastructure gaps (demo → product)

| Gap | Current state | Required for product |
|---|---|---|
| **API auth** | None — open endpoint | Clerk/Supabase auth + JWT |
| **Request timeout** | No per-step wall-clock timeout | Stuck agent blocks entire pipeline |
| **Rate limiting** | None | Per-user + global rate limits |
| **CORS** | `allow_origins=["*"]` | Lock to production domains |
| **Cache eviction** | Disk cache never evicts | TTL + LRU eviction |
| **CI pipeline** | None | FinBen eval harness as CI gate |
| **Python version** | Dockerfile=3.11, pyproject=3.12, conda=3.13 | Pin to 3.12 everywhere |
