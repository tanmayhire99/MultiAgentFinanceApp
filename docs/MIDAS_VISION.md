# MIDAS — Combined Plan, Roadmap & Progress

The three MIDAS projects compose into **one India-native financial-intelligence
platform** spanning **data → reasoning → systematic execution**, with FinAI as
the hub. Per the competitive analysis in `FROM_DEMO_TO_PRODUCT.md`, nobody has
shipped this combination for the Indian market. _Last updated: 2026-06-30._

```
 equity-pipeline ──data + technicals──►  FinAI  ◄──read-only backtests── automated-trading
   (the backbone)        (brain + UI + memory + alerts + live feed)        (the quant engine, paused)
                                   ▲ broker portfolio sync (next) ▼
```

Per-project plans (each has its own Completed / Remaining sections):
- FinAI — `docs/ROADMAP.md` (this repo)
- equity-pipeline — its `docs/ROADMAP.md`
- automated-trading — its `docs/ROADMAP.md`

## The three roles
- **equity-pipeline = data backbone** — clean, exchange-sourced NSE data + a
  quality monitor + derived technicals; widening universe + intraday ahead.
- **FinAI = reasoning + UX** — multi-agent research, investor-panel debate,
  persistent memory, event-driven alerts. The user-facing product.
- **automated-trading = quant engine** — rigorously validated systematic
  strategies; backtesting-as-a-service today, paper/live (gated) for power users.
  **Currently paused** with an honest "no edge yet" verdict.

## Integration roadmap
| Phase | What | Status |
|---|---|---|
| **I** | warehouse → FinAI (Indian data); quant → FinAI (read-only backtesting) via MCP | ✅ **done** |
| **I.5** | warehouse **technicals** → FinAI `indian_stock` tool; **live day-move feed** → FinAI alerts | ✅ **done this session** |
| **II** | Identity + **broker portfolio sync** (read-only) → memory + alerts run on *real* holdings; holdings feed quant risk context | ☐ next — highest leverage |
| **III** | **Close the loop:** FinAI panel verdicts become backtestable via automated-trading ("if you'd followed the panel on X in 2023…") | ☐ planned (quant track paused) |
| **IV** | **Data flywheel:** widen equity-pipeline (universe + intraday) → richer FinAI analysis + alerts drive engagement → feedback in | ◑ technicals shipped; universe/intraday ahead |
| **V** | **Moat:** Ganga-LLM (India-domain) + persona LoRAs, FinBen-benchmarked | ◑ persona LoRA done (team); Ganga-LLM ahead |

## Completed this session (cross-project)
**FinAI** — persistent memory (`bb02557`), verify-numbers correction + badge
(`9e8c2ac`), event-driven alerts foundation (`0f137fc`), warehouse technicals MCP
tool (`1c65723`), authed `/alerts` API (`68ba690`), live day-move feed
(`93441b0`). Tests 377 → **433**.

**equity-pipeline** — reusable data-quality monitor + alerting + standalone
runner + scheduler hook (`bc0b984`, green on the live warehouse), derived
technical-analytics layer (`491c4f2`). Tests 27 → **51**.

**automated-trading** — durable experiment persistence (`053064b`), research loop
+ hypothesis safety boundary (`1524cdd`), real 30-config search + honest campaign
(0/30 pass, gate untouched) (`03a0dfa`). Tests 58 → **76**.

**Net:** the MIDAS data→reasoning loop is now wired both ways — equity-pipeline
data *and* technicals flow into FinAI, FinAI personalises (memory) and proactively
alerts (live feed), and the quant engine is rigorously honest about edge.

## Remaining (combined)
- ☐ **Broker portfolio sync** (Phase II) — the single highest-leverage step:
  turns memory + alerts from the demo fixture into a real product.
- ☐ **Alerts UX** — unread badge in LibreChat + a scheduled background scan.
- ☐ **Panel-verdict backtesting** (Phase III) — needs the quant track resumed.
- ☐ **equity-pipeline**: productionized scheduler keeping Supabase auto-fresh;
  widen universe; corporate actions; intraday.
- ☐ **automated-trading** (paused): a genuinely different hypothesis for a real
  edge; wire Claude; paper trading; SEBI compliance; then (gated) live.
- ☐ **Ganga-LLM moat** (Phase V) — India-domain pretraining, FinBen-gated.

## The "big" product
One prosumer app where an Indian investor gets: (a) exchange-grade data +
technicals, (b) a multi-agent + investor-panel research experience **with memory
and proactive alerts**, and (c) optional **rigorously-validated systematic
strategies** — execution always behind a deterministic risk gate.

## Progress scorecard (2026-06)
| Track | Where we are | ~% to its vision |
|---|---|---|
| **FinAI** | hardened demo + **memory, alerts (+ live feed), correction, technicals shipped**; persona LoRA done | **~50%** (was ~30%) |
| **equity-pipeline** | v1 warehouse + secured analytics + **quality monitoring + technicals shipped** | **~55%** (was ~45%) |
| **automated-trading** | engine + gate + risk + **persistence + research loop + honest 30-config search**; no validated edge; paused | **Phase 1 ~85%; toward live ~25–30%** |
| **MIDAS combined** | Phases I + I.5 shipped (data + technicals + alerts wired both ways); II–V ahead | **~30%** (was ~20%) |

**Headline:** the foundation is built across all three, and this session wired the
platform into one working system — data + technicals → reasoning → personalised,
proactive UX, with a quant engine that refuses to bless an unproven edge. What
remains is **real-holding broker sync** (makes it a product), resuming the quant
track behind a *different* hypothesis, and the **Ganga-LLM moat**.
