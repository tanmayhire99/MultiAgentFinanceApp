# MIDAS — Combined Vision & Progress

The three MIDAS projects compose into **one India-native financial-intelligence
platform** spanning **data → reasoning → systematic execution**, with FinAI as
the hub. Per the competitive analysis in `FROM_DEMO_TO_PRODUCT.md`, nobody has
shipped this combination for the Indian market.

```
 equity-pipeline ──data──►  FinAI (brain + UI + memory + alerts)  ◄──backtests── automated-trading
   (the backbone)          the product surface the user touches        (the quant engine)
                                   ▲ broker portfolio sync ▼
```

Per-project plans:
- FinAI — `docs/ROADMAP.md` (this repo)
- equity-pipeline — its `docs/ROADMAP.md`
- automated-trading — its `docs/ROADMAP.md`

## The three roles
- **equity-pipeline = data backbone** — clean, exchange-sourced NSE data;
  expanding universe + fundamentals + (later) intraday.
- **FinAI = reasoning + UX** — multi-agent research, investor-panel debate,
  memory, alerts. The user-facing product.
- **automated-trading = quant engine** — rigorously validated systematic
  strategies; backtesting-as-a-service now, paper/live (gated) for power users.

## Integration roadmap
| Phase | What | Status |
|---|---|---|
| **I** | warehouse → FinAI (Indian data); quant → FinAI (read-only backtesting) via MCP | ✅ **done** |
| **II** | Identity + **broker portfolio sync** (read-only) → FinAI memory knows real holdings; holdings feed automated-trading's risk context | planned |
| **III** | **Close the loop:** FinAI panel verdicts become backtestable via automated-trading ("if you'd followed the panel on X in 2023…") | planned |
| **IV** | **Data flywheel:** widen equity-pipeline (universe + fundamentals + intraday) → richer FinAI analysis + better strategies → alerts drive engagement → feedback in | planned |
| **V** | **Moat:** Ganga-LLM (India-domain) + persona LoRAs trained on the accumulated corpus; FinBen-benchmarked | planned |

## The "big" product
One prosumer app where an Indian investor gets: (a) exchange-grade data, (b) a
multi-agent + investor-panel research experience **with memory and alerts**, and
(c) optional **rigorously-validated systematic strategies** — execution always
behind a deterministic risk gate.

## Progress scorecard (2026-06)
| Track | Where we are | ~% to its vision |
|---|---|---|
| **FinAI** | demo + architecture + production hardening done; product features pending | ~30% |
| **equity-pipeline** | v1 warehouse working, secured, dual-target, integrated | ~45% |
| **automated-trading** | engine + validation gate + risk + paper-promotion built; research-loop/persistence/live stubbed; **no validated edge yet** | Phase 1 ~70%; toward live ~25% |
| **MIDAS combined** | Phase I shipped (one wired system); II–V ahead | ~20% |

**Headline:** the hard foundation is built across all three — clean data, a
multiple-testing-aware quant gate, a secure multi-agent app, and the integration
plumbing that makes them one system. What remains: the product surface (memory,
alerts, personalization), the quant payoff (a validated edge — the gate is
currently, correctly, saying "not yet"), and the moat (domain LLM).
