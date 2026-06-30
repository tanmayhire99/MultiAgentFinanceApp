# FinAI — Roadmap & Progress

_Part of the MIDAS platform. Combined vision + cross-project scorecard:_
[`docs/MIDAS_VISION.md`](MIDAS_VISION.md). _Productization research memo:_
[`docs/FROM_DEMO_TO_PRODUCT.md`](FROM_DEMO_TO_PRODUCT.md). _Deploy/migrate:_
[`docs/DEPLOYMENT.md`](DEPLOYMENT.md). _Last updated: 2026-06-30._

## What FinAI is
A planner-first multi-agent finance assistant (NVIDIA NIM `gpt-oss-120b`):
intent router → LLM planner builds a DAG of scoped agents (research / US-stock /
indian-stock / portfolio / filings) → executor runs them with debate → investor
**panel** (Buffett / Wood / Graham) → synthesizer. MCP tool layer backs the
agents; `verify_numbers` guards numeric claims; LibreChat is the UI.

## Current state (2026-06)
- **433 tests**, CI green, gitleaks secret-scanning (CI + pre-commit).
- Production hardening done: JWT auth, per-user rate limiting, per-step timeouts,
  bounded response cache, CORS lock, fail-fast config, request-ID logging, deep
  `/health`.
- Eval gate (offline numeric-accuracy + FinBen scorecard regression check).
- Persistent **per-user memory**, **event-driven alerts** (+ live day-move feed),
  numeric **correction surfacing**, and a warehouse-backed **technicals** tool —
  all shipped (see Completed). 38 MCP tools across 8 agents.
- Cross-project integrations live: **equity-pipeline** (Indian EOD data +
  technicals behind `indian_stock`) and **automated-trading** (read-only quant
  backtesting), both opt-in/env-gated.

## Phased roadmap
| Phase | Work | Status |
|---|---|---|
| **A** | Persistent **memory** (per-user profile + topics) + risk profile | ✅ memory shipped; ☐ risk-profile onboarding UI |
| **B** | **Event-driven alerts** on holdings (concentration + live price moves) | ✅ store + rules + API + live feed shipped; ☐ UI badge + scheduled scans |
| **C** | Numeric **correction + badge**; **persona LoRA** fine-tuning | ✅ both done (LoRA handled by the team) |
| **D** | **Ganga-LLM** — India-domain continued-pretrain + persona LoRAs, FinBen-benchmarked | ☐ the moat — not started |

## Completed progress (this session)
- ✅ **Persistent memory** (`src/core/memory.py`) — SQLite profile (risk /
  horizon / goals) + topics, keyed by user_id; injected into the planner and
  **every** agent prompt; observed each turn; auth-gated. `bb02557`
- ✅ **verify_numbers correction + badge** — flagged numbers now show the source
  figure inline ("$2.5B ✗ (source data: 1.23B)") + a counts badge. `9e8c2ac`
- ✅ **Event-driven alerts foundation** (`src/core/alerts.py`) — durable store,
  pure rules (concentration, price-move), dedup, scan CLI. `0f137fc`
- ✅ **`indian_stock__get_technicals` MCP tool** — SMA 20/50/200, trailing
  returns, volatility, max drawdown from the equity-pipeline warehouse
  (INR→USD); registry 37→38 tools. `1c65723`
- ✅ **Authed `/alerts` API** — `GET /alerts` (+ unread_count badge),
  `POST /alerts/scan`, `POST /alerts/mark-read`, same JWT gate as chat. `68ba690`
- ✅ **Live day-move feed** — `live_quote_change` routes NSE→warehouse /
  US→live quote so `price_move` alerts fire for real. `93441b0`
- ✅ **Persona LoRA fine-tuning** — completed by the team (Character-LLM method).

## Remaining progress
- ☐ **Broker / portfolio sync** (read-only Kite/Upstox) so memory + alerts run on
  *real* holdings instead of the fixture — the highest-leverage next step.
- ☐ **Risk-profile onboarding** UX (memory extracts signals; needs an explicit
  5-question flow + surfacing).
- ☐ **Alerts UI**: unread badge in LibreChat + a scheduled background scan job.
- ☐ **Phase D — Ganga-LLM**: India-domain continued pretraining, FinBen-gated
  across releases (the moat; large, expensive).
- ☐ Nightly FinBen eval in CI (offline gate exists; wire the scheduled run).

## Progress
Demo + architecture + production hardening + **the core product features
(memory, alerts, correction, technicals) now shipped**, and persona LoRA done.
**~50% to the product vision** (up from ~30%). What remains is real-holding
**broker sync**, the **alerts/onboarding UX**, and the **Ganga-LLM moat**.
Recommended next: **broker portfolio sync** (turns memory + alerts from demo into
product), then the moat.
