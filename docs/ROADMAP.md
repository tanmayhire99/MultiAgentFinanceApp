# FinAI — Roadmap

_Part of the MIDAS platform. Combined vision + cross-project scorecard:_
[`docs/MIDAS_VISION.md`](MIDAS_VISION.md). _Productization research memo:_
[`docs/FROM_DEMO_TO_PRODUCT.md`](FROM_DEMO_TO_PRODUCT.md). _Deploy/migrate:_
[`docs/DEPLOYMENT.md`](DEPLOYMENT.md).

## What FinAI is
A planner-first multi-agent finance assistant (NVIDIA NIM `gpt-oss-120b`):
intent router → LLM planner builds a DAG of scoped agents (research / US-stock /
indian-stock / portfolio / filings) → executor runs them with debate → investor
**panel** (Buffett / Wood / Graham) → synthesizer. MCP tool layer backs the
agents; `verify_numbers` guards numeric claims; LibreChat is the UI.

## Current state (2026-06)
- **377 tests**, CI green, gitleaks secret-scanning (CI + pre-commit).
- Production hardening **done**: JWT auth, per-user rate limiting, per-step
  timeouts, bounded response cache, CORS lock (`FINAI_ALLOWED_ORIGINS`),
  fail-fast config, request-ID access logging, deep `/health`.
- Eval gate (offline numeric-accuracy + FinBen scorecard regression check).
- Cross-project integrations live: **equity-pipeline** (Indian EOD data behind
  `indian_stock`) and **automated-trading** (read-only quant backtesting), both
  opt-in/env-gated.

## Strengths
Architecture is ahead of most 2024–25 literature (real multi-agent debate,
planner-first replan loop, claim-tracking, retrieval re-ranking). Secure,
tested, deployable today.

## Gaps (vs a real product — see FROM_DEMO_TO_PRODUCT §4)
- No persistent user **memory** (every session is cold).
- No personalized **risk profile** (same advice for everyone).
- `verify_numbers` **flags** but doesn't **auto-correct**; no UI badge.
- Personas are **prompt-only**, not fine-tuned.
- No event-driven **alerts**; no broker/portfolio sync (portfolio is a fixture).

## Phased roadmap
| Phase | Work | Why |
|---|---|---|
| **A** | Persistent **memory** (LangGraph store + vector by user_id) + 5-question risk profile | #1 reason B2C AI feels like a toy is amnesia |
| **B** | **Event-driven alerts** on holdings (scheduled re-research, claim-flip notifications) | the engagement multiplier |
| **C** | Numeric **auto-correct + UI badges**; **persona LoRA** fine-tuning (Character-LLM method) | close the QRData gap; make personas authentic |
| **D** | **Ganga-LLM** — India-domain continued-pretrain + persona LoRAs, FinBen-benchmarked | the moat |

## Progress
Demo + architecture + **production hardening complete (~30% to the product
vision)**. The differentiating product features (A–C) and the moat (D) are
ahead. Recommended next concrete task: **Phase A — persistent memory**.
