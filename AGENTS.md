# AGENTS.md — working notes for this repo

FinAI: a planner-first multi-agent finance assistant (FastAPI backend + LibreChat
frontend). This file captures the conventions, build/test commands, and gotchas
so future sessions don't have to rediscover them.

## Repository layout

- `src/` — application code
  - `src/core/` — planner, executor, pipeline, joiner, panel/debate, types
  - `src/core/agents/` — `ScopedAgent` runtime + **one file per agent**
    (`research_agent.py`, `filings_agent.py`, `us_stock_agent.py`,
    `indian_stock_agent.py`, `portfolio_agent.py`, `claim_agent.py`,
    `synthesizer.py`, `panel_agent.py`), with `factory_dispatch.py`
    (`_FACTORY_MAP` + `build_scoped_agent_for_step`) and `_model.py`
    (the single `build_chat_model` seam — patch this in tests)
  - `src/mcp/` — MCP tool servers; `src/personas/` — debate personas
- `tests/` — the real test suite
- `docs/` — architecture + migration narrative; `docs/FROM_DEMO_TO_PRODUCT.md`
  is the productization roadmap
- `LibreChat/` — vendored frontend (ignored by git tooling here)

## Dev environment

The committed `.venv/` is **Linux-only** (`[tool.uv] environments =
['sys_platform == "linux"']`), so it will not run on macOS. On a Mac, create a
local env (ignored by git via `.venv*/`):

```bash
uv venv .venv-mac --python 3.12
uv pip install --python .venv-mac/bin/python \
  pytest pyjwt fastapi langchain-openai langgraph langchain-mcp-adapters \
  mcp fastmcp openai requests tavily-python ddgs python-dotenv dotenv \
  uvicorn sentence-transformers
```

`sentence-transformers` (pulls `torch`) is only needed for the retrieval
re-ranker (`src/mcp/_retrieval.py`) and `tests/test_retrieval.py`; the rest of
the suite runs without it.

`.venv-mac` is the macOS dev convention; on Linux use `uv venv .venv && uv sync`
(the committed env is Linux-targeted). **Deploying or migrating to Ubuntu? See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)** for the full end-to-end runbook, and
[`deploy/librechat/`](deploy/librechat/) for the LibreChat config + upgrade path.

## Tests

```bash
.venv-mac/bin/python -m pytest          # whole suite
.venv-mac/bin/python -m pytest -q tests/test_factories.py   # one file
```

`pyproject.toml` pins `testpaths = ["tests"]` and `norecursedirs` so plain
`pytest` ignores `docs/migration/snapshots/**` and the embedded
`evals/finlm_eval/` repo (both contain stray test files that otherwise break
collection).

## Evaluation & CI

- **CI** (`.github/workflows/ci.yml`): runs `pytest` on every push to `main`
  and every PR. Fast lane — light deps, no secrets; retrieval tests auto-skip
  when `sentence-transformers` is absent.
- **FinBen gate** (`.github/workflows/eval-finben.yml`): nightly + manual.
  Runs the MMLU-Finance scorecard via NIM and fails on a regression vs
  `evals/results/baseline.json`. Needs the `NVIDIA_API_KEY` repo secret
  (skips cleanly without it).
- The regression-gate logic + a deterministic numeric-accuracy eval live in
  `evals/scorecard.py` and `tests/test_eval_gate.py`. See `evals/README.md`.
- To move the baseline intentionally (e.g. model upgrade): run
  `python evals/scorecard.py run` and commit the new `baseline.json` separately.

## MIDAS cross-project integrations (opt-in)

FinAI is the hub; sibling MIDAS projects plug in via the MCP tool layer. Both
are opt-in (env-gated) so FinAI runs standalone by default.

- **equity-pipeline (data)** — set `WAREHOUSE_DATABASE_URL` and the
  `indian_stock` worker reads exchange-sourced NSE EOD data (quote / price
  history / top movers / sector performance) from that Postgres warehouse
  (`src/mcp/_warehouse.py`); falls back to yfinance when unset.
- **automated-trading (quant)** — set `QUANT_MCP_PYTHON` + `QUANT_MCP_CWD` and
  FinAI spawns that repo's READ-ONLY `quant_mcp.py` server **in its own Python
  3.14 venv** (cross-runtime, via a per-server `command` in
  `src/config/mcp_servers.py`), exposing a `quant_agent` for listing/backtesting
  NIFTY F&O strategy templates. **No execution surface is ever exposed** — the
  hard boundary in `automated-trading/docs/CONTEXT.md` is preserved.
  When unset, the quant agent + server are absent (registry stays at 37 tools).

## Git identity (important)

Commits here MUST be attributed to **`tanmay.hire99@gmail.com`** (GitHub:
`tanmayhire99`). The machine's **global** git email is a work address
(`...@wdc.com`); this repo carries a **local override** so commits come out
correct — verify before committing:

```bash
git config user.email   # expect: tanmay.hire99@gmail.com
```

If it ever shows the work email, set the local override:
`git config user.email "tanmay.hire99@gmail.com"`.

## Workflow / avoiding divergence

- Work in **one clone** (this repo: `Multi-Agent-Finance-App/MultiAgentFinanceApp`).
  Two divergent copies previously caused weeks of duplicated work. Use
  `git worktree add` for parallel branches instead of a second clone.
- Push regularly so work isn't laptop-only. `main` is the canonical line.
- Remote: `git@github.com:tanmayhire99/MultiAgentFinanceApp.git`.
