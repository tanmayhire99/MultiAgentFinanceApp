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

## Tests

```bash
.venv-mac/bin/python -m pytest          # whole suite
.venv-mac/bin/python -m pytest -q tests/test_factories.py   # one file
```

`pyproject.toml` pins `testpaths = ["tests"]` and `norecursedirs` so plain
`pytest` ignores `docs/migration/snapshots/**` and the embedded
`evals/finlm_eval/` repo (both contain stray test files that otherwise break
collection).

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
