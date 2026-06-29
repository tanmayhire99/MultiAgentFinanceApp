# MIDAS — Deployment & Migration Runbook (Ubuntu)

End-to-end steps to stand up the whole system on a fresh **Ubuntu** box. The
code is OS-portable (audited: no hardcoded paths, no macOS-only calls); the only
real work is installing the right runtimes and wiring env. Migration to a new
machine is therefore a clean, repeatable process.

## Components & ports

| Component | Repo | Runtime | Port |
|---|---|---|---|
| FinAI hub (FastAPI) | `MultiAgentFinanceApp` | **Python 3.12** | 8000 |
| equity-pipeline (warehouse ETL + consumer API) | `Equity-Analytics-Warehouse` | **Python 3.14** | — |
| automated-trading (quant backtester) | `AI-Assisted-Trading` | **Python 3.14** | — (MCP stdio) |
| Shared warehouse | Postgres 16 (Docker) or Supabase | — | 5433 |
| LibreChat UI | upstream Docker image `v0.8.4` | container | 3080 |

## 0. Prerequisites (Ubuntu 22.04/24.04)

```bash
sudo apt-get update && sudo apt-get install -y git curl build-essential
# Docker Engine + Compose plugin (https://docs.docker.com/engine/install/ubuntu/)
# uv — manages BOTH Python 3.12 and 3.14 without system installs:
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12 3.14
# optional but recommended: gitleaks for the pre-commit secret guard
#   (https://github.com/gitleaks/gitleaks/releases) + `pipx install pre-commit`
```

> **Why uv:** equity-pipeline and automated-trading need Python **3.14** (pinned
> to cp314 wheels — do not downgrade), FinAI needs **3.12**. `uv` fetches both;
> no deadsnakes/pyenv juggling. The committed `.venv/` in FinAI is Linux-targeted
> (`[tool.uv] environments = ['sys_platform == "linux"']`) — recreate it with
> `uv sync`, never copy a venv between machines.

## 1. Clone the repos (side by side)

```bash
mkdir -p ~/MIDAS && cd ~/MIDAS
git clone https://github.com/tanmayhire99/MultiAgentFinanceApp.git
git clone https://github.com/tanmayhire99/Equity-Analytics-Warehouse.git equity-pipeline
git clone https://github.com/tanmayhire99/AI-Assisted-Trading.git automated-trading
```

## 2. Shared warehouse (Postgres)

```bash
docker run -d --name equity-pg -p 5433:5432 \
  -e POSTGRES_USER=equity_user -e POSTGRES_PASSWORD=equity_pass -e POSTGRES_DB=equity_db \
  -v equity_pgdata:/var/lib/postgresql/data postgres:16
```
(Or use the top-level `docker-compose.yml`, or point at Supabase via `DB_TARGET=supabase`.)

## 3. equity-pipeline (Python 3.14)

```bash
cd ~/MIDAS/equity-pipeline
uv venv .venv --python 3.14 && uv pip install -r requirements.txt
cp .env.example .env    # set DATABASE_URL=postgresql://equity_user:equity_pass@localhost:5433/equity_db
python setup_db.py                      # schema + dims + analytics views (security_invoker)
BACKFILL_DAYS=400 python run_backfill.py   # populate NIFTY-50 EOD; then sp_refresh_reporting
```

## 4. automated-trading (Python 3.14)

```bash
cd ~/MIDAS/automated-trading
uv venv .venv --python 3.14 && uv pip install -r requirements.txt
python -m data.warehouse.ingest_bhavcopy --last 400   # builds data/warehouse/market.duckdb
python run_backtest.py --strategy straddle            # smoke test
```

## 5. FinAI hub (Python 3.12)

```bash
cd ~/MIDAS/MultiAgentFinanceApp
uv venv .venv --python 3.12 && uv sync       # or: uv pip install -r requirements.prod.txt
cp .env.example .env                          # fill in the keys below
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
curl localhost:8000/health
```

### Required / optional env (FinAI `.env`)

```
NVIDIA_API_KEY=nvapi-...            # REQUIRED (fail-fast at startup if missing)
NVIDIA_API_KEY_1..4=...             # optional: per-persona key pool for concurrency
TAVILY_API_KEY=...                  # optional (DuckDuckGo fallback otherwise)
FINAI_ALLOWED_ORIGINS=https://your-ui-origin   # REQUIRED in prod (CORS lock)
# Cross-project integrations (opt-in):
WAREHOUSE_DATABASE_URL=postgresql://equity_user:equity_pass@localhost:5433/equity_db
QUANT_MCP_PYTHON=/home/<you>/MIDAS/automated-trading/.venv/bin/python
QUANT_MCP_CWD=/home/<you>/MIDAS/automated-trading
```

## 6. LibreChat UI

See [`deploy/librechat/README.md`](../deploy/librechat/README.md) — clone upstream
LibreChat `v0.8.4`, drop in our `librechat.yaml` + `docker-compose.override.yml`,
set its `.env` (CREDS_KEY/CREDS_IV/JWT_SECRET/JWT_REFRESH_SECRET + the FinAI
passthrough vars), then `docker compose up -d`. LibreChat runs from the official
image, so it is fully OS-agnostic.

## 7. One-command (containerized) alternative

From the MIDAS root, the top-level `docker-compose.yml` brings up Postgres + the
FinAI API together; the warehouse still needs the one-time seed (step 3). The
quant integration runs on the host (different Python runtime), not in that
container — set `QUANT_MCP_*` for host runs.

## Migration checklist (highest priority)

- [x] **No code changes needed** — audited: zero hardcoded `/Users/` paths, no
      `platform`/`darwin` branches, no macOS `sed`/`brew` in tracked code.
- [ ] Install Python **3.12 + 3.14** via `uv` (don't downgrade the 3.14 repos).
- [ ] Recreate each `.venv` with `uv` on the target — never copy a venv across machines/OSes.
- [ ] Re-create every `.env` from the committed `.env.example` (real secrets are
      gitignored and never travel with the repo).
- [ ] LibreChat: clone upstream + apply `deploy/librechat/*` (image is OS-agnostic).
- [ ] Seed/backfill the warehouse (or point at Supabase) before first run.
- [ ] `curl :8000/health` shows `status: healthy` and `mcp_tools_loaded > 0`.
