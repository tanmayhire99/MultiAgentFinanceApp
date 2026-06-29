# LibreChat configuration for FinAI

The chat UI is **upstream LibreChat, consumed as a prebuilt Docker image** — we
do **not** fork or build it from source. Everything FinAI-specific lives in the
two small files in this directory, which is all that needs to be version
controlled or migrated:

| File | What it does |
|---|---|
| `librechat.yaml` | Registers the **FinAI custom endpoint** (`http://finai-api:8000/v1`), enables agents, and labels the model "FinAI Advisor". |
| `docker-compose.override.yml` | Pins the LibreChat image to **`ghcr.io/danny-avila/librechat:v0.8.4`**, adds the `finai` service (built from this repo's `Dockerfile`) to LibreChat's network, mounts `librechat.yaml`, and passes env through. |

> Do **not** commit the local `LibreChat/` clone (it's ~2.9 GB of upstream
> source + `node_modules` and is gitignored). It is not a runtime dependency —
> the image is pulled from the registry.

## Stand up LibreChat on a fresh machine (Ubuntu or any Docker host)

```bash
# 1. Get upstream LibreChat (pinned to the version we built against)
git clone https://github.com/danny-avila/LibreChat.git
cd LibreChat && git checkout v0.8.4

# 2. Drop in the FinAI customizations from this repo
cp /path/to/MultiAgentFinanceApp/deploy/librechat/librechat.yaml .
cp /path/to/MultiAgentFinanceApp/deploy/librechat/docker-compose.override.yml .

# 3. Configure env (LibreChat essentials + FinAI passthrough)
cp .env.example .env
#   set: MONGO_URI (bundled mongo is fine), and generate secrets:
#     CREDS_KEY (32-byte hex), CREDS_IV (16-byte hex),
#     JWT_SECRET, JWT_REFRESH_SECRET
#   add the FinAI passthrough vars the override reads:
#     NVIDIA_API_KEY=...           # (and NVIDIA_API_KEY_1..4 if used)
#     TAVILY_API_KEY=...           # optional
#     LIBRECHAT_JWT=...            # token sent as X-LibreChat-Token to FinAI auth

# 4. Bring it up (pulls the LibreChat image, builds the finai service)
docker compose up -d
#   UI:   http://localhost:3080
#   FinAI: http://localhost:8000/health
```

The override builds `finai` from `../Dockerfile`, so the LibreChat clone is
expected to sit **next to** this repo, or adjust the `build.context` path.

## Upgrading LibreChat

You are **not forked**, so upgrades are cheap:

1. Bump the image tag in `docker-compose.override.yml`
   (`ghcr.io/danny-avila/librechat:v0.8.4` → the new stable tag) and
   `git checkout <new-tag>` in the LibreChat clone (for its base compose).
2. Check `librechat.yaml`'s top `version:` field against the new release's
   `librechat.example.yaml` — the config schema occasionally adds fields; ours
   uses only stable keys (custom endpoint, agents, interface).
3. `docker compose pull && docker compose up -d`, then smoke-test a chat.

Keep this directory in sync if you change the live config:
`cp ../../LibreChat/{librechat.yaml,docker-compose.override.yml} .`
