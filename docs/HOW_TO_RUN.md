# How to Run the FinAI Multi-Agent Financial Advisor

## Prerequisites

- **Docker** and **Docker Compose** installed
- **NVIDIA_API_KEY** - Required for LLM operations
- **Minimum 4GB RAM** available for Docker
- **Ports 3080 and 8000** should be free

---

## Quick Start (5 Steps)

### Step 1: Configure Environment Variables

Ensure you have the NVIDIA API key set in both locations:

```bash
# Check FinAI .env file
cat .env

# Should contain:
# NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxx
```

Also add the key to LibreChat's .env:

```bash
# Add to LibreChat/.env
echo "NVIDIA_API_KEY=your_key_here" >> LibreChat/.env
```

### Step 2: Start LibreChat

```bash
cd LibreChat
docker compose up -d
```

Wait for all containers to be healthy (2-3 minutes). Check status:

```bash
docker compose ps
```

### Step 3: Start FinAI Service

```bash
cd LibreChat
docker compose up -d
```

This automatically reads `docker-compose.override.yml` which includes FinAI.

### Step 4: Access LibreChat Web Interface

1. Open browser: **http://localhost:3080**
2. Create an account (email/password)
3. Login to the interface

### Step 5: Use FinAI Agent

1. Click **endpoint selector** (top-left dropdown menu)
2. Select **"FinAI"** from the list
3. Choose model **"finai-advisor"**
4. Start chatting with financial queries

---

## Detailed Architecture

```
┌─────────────────────────────────────────────┐
│           User Browser                       │
│         http://localhost:3080               │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │   LibreChat Container    │
    │   (Port 3080)            │
    └────────────┬────────────┘
                 │
    ┌────────────┴────────────┐
    │    FinAI API Container   │
    │    (Port 8000)           │
    │                          │
    │  ┌────────────────────┐  │
    │  │   Orchestrator     │  │
    │  └─────────┬──────────┘  │
    │            │              │
    │  ┌─────────┴──────────┐  │
    │  │  6 Agents:         │  │
    │  │  - upstox          │  │
    │  │  - deep_web_research│  │
    │  │  - us_stock        │  │
    │  │  - indian_stock    │  │
    │  │  - digital_twin    │  │
    │  │  - general_advisor │  │
    │  └────────────────────┘  │
    └──────────────────────────┘
```

---

## Complete Command Reference

### Starting Services

```bash
# Start all services (LibreChat + FinAI)
cd LibreChat && docker compose up -d

# Start only LibreChat (without FinAI)
cd LibreChat && docker compose up -d --no-deps api

# Start only FinAI
cd LibreChat && docker compose up -d finai
```

### Stopping Services

```bash
# Stop all services
cd LibreChat && docker compose down

# Stop only FinAI
cd LibreChat && docker compose stop finai

# Stop and remove all containers + volumes
cd LibreChat && docker compose down -v
```

### Rebuilding Services

```bash
# Rebuild FinAI after code changes
cd LibreChat && docker compose down finai
cd LibreChat && docker compose up -d --build finai

# Force rebuild without cache
docker compose build --no-cache finai
docker compose up -d finai
```

### Viewing Logs

```bash
# View all logs
cd LibreChat && docker compose logs

# View FinAI logs (real-time)
docker logs finai-api -f

# View last 100 lines of FinAI logs
docker logs finai-api --tail 100

# View LibreChat logs
docker logs LibreChat -f

# View specific service logs
docker logs chat-mongodb
docker logs chat-meilisearch
```

### Checking Status

```bash
# View all running containers
docker ps

# View container status
cd LibreChat && docker compose ps

# Check container health
docker inspect finai-api | grep -A 5 "Health"

# Check network connectivity
docker exec LibreChat ping finai-api -c 3
```

---

## Testing FinAI API Directly

### Health Check

```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","service":"finai","version":"1.0.0",...}
```

### List Available Models

```bash
curl http://localhost:8000/v1/models
# Expected: {"object":"list","data":[{"id":"finai-advisor",...}]}
```

### Test Chat Completion (Non-streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "finai-advisor",
    "messages": [{"role": "user", "content": "Analyze my portfolio"}],
    "stream": false
  }'
```

### Test Chat Completion (Streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "finai-advisor",
    "messages": [{"role": "user", "content": "What is compound interest?"}],
    "stream": true
  }'
```

### Test Query Endpoint

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze my portfolio",
    "profile": {"user_id": "test123", "name": "John", "age": 30, "risk_profile": "moderate"}
  }'
```

---

## Configuration Files

### Key Configuration Files

| File | Purpose | Location |
|------|---------|----------|
| `librechat.yaml` | LibreChat endpoint config | `LibreChat/librechat.yaml` |
| `docker-compose.override.yml` | FinAI service definition | `LibreChat/docker-compose.override.yml` |
| `Dockerfile` | FinAI container build | `./Dockerfile` |
| `.env` | API keys and secrets | `./.env` and `LibreChat/.env` |
| `requirements.prod.txt` | Python dependencies | `./requirements.prod.txt` |

### librechat.yaml Configuration

```yaml
endpoints:
  custom:
    - name: "FinAI"
      apiKey: "finai-production"
      baseURL: "http://finai-api:8000/v1"
      models:
        default:
          - "finai-advisor"
      fetch: false
      titleConvo: true
      titleModel: "current_model"
      modelDisplayLabel: "FinAI Advisor"
      dropParams:
        - "stop"
```

---

## Troubleshooting Guide

### Issue 1: FinAI not appearing in endpoint dropdown

**Symptoms:** Cannot see "FinAI" in the LibreChat endpoint selector.

**Solution:**
```bash
# Check if configuration is valid
docker logs LibreChat | grep -i "config\|finai\|error"

# Restart LibreChat to reload config
docker restart LibreChat

# Wait 30 seconds and refresh browser
```

### Issue 2: Empty responses from FinAI

**Symptoms:** "Received empty response from chat model call" error.

**Solution:**
```bash
# Check FinAI logs for errors
docker logs finai-api --tail 50

# Verify FinAI is healthy
curl http://localhost:8000/health

# Test direct API call
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"finai-advisor","messages":[{"role":"user","content":"test"}]}'

# Rebuild FinAI if needed
cd LibreChat && docker compose up -d --build finai
```

### Issue 3: Container name conflict

**Symptoms:** "The container name '/finai-api' is already in use"

**Solution:**
```bash
# Remove old container
docker rm -f finai-api

# Start fresh
cd LibreChat && docker compose up -d finai
```

### Issue 4: NVIDIA_API_KEY not set

**Symptoms:** Warnings about NVIDIA_API_KEY being blank.

**Solution:**
```bash
# Add to both .env files
echo "NVIDIA_API_KEY=your_actual_key" > .env
echo "NVIDIA_API_KEY=your_actual_key" >> LibreChat/.env

# Restart services
cd LibreChat && docker compose restart
```

### Issue 5: YAML configuration errors

**Symptoms:** "Config file YAML format is invalid" in LibreChat logs.

**Solution:**
```bash
# Validate YAML syntax
cat LibreChat/librechat.yaml

# Check indentation (must be 2 spaces)
# Ensure all lists use proper YAML format
```

### Issue 6: Containers not starting

**Symptoms:** Containers exit immediately or show "unhealthy".

**Solution:**
```bash
# Check all container statuses
docker ps -a

# View container exit logs
docker logs finai-api

# Check disk space
df -h

# Prune Docker system
docker system prune -af

# Rebuild from scratch
cd LibreChat && docker compose down -v
cd LibreChat && docker compose up -d --build
```

---

## Port Reference

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| LibreChat UI | 3080 | HTTP | Web interface |
| FinAI API | 8000 | HTTP | REST API endpoint |
| MongoDB | 27017 | TCP | Database |
| Meilisearch | 7700 | HTTP | Search engine |
| RAG API | 8000 | HTTP | Retrieval API |
| VectorDB | 5432 | TCP | PostgreSQL with vectors |

---

## Common Use Cases

### Example Financial Queries

Try these queries in LibreChat with FinAI endpoint:

```
- "Analyze my portfolio and suggest improvements"
- "What is the difference between mutual funds and ETFs?"
- "How does compound interest work?"
- "Explain dollar-cost averaging strategy"
- "What are the risks of investing in stocks?"
- "How to diversify my investment portfolio?"
- "What is a bear market and should I be worried?"
```

### Running FinAI Locally (Without Docker)

```bash
# Install dependencies
pip install -r requirements.prod.txt

# Set environment
export NVIDIA_API_KEY=your_key_here

# Run the API server
uvicorn src.app:app --host 0.0.0.0 --port 8000

# In another terminal, test
curl http://localhost:8000/health
```

---

## Maintenance Tasks

### Updating FinAI Code

```bash
# 1. Make your code changes
# 2. Rebuild and restart
cd LibreChat && docker compose down finai
cd LibreChat && docker compose up -d --build finai

# 3. Verify it's working
docker logs finai-api --tail 20
```

### Backing Up Data

```bash
# Backup MongoDB
docker exec chat-mongodb mongodump --out /backup

# Copy backup to host
docker cp chat-mongodb:/backup ./mongodb-backup
```

### Cleaning Up

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Full cleanup (WARNING: removes everything)
docker system prune -a --volumes
```

---

## Performance Tuning

### Adjusting Resource Limits

Edit `LibreChat/docker-compose.override.yml`:

```yaml
finai:
  deploy:
    resources:
      limits:
        cpus: '4.0'      # Increase CPU limit
        memory: 4G       # Increase memory limit
```

### Monitoring Resource Usage

```bash
# Container stats
docker stats finai-api

# Container resource usage
docker inspect finai-api | grep -A 10 "Memory"
```

---

## Security Considerations

### Production Checklist

- [ ] Change default JWT secrets in LibreChat/.env
- [ ] Use strong CREDS_KEY and CREDS_IV
- [ ] Restrict CORS origins in src/app.py
- [ ] Add authentication to FinAI API
- [ ] Enable HTTPS/TLS
- [ ] Set up rate limiting
- [ ] Regular security updates

### Generating Secret Keys

```bash
# Use LibreChat's generator
# Visit: https://www.librechat.ai/toolkit/creds_generator

# Or generate manually
openssl rand -hex 32
```

---

## Support Resources

- **FinAI Documentation**: `README.md`, `SETUP_COMPLETE.md`
- **LibreChat Docs**: https://www.librechat.ai/docs
- **LibreChat Config Guide**: https://www.librechat.ai/docs/configuration/librechat_yaml
- **Docker Compose Docs**: https://docs.docker.com/compose/

---

## Quick Reference Card

```bash
# Start everything
cd LibreChat && docker compose up -d

# Check status
docker ps

# View logs
docker logs finai-api -f

# Test API
curl http://localhost:8000/health

# Restart
cd LibreChat && docker compose restart

# Stop everything
cd LibreChat && docker compose down

# Rebuild
cd LibreChat && docker compose up -d --build
```

---

**Last Updated**: April 12, 2026  
**Version**: 2.0.0  
**Status**: Production Ready
