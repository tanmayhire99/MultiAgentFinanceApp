# LibreChat Custom Agent Integration Guide

This guide explains exactly what needs to be configured in LibreChat to integrate a custom agent like FinAI.

---

## Overview

To make your custom agent appear and work in LibreChat, you need to modify 2 key files:

1. **`LibreChat/librechat.yaml`** - Endpoint configuration
2. **`LibreChat/docker-compose.override.yml`** - Docker service definition

---

## File 1: librechat.yaml

### Location
```
LibreChat/librechat.yaml
```

### What This File Does
This file tells LibreChat about your custom endpoint. It defines:
- The name of your endpoint
- The API URL where your agent is running
- Available models
- Display settings

### Complete Configuration

```yaml
# LibreChat Configuration for Custom Agent Integration
# Version: 1.3.5

version: 1.3.5

cache: true

interface:
  endpointsMenu: true
  modelSelect: true
  sidePanel: true
  agents:
    use: true

endpoints:
  # Agents configuration (optional)
  agents:
    recursionLimit: 25
    maxRecursionLimit: 50
    disableBuilder: false
    capabilities:
      - "execute_code"
      - "file_search"
      - "actions"
      - "tools"
      - "artifacts"
      - "context"
      - "web_search"

  # Custom endpoints - THIS IS WHERE YOUR AGENT GOES
  custom:
    - name: "FinAI"                          # Display name in LibreChat UI
      apiKey: "finai-production"             # API key (can be any string for local)
      baseURL: "http://finai-api:8000/v1"    # URL to your agent API
      models:
        default:
          - "finai-advisor"                  # Model name to show in UI
      fetch: false                           # Don't fetch models from endpoint
      titleConvo: true                       # Auto-generate conversation titles
      titleModel: "current_model"            # Use current model for titles
      modelDisplayLabel: "FinAI Advisor"     # Label shown next to messages
      dropParams:
        - "stop"                             # Parameters to exclude from requests
```

### Key Fields Explained

| Field | Description | Example Value |
|-------|-------------|---------------|
| `name` | Display name in endpoint dropdown | `"FinAI"` |
| `apiKey` | API key (use any value for local) | `"finai-production"` |
| `baseURL` | URL to your agent's OpenAI-compatible API | `"http://finai-api:8000/v1"` |
| `models.default` | List of available models | `["finai-advisor"]` |
| `fetch` | Whether to fetch models from endpoint | `false` (use defined models) |
| `modelDisplayLabel` | Label shown next to messages | `"FinAI Advisor"` |

### Important Notes

1. **baseURL Format**: Must end with `/v1` for OpenAI compatibility
2. **Docker Network**: Use container name (`finai-api`) not `localhost`
3. **YAML Indentation**: Use 2 spaces, not tabs
4. **Under `endpoints.custom`**: Must be nested under `endpoints:`

---

## File 2: docker-compose.override.yml

### Location
```
LibreChat/docker-compose.override.yml
```

### What This File Does
This file tells Docker to:
- Build and run your custom agent container
- Connect it to LibreChat's network
- Set environment variables
- Configure health checks

### Complete Configuration

```yaml
# Docker Compose Override for Custom Agent Integration
# This file adds your agent service to LibreChat's Docker network

services:
  # Mount librechat.yaml configuration file
  api:
    volumes:
      - type: bind
        source: ./librechat.yaml
        target: /app/librechat.yaml
    depends_on:
      - finai  # Your agent service name

  # Your Custom Agent Service Definition
  finai:
    build:
      context: ../              # Path to your agent's Dockerfile (parent directory)
      dockerfile: Dockerfile    # Name of your Dockerfile
    container_name: finai-api   # Container name (used in baseURL)
    restart: unless-stopped

    # Connect to LibreChat's network
    networks:
      - default

    # Expose port for direct access (optional, for debugging)
    ports:
      - "8000:8000"

    # Environment variables
    environment:
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}  # Pass API key from .env
      - LOG_LEVEL=INFO
      - PORT=8000
      - HOST=0.0.0.0

    # Health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

    # Resource limits (adjust as needed)
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### Key Fields Explained

| Field | Description | Example Value |
|-------|-------------|---------------|
| `build.context` | Path to agent's code (relative to LibreChat/) | `../` |
| `build.dockerfile` | Name of Dockerfile | `Dockerfile` |
| `container_name` | Docker container name | `finai-api` |
| `ports` | Port mapping | `"8000:8000"` |
| `networks` | Network to join | `default` (LibreChat's network) |

---

## Requirements for Your Custom Agent

Your agent must implement an **OpenAI-compatible API** with these endpoints:

### Required Endpoints

#### 1. Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "your-agent-name",
        "version": "1.0.0"
    }
```

**Purpose**: Used by Docker health checks and monitoring.

#### 2. Models Endpoint

```python
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "your-model-name",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "your-org"
            }
        ]
    }
```

**Purpose**: LibreChat queries this to list available models.

#### 3. Chat Completions Endpoint (Non-streaming)

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Process the request
    response_text = your_agent_process(request.messages)
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ]
    }
```

**Purpose**: Main endpoint for chat interactions.

#### 4. Chat Completions Endpoint (Streaming) - OPTIONAL but RECOMMENDED

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if request.stream:
        async def generate_stream():
            # First chunk with role and content
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Final chunk with finish_reason
            final_chunk = {
                "id": chunk["id"],
                "object": "chat.completion.chunk",
                "created": chunk["created"],
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }]
    }
```

**Purpose**: Enable streaming responses for better UX.

---

## Request/Response Format

### Request Format

LibreChat will send:

```json
{
  "model": "finai-advisor",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"}
  ],
  "user": "user_id_here",
  "stream": true
}
```

### Response Format (Non-streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "finai-advisor",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Response Format (Streaming)

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"finai-advisor","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"finai-advisor","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

## Step-by-Step Integration Checklist

### Step 1: Prepare Your Agent

- [ ] Implement `/health` endpoint
- [ ] Implement `/v1/models` endpoint
- [ ] Implement `/v1/chat/completions` endpoint (with streaming support)
- [ ] Create Dockerfile for your agent
- [ ] Test endpoints with curl

### Step 2: Configure LibreChat

- [ ] Copy `librechat.yaml` to `LibreChat/librechat.yaml`
- [ ] Copy `docker-compose.override.yml` to `LibreChat/docker-compose.override.yml`
- [ ] Add API keys to `LibreChat/.env`

### Step 3: Build and Run

- [ ] Run `cd LibreChat && docker compose up -d --build`
- [ ] Check logs: `docker logs your-agent-container`
- [ ] Verify health: `curl http://localhost:8000/health`

### Step 4: Test in LibreChat UI

- [ ] Open http://localhost:3080
- [ ] Create account and login
- [ ] Select your endpoint from dropdown
- [ ] Send test message
- [ ] Verify response appears

---

## Common Mistakes to Avoid

### Mistake 1: Wrong YAML Indentation

**Wrong:**
```yaml
custom:
- name: "FinAI"  # Wrong indentation
```

**Correct:**
```yaml
custom:
  - name: "FinAI"  # Correct: 2 spaces before dash
```

### Mistake 2: Using localhost in baseURL

**Wrong:**
```yaml
baseURL: "http://localhost:8000/v1"  # Won't work in Docker
```

**Correct:**
```yaml
baseURL: "http://finai-api:8000/v1"  # Use container name
```

### Mistake 3: Missing /v1 in baseURL

**Wrong:**
```yaml
baseURL: "http://finai-api:8000"  # Missing /v1
```

**Correct:**
```yaml
baseURL: "http://finai-api:8000/v1"  # Include /v1
```

### Mistake 4: Not Implementing Streaming

If you don't implement streaming, LibreChat may show empty responses.

**Solution:** Always implement streaming support in your `/v1/chat/completions` endpoint.

### Mistake 5: Forgetting CORS

Your agent needs CORS middleware to accept requests from LibreChat.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify LibreChat origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Testing Your Integration

### Test 1: Health Check

```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy",...}
```

### Test 2: Models Endpoint

```bash
curl http://localhost:8000/v1/models
# Expected: {"object":"list","data":[...]}
```

### Test 3: Non-streaming Chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"test"}],"stream":false}'
```

### Test 4: Streaming Chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"test"}],"stream":true}'
```

---

## Environment Variables

### In LibreChat/.env

```bash
# Your API keys
NVIDIA_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Secrets (generate new ones for production)
CREDS_KEY=your_creds_key_here
CREDS_IV=your_creds_iv_here
JWT_SECRET=your_jwt_secret_here
JWT_REFRESH_SECRET=your_refresh_secret_here
```

### In docker-compose.override.yml

Pass environment variables to your container:

```yaml
finai:
  environment:
    - NVIDIA_API_KEY=${NVIDIA_API_KEY}
    - LOG_LEVEL=INFO
```

---

## File Structure Overview

```
MultiAgentFinanceApp/
├── Dockerfile                          # Your agent's Dockerfile
├── requirements.prod.txt               # Python dependencies
├── .env                                # API keys
├── src/
│   └── app.py                          # Your FastAPI app
│
└── LibreChat/
    ├── librechat.yaml                  # ← MODIFY THIS (endpoint config)
    ├── docker-compose.override.yml     # ← MODIFY THIS (service definition)
    ├── .env                            # ← ADD API KEYS HERE
    └── docker-compose.yml              # (don't modify)
```

---

## Summary: What Files to Modify

### 1. LibreChat/librechat.yaml

**Purpose**: Define your custom endpoint in LibreChat

**What to change**:
- `endpoints.custom` section
- Set your endpoint name, URL, and models

### 2. LibreChat/docker-compose.override.yml

**Purpose**: Add your agent as a Docker service

**What to change**:
- Add new service under `services:`
- Configure build context, environment, ports

### 3. LibreChat/.env

**Purpose**: Store API keys and secrets

**What to change**:
- Add `NVIDIA_API_KEY` or other required keys

---

## Complete Example: Adding a New Custom Agent

Let's say you want to add a "WeatherBot" agent:

### Step 1: Create librechat.yaml entry

```yaml
endpoints:
  custom:
    - name: "WeatherBot"
      apiKey: "weather-bot-key"
      baseURL: "http://weatherbot-api:9000/v1"
      models:
        default:
          - "weather-model"
      fetch: false
      modelDisplayLabel: "Weather Bot"
```

### Step 2: Add to docker-compose.override.yml

```yaml
services:
  api:
    depends_on:
      - weatherbot
  
  weatherbot:
    build:
      context: ./weatherbot  # Path to WeatherBot code
      dockerfile: Dockerfile
    container_name: weatherbot-api
    ports:
      - "9000:9000"
    environment:
      - WEATHER_API_KEY=${WEATHER_API_KEY}
    networks:
      - default
```

### Step 3: Add API key

```bash
echo "WEATHER_API_KEY=your_weather_key" >> LibreChat/.env
```

### Step 4: Restart

```bash
cd LibreChat && docker compose up -d --build
```

---

## Debugging Tips

### Check LibreChat Configuration Loading

```bash
docker logs LibreChat | grep -i "config\|custom\|error"
```

### Check Network Connectivity

```bash
docker exec LibreChat ping your-agent-container -c 3
```

### Check Request/Response

```bash
# View all FinAI logs in real-time
docker logs finai-api -f

# Check LibreChat trying to reach your agent
docker logs LibreChat -f | grep "your-agent"
```

---

**Last Updated**: April 12, 2026  
**Version**: 1.0.0
