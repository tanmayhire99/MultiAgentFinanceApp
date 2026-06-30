"""FinAI FastAPI service.

This module is the single LibreChat-facing surface. It preserves the OpenAI
compatibility contract (``/v1/chat/completions`` + ``/v1/models``) while the
actual work is delegated to the Investor Panel supervisor in
:mod:`src.core.panel`.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .personas.base import (
    DEFAULT_MODEL,
    PERSONA_API_KEY_SLOT,
    list_configured_slots,
)
from .mcp._fixtures import load_fixture
from .config import mcp_servers
from .core import alerts
from .core.auth import authenticate_request, get_jwt_secret, is_auth_enabled
from .core.dispatcher import run_analysis
from .core.ratelimit import create_limiter_from_env
from .core.router import INTENTS
from .core.streaming import collect_transcript, stream_openai_chunks


# ---------------------------------------------------------------------------
# Portfolio user resolution
# ---------------------------------------------------------------------------
# The portfolio fixture is the single source of truth for which user IDs
# the demo can analyse. LibreChat, the Responses playground, and curl tests
# all send different things in the OpenAI ``user`` field (a MongoDB
# ObjectId, the literal string "anonymous", or nothing at all). Any value
# that isn't one of the fixture's actual users is mapped to the default
# ``demo`` portfolio so the Portfolio Agent always has something to work
# with - this is the demo's equivalent of "resolve session -> portfolio".
_KNOWN_PORTFOLIO_USERS = {
    k for k in load_fixture("portfolio").keys() if not k.startswith("_")
}
_DEFAULT_PORTFOLIO_USER = (
    "demo" if "demo" in _KNOWN_PORTFOLIO_USERS
    else next(iter(_KNOWN_PORTFOLIO_USERS), "demo")
)


def _resolve_portfolio_user(requested: Optional[str]) -> str:
    """Map an OpenAI-style ``user`` field to a known portfolio fixture user."""
    if requested and requested in _KNOWN_PORTFOLIO_USERS:
        return requested
    return _DEFAULT_PORTFOLIO_USER


def _require_user(request: Request, fallback: Optional[str] = None) -> str:
    """Authenticate the request (when auth is enabled) and resolve a user id.

    Mirrors the gate used by the chat endpoint so the alerts API enforces the
    same JWT policy, then maps to a known portfolio user.
    """
    auth = authenticate_request(dict(request.headers), fallback_user=fallback)
    if not auth.authenticated and is_auth_enabled():
        raise HTTPException(status_code=401, detail=auth.error or "Unauthorized")
    return _resolve_portfolio_user(auth.user_id if auth.authenticated else fallback)


load_dotenv(".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

_rate_limiter = create_limiter_from_env()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Pre-warm the MCP workers so the first user request isn't slowed by
    # subprocess spawn time. This is a best-effort attempt - if warming fails
    # we still start, and the first real request will retry.
    try:
        await mcp_servers.get_tools()
    except Exception:
        logging.getLogger("finai.app").exception("MCP pre-warm failed (will retry on first request)")
    yield
    await mcp_servers.shutdown()


app = FastAPI(
    title="FinAI Multi-Agent Router",
    description=(
        "Multi-agent finance demo. Exposes an OpenAI-compatible "
        "chat-completions endpoint used by LibreChat. Every request is "
        "first classified by an intent router (GPT-OSS-120B) and then "
        "routed through a planner-first multi-agent pipeline — the "
        "planner generates a DAG of standalone agent steps, executed "
        "with debate and synthesis. MCP workers back the agents that "
        "need live data."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=_lifespan,
)
# CORS: lock to explicit origins in production via FINAI_ALLOWED_ORIGINS
# (comma-separated). Wildcard "*" is unsafe for a credentialed API and is in fact
# rejected by browsers alongside allow_credentials=True. Defaults to local dev.
_allowed_origins = [
    o.strip()
    for o in os.environ.get(
        "FINAI_ALLOWED_ORIGINS", "http://localhost:3080,http://localhost:8000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_access_log = logging.getLogger("finai.access")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request ID, time the request, and emit one structured access
    log line per request. Honors an inbound ``X-Request-ID`` (so logs correlate
    across a reverse proxy) and echoes it back on the response."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        dur_ms = (time.perf_counter() - start) * 1000.0
        _access_log.exception(
            "rid=%s %s %s -> EXCEPTION in %.1fms", rid, request.method, request.url.path, dur_ms
        )
        raise
    dur_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = rid
    _access_log.info(
        "rid=%s %s %s -> %d in %.1fms",
        rid, request.method, request.url.path, response.status_code, dur_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Schema (OpenAI-compatible)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    # Whatever the caller sends here (LibreChat uses the MongoDB user ID,
    # OpenAI playgrounds send a random string, curl tests often omit it)
    # is mapped to a portfolio fixture user by ``_resolve_portfolio_user``.
    # Leaving this ``None`` lets the resolver pick the default fixture user.
    user: Optional[str] = None
    stream: Optional[bool] = False


class QueryRequest(BaseModel):
    """Schema for the legacy ``POST /query`` endpoint."""

    query: str
    profile: Optional[dict] = None
    transactions: Optional[List[dict]] = None


# ---------------------------------------------------------------------------
# LibreChat-facing endpoints
# ---------------------------------------------------------------------------
def _last_user_message(messages: List[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return msg.content
    raise HTTPException(status_code=400, detail="No user message found in request")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    user_message = _last_user_message(req.messages)
    auth = authenticate_request(
        dict(request.headers),
        fallback_user=req.user,
    )
    if not auth.authenticated and is_auth_enabled():
        raise HTTPException(status_code=401, detail=auth.error or "Unauthorized")
    raw_user = auth.user_id if auth.authenticated else req.user
    user_id = _resolve_portfolio_user(raw_user)
    rl = _rate_limiter.check(user_id)
    if not rl.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for user '{user_id}'",
            headers={"Retry-After": str(rl.retry_after_seconds or 1)},
        )
    # Forward the full conversation history so the intent router can
    # chain follow-ups ("do panel analysis as well", "what about P/E?")
    # onto the previous turn's classification card.
    history = [m.model_dump() for m in req.messages]
    events = run_analysis(user_message, user_id=user_id, history=history)

    if req.stream:
        return StreamingResponse(
            stream_openai_chunks(events, model=req.model),
            media_type="text/event-stream",
        )

    transcript = await collect_transcript(events)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": transcript},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint used by LibreChat."""
    return {
        "object": "list",
        "data": [
            {
                "id": "finai-advisor",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "finai",
                "permission": [
                    {
                        "id": "modelperm-fini",
                        "object": "model_permission",
                        "created": int(time.time()),
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False,
                    }
                ],
                "root": "finai-advisor",
                "parent": None,
            }
        ],
    }


@app.get("/health")
async def health_check():
    """Liveness + cheap readiness. Intentionally does NO network/DB I/O (the
    container HEALTHCHECK polls this often); deep dependency checks would belong
    in a separate readiness probe."""
    tools = mcp_servers.tools_loaded_count()
    checks = {
        "llm_key_configured": bool(os.environ.get("NVIDIA_API_KEY")),
        "mcp_tools_loaded": tools,
        "mcp_warm": tools > 0,
        "warehouse_enabled": bool(os.environ.get("WAREHOUSE_DATABASE_URL")),
        "quant_enabled": "quant" in mcp_servers.MCP_SERVERS,
    }
    return {
        "status": "healthy" if checks["llm_key_configured"] else "degraded",
        "service": "finai",
        "version": "3.0.0",
        "timestamp": int(time.time()),
        "checks": checks,
    }


@app.get("/")
async def root():
    return {
        "service": "FinAI Multi-Agent Router",
        "version": "3.0.0",
        "description": (
            "Intent router + planner-first multi-agent pipeline. "
            "Every request is classified by a small LLM call, "
            "then the planner orchestrates a DAG of standalone agents "
            "(research, portfolio, filings, panel, synthesizer). "
            "fast-path intents (smalltalk, meta_help) skip the planner."
        ),
        "auth": {
            "enabled": is_auth_enabled(),
            "jwt_configured": get_jwt_secret() is not None,
        },
        "router": {
            "intents": list(INTENTS),
            "classifier_model": DEFAULT_MODEL,
        },
        "pipeline": {
            "intents": list(INTENTS),
            "description": "All intents route through planner-first pipeline; smalltalk/meta_help are zero-LLM fast paths",
        },
        "personas": ["buffett", "wood", "graham"],
        "mcp_workers": list(mcp_servers.MCP_SERVERS.keys()),
        "llm": {
            "backend": "NVIDIA NIM",
            "model": DEFAULT_MODEL,
        },
        "api_keys": {
            "configured_slots": list_configured_slots(),
            "persona_slot_assignments": PERSONA_API_KEY_SLOT,
        },
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "docs": "/docs",
            "legacy_query": "/query",
        },
    }


# ---------------------------------------------------------------------------
# Legacy endpoint kept for backwards compatibility with HOW_TO_RUN.md
# ---------------------------------------------------------------------------
@app.post("/query")
async def query_entry(body: QueryRequest, request: Request):
    """Legacy endpoint: runs the panel and returns the full transcript once.

    Retained for backwards compatibility with the old documentation; new
    clients should prefer ``/v1/chat/completions``.
    """
    auth = authenticate_request(dict(request.headers))
    if not auth.authenticated and is_auth_enabled():
        raise HTTPException(status_code=401, detail=auth.error or "Unauthorized")
    rl_user = auth.user_id if auth.authenticated else "anonymous"
    rl = _rate_limiter.check(rl_user)
    if not rl.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded",
            headers={"Retry-After": str(rl.retry_after_seconds or 1)},
        )
    events = run_analysis(body.query)
    transcript = await collect_transcript(events)
    return {
        "pattern": "finai_router",
        "query": body.query,
        "answer": transcript,
    }


# ---------------------------------------------------------------------------
# Event-driven portfolio alerts API (src.core.alerts)
# ---------------------------------------------------------------------------
class MarkReadRequest(BaseModel):
    alert_id: Optional[int] = None  # None => mark all of the user's alerts read


@app.get("/alerts")
async def get_alerts(request: Request, unread_only: bool = False, limit: int = 50):
    """List the authenticated user's portfolio alerts, newest first.

    Returns ``unread_count`` alongside the list so a UI can render an unread
    badge from a single call.
    """
    user_id = _require_user(request)
    limit = max(1, min(int(limit or 50), 200))
    return {
        "user_id": user_id,
        "unread_count": alerts.unread_count(user_id),
        "alerts": alerts.list_alerts(user_id, unread_only=unread_only, limit=limit),
    }


@app.post("/alerts/scan")
async def scan_alerts(request: Request):
    """Scan the user's current holdings for new alerts.

    Runs the concentration rule plus the **live day-move feed** (fetches each
    holding's 1-day change from the US live quote / NSE warehouse) so price-move
    alerts fire for real. Rate-limited like the analysis endpoints since it does
    work on behalf of the user. Returns new-alert + unread counts.
    """
    user_id = _require_user(request)
    rl = _rate_limiter.check(user_id)
    if not rl.allowed:
        raise HTTPException(
            status_code=429, detail=f"Rate limit exceeded for user '{user_id}'",
            headers={"Retry-After": str(rl.retry_after_seconds or 1)},
        )
    new_ids = alerts.run_scan(user_id, quote_fn=alerts.live_quote_change)
    return {
        "user_id": user_id,
        "new_alerts": len(new_ids),
        "unread_count": alerts.unread_count(user_id),
        "alerts": alerts.list_alerts(user_id, limit=50),
    }


@app.post("/alerts/mark-read")
async def mark_alerts_read(body: MarkReadRequest, request: Request):
    """Mark one alert (by id) or all of the user's alerts as read."""
    user_id = _require_user(request)
    alerts.mark_read(user_id, body.alert_id)
    return {"user_id": user_id, "unread_count": alerts.unread_count(user_id)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
