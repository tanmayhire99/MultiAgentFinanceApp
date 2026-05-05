"""FinAI FastAPI service.

This module is the single LibreChat-facing surface. It preserves the OpenAI
compatibility contract (``/v1/chat/completions`` + ``/v1/models``) while the
actual work is delegated to the Investor Panel supervisor in
:mod:`src.core.panel`.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
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
from .core.dispatcher import run_analysis
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


load_dotenv(".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


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
        "dispatched to one of four flows: portfolio_analysis (full "
        "Buffett/Wood/Graham panel), stock_research (focused deep dive), "
        "topic_research (open-ended web search), or educational (concept "
        "explanation). MCP workers back the flows that need data."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def chat_completions(req: ChatCompletionRequest):
    user_message = _last_user_message(req.messages)
    # Map whatever the caller sent in ``user`` (LibreChat session id,
    # "anonymous", None, ...) to an actual portfolio fixture user.
    user_id = _resolve_portfolio_user(req.user)
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
    return {
        "status": "healthy",
        "service": "finai",
        "version": "2.0.0",
        "timestamp": int(time.time()),
    }


@app.get("/")
async def root():
    return {
        "service": "FinAI Multi-Agent Router",
        "version": "3.0.0",
        "description": (
            "Intent router + four flows (portfolio_analysis, stock_research, "
            "topic_research, educational). Every request is classified by a "
            "small LLM call before the right agents are invoked."
        ),
        "router": {
            "intents": list(INTENTS),
            "classifier_model": DEFAULT_MODEL,
        },
        "flows": {
            "portfolio_analysis": "Full Buffett/Wood/Graham panel on the user's portfolio",
            "stock_research": "Focused deep dive on specific ticker(s); optional panel via want_panel",
            "topic_research": "Open-ended web research via the Research Agent",
            "educational": "Direct LLM explanation; no agents, no tools",
        },
        "personas": ["buffett", "wood", "graham"],
        "mcp_workers": list(mcp_servers.MCP_SERVERS.keys()),
        "llm": {
            "backend": "NVIDIA NIM",
            "model": DEFAULT_MODEL,
        },
        "api_keys": {
            # Do NOT expose key material; just show which slots are populated.
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
async def query_entry(body: QueryRequest):
    """Legacy endpoint: runs the panel and returns the full transcript once.

    Retained for backwards compatibility with the old documentation; new
    clients should prefer ``/v1/chat/completions``.
    """
    events = run_analysis(body.query)
    transcript = await collect_transcript(events)
    return {
        "pattern": "finai_router",
        "query": body.query,
        "answer": transcript,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
