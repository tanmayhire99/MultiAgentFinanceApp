"""Streaming adapter: turn :class:`PanelEvent`s into OpenAI SSE chunks.

The renderer is intentionally simple - each event becomes at most one SSE
``delta.content`` chunk - so that LLM tokens flow through the OpenAI-compat
contract at roughly the same rate they arrive from the upstream model.

The same renderer is used for the non-streaming response path: we just
concatenate every rendered chunk into a single ``content`` string.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from .panel import PanelEvent


STANCE_ICON = {
    "bullish": "🟢",
    "neutral": "⚪",
    "cautious": "🟡",
    "bearish": "🔴",
}
CONFIDENCE_ICON = {
    "low": "◔",
    "medium": "◑",
    "high": "●",
}

# Human-readable "Agent" labels for each MCP server, so viewers understand
# these tool calls are agent-to-agent handoffs (orchestration), not just
# function calls.
SERVER_AGENT_LABEL = {
    "portfolio": "Portfolio Agent",
    "us_stock": "US Stock Agent",
    "indian_stock": "Indian Stock Agent",
    "research": "Research Agent",
}

# Friendly labels for persona identifiers coming through on events.
PERSONA_LABEL = {
    "orchestrator": "🧭 Orchestrator",
    "moderator": "🎙 Moderator",
    "buffett": "Warren Buffett",
    "wood": "Cathie Wood",
    "graham": "Benjamin Graham",
}


def _format_tool_args(args: Dict[str, Any]) -> str:
    if not args:
        return ""
    try:
        return ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items()
        )
    except (TypeError, ValueError):
        return str(args)


def render_event(ev: PanelEvent) -> str:
    """Render a single panel event as a Markdown chunk for LibreChat.

    Returns an empty string for structural events that have no visible
    output (``panel_done``) so the caller can skip sending an empty SSE
    chunk.
    """
    kind = ev.get("type")

    if kind in ("header", "text"):
        return ev.get("text", "")

    if kind == "tool_call":
        tool = ev.get("tool", "?")
        short_tool = tool.split("__", 1)[-1] if "__" in tool else tool
        server = tool.split("__", 1)[0] if "__" in tool else ""
        server_label = SERVER_AGENT_LABEL.get(server, server or "Agent")
        persona_key = ev.get("persona", "?")
        persona_label = ev.get("persona_label") or PERSONA_LABEL.get(persona_key, persona_key)
        args = _format_tool_args(ev.get("args", {}))
        return (
            f"\n> 🔗 **{persona_label}** → **{server_label}** · "
            f"`{short_tool}({args})`\n"
        )

    if kind == "tool_result":
        preview = ev.get("result_preview", "").strip()
        if not preview:
            return ""
        return f"> ↩ _{preview}_\n\n"

    if kind == "persona_verdict":
        stance = str(ev.get("stance", "neutral")).lower()
        conf = str(ev.get("confidence", "low")).lower()
        one_liner = ev.get("one_liner", "")
        tools = ev.get("tools_used", [])
        tools_line = (
            f"\n\n_Tools consulted: {', '.join(f'`{t}`' for t in tools)}_"
            if tools
            else ""
        )
        stance_icon = STANCE_ICON.get(stance, "•")
        conf_icon = CONFIDENCE_ICON.get(conf, "◔")
        return (
            f"{tools_line}\n\n"
            f"**{stance_icon} Verdict:** {one_liner}  \n"
            f"**Stance:** {stance.title()}  |  **Confidence:** {conf_icon} {conf.title()}"
        )

    if kind == "panel_done":
        return ""

    if kind == "error":
        return f"\n> ⚠️ **Error:** {ev.get('text', 'unknown error')}\n\n"

    # Unknown event type: render a safe debug line rather than crashing the stream.
    return f"\n<!-- unhandled event: {json.dumps(ev, default=str)} -->\n"


# ---------------------------------------------------------------------------
# OpenAI SSE chunk builder
# ---------------------------------------------------------------------------
def _chunk(
    completion_id: str,
    created: int,
    model: str,
    *,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_openai_chunks(
    events: AsyncIterator[PanelEvent],
    *,
    model: str,
    completion_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """Yield Server-Sent-Events strings in OpenAI chat-completions format.

    Every non-empty rendered chunk becomes one ``delta.content`` SSE frame,
    so LibreChat (or any OpenAI-compatible client) sees the transcript grow
    token-by-token rather than in big section-sized dumps.
    """
    completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # First chunk establishes the role and avoids some clients treating an
    # empty initial delta as the whole message.
    yield _chunk(
        completion_id,
        created,
        model,
        delta={"role": "assistant", "content": ""},
    )

    async for ev in events:
        rendered = render_event(ev)
        if rendered:
            yield _chunk(
                completion_id,
                created,
                model,
                delta={"content": rendered},
            )

    yield _chunk(completion_id, created, model, delta={}, finish_reason="stop")
    yield "data: [DONE]\n\n"


async def collect_transcript(events: AsyncIterator[PanelEvent]) -> str:
    """Non-stream mode: concatenate every rendered event into one transcript."""
    parts = []
    async for ev in events:
        rendered = render_event(ev)
        if rendered:
            parts.append(rendered)
    return "".join(parts).strip()
