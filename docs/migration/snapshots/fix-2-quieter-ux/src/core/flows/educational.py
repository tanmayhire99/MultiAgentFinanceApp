"""Educational flow - direct concept explanation, zero agents, zero tools.

Triggered by the router when the user asks *what / how / explain* style
conceptual questions ("Explain compound interest", "What is beta?"). We
deliberately bypass every agent and MCP tool: the answer is purely the
LLM grounded in the educator system prompt.

This keeps the response fast (~3-5 s TTFT on GPT-OSS-120B), gives the
audience a visibly different shape ("no tool calls -> no handoffs"), and
makes the "right tool for the job" story of the router legible.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.personas.base import build_chat_model
from src.core.panel import PanelEvent
from src.core.router import RouteDecision


log = logging.getLogger("finai.flows.educational")


_SYSTEM = (
    "You are a clear, trustworthy financial educator. The user has asked a "
    "CONCEPT question - their goal is understanding, not a trade "
    "recommendation. Write a concise, structured explanation in this "
    "format:\n\n"
    "**Definition** — the concept in one plain-English sentence.\n"
    "**How it works** — 2-4 sentences on the underlying mechanics.\n"
    "**A worked example** — one short numerical walkthrough (e.g. rupee "
    "or dollar amounts, simple calculation). Use a bulleted list or a "
    "tiny markdown table if it clarifies the numbers.\n"
    "**Common misconceptions** — 1-2 bullets the user should NOT believe.\n"
    "**Related concepts worth exploring** — 2-3 bullets pointing at the "
    "next ideas they could learn.\n\n"
    "Rules:\n"
    "- Aim for 220-360 words total.\n"
    "- Do NOT reference specific stocks, tickers, or the user's "
    "portfolio unless they explicitly asked about one.\n"
    "- Do NOT provide investment advice.\n"
    "- Do NOT use hype words. Be factual and neutral.\n"
    "- End with ONE sentence suggesting the next concept to learn."
)


async def run(
    query: str,
    decision: Optional[RouteDecision] = None,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Emit the educational answer as streaming text events.

    No MCP tool calls, no agent handoffs - a single streamed LLM call on
    a dedicated educator system prompt.

    The flow used to emit a banner header + a dev-facing italic note
    explaining that no agent was being called. Both have been removed
    so the user sees the answer directly, the way they would from any
    chat assistant. The dispatcher's ``verbose_trace`` already shows
    the routing decision when the developer wants it.
    """
    llm = build_chat_model(temperature=0.3, max_tokens=900, streaming=True)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"User's question: {query}\n\nAnswer it now."),
    ]

    try:
        async for chunk in llm.astream(messages):
            text = getattr(chunk, "content", None)
            if text:
                yield {"type": "text", "text": text, "persona": "moderator"}
    except Exception as e:
        log.exception("Educational LLM call failed")
        yield {"type": "error", "text": f"Educator LLM call failed: {e}"}
        return

    yield {"type": "text", "text": "\n\n", "persona": "moderator"}
