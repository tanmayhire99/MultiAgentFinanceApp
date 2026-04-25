"""Moderator node: opens and closes the investor panel.

Unlike the three persona agents, the moderator has no tools - it is a single
LLM call that (a) re-states the user's query for the panel and (b) synthesises
the persona verdicts into a balanced final briefing.

Both functions are exposed as **streaming async generators** so the FastAPI
SSE adapter can forward tokens to LibreChat as they arrive, giving the
audience a live "typing" effect instead of one chunky dump per section.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from .base import PersonaVerdict, build_chat_model


_OPEN_SYSTEM = (
    "You are the moderator of the FinAI Investor Panel. Your job is to open "
    "the session for the three analyst personas who will speak next: Warren "
    "Buffett (value), Cathie Wood (disruptive innovation) and Benjamin Graham "
    "(defensive).\n\n"
    "Given the user's question and (if present) a snapshot of their "
    "portfolio, write ONE short paragraph (80-120 words) that:\n"
    "1. Clearly re-states the question for the panel.\n"
    "2. Identifies the tickers, sectors, or topics in focus — quoting one "
    "or two concrete numbers from the portfolio snapshot when relevant.\n"
    "3. Frames the axes of disagreement the audience should watch for "
    "(e.g. tech concentration vs innovation exposure, margin of safety vs "
    "growth runway).\n\n"
    "Write in a neutral, matter-of-fact broadcaster voice. Address the user "
    "directly in the second person. Do NOT pre-empt any persona's view."
)


_SYNTH_SYSTEM = (
    "You are the moderator of the FinAI Investor Panel. The three analyst "
    "personas have just delivered their views. Your job is to write the "
    "**Closing Brief** section of a portfolio analysis report, in the "
    "context of the user's actual portfolio.\n\n"
    "Follow this structure exactly:\n"
    "**Where the panel agrees:** 1-2 bullets.\n"
    "**Where the panel disagrees:** 1-2 bullets, naming who is on each side.\n"
    "**What it means for your portfolio:** 2-3 sentences that connect the "
    "disagreement back to specific holdings or concentration flags from the "
    "snapshot. Do NOT pick a winner; surface the trade-off.\n"
    "**What to watch next:** 2-4 bullets with *educational* things to monitor "
    "(e.g. specific catalysts from the research snapshot, sector concentration "
    "thresholds, earnings releases). Do NOT suggest buying/selling any security.\n\n"
    "Be concise: the full brief should be 180-260 words. Quote specific "
    "numbers or phrases from the panel and the portfolio snapshot where "
    "helpful. Do not invent new metrics. Do not include a caveat - a "
    "standalone disclaimer is shown separately after your brief."
)


async def moderator_open_stream(
    user_query: str, portfolio_brief: str = ""
) -> AsyncIterator[str]:
    """Yield the moderator's opening paragraph token-by-token.

    Args:
        user_query: Original user question (free text).
        portfolio_brief: One-line deterministic portfolio summary (as
            produced by :meth:`src.core.panel.PortfolioContext.moderator_context_block`).
            Empty string when no portfolio context is available.
    """
    llm = build_chat_model(temperature=0.3, max_tokens=500)
    user_block = f"User's question: {user_query}"
    if portfolio_brief:
        user_block += (
            f"\n\nPortfolio snapshot (already computed by the orchestrator): "
            f"{portfolio_brief}"
        )
    user_block += "\n\nOpen the panel now."
    messages = [
        SystemMessage(content=_OPEN_SYSTEM),
        HumanMessage(content=user_block),
    ]
    async for chunk in llm.astream(messages):
        text = getattr(chunk, "content", None)
        if text:
            yield text


async def moderator_open(user_query: str, portfolio_brief: str = "") -> str:
    """Non-streaming convenience wrapper around :func:`moderator_open_stream`."""
    parts: List[str] = []
    async for chunk in moderator_open_stream(user_query, portfolio_brief):
        parts.append(chunk)
    return "".join(parts).strip()


def _format_verdicts_for_synthesis(verdicts: List[PersonaVerdict]) -> str:
    lines: List[str] = []
    for v in verdicts:
        lines.append(f"### {v.get('title', v.get('persona', 'Unknown'))}")
        lines.append(f"Stance: {v.get('stance', 'neutral')}  |  Confidence: {v.get('confidence', 'low')}")
        lines.append(f"One-liner: {v.get('one_liner', '(none)')}")
        rationale = v.get("rationale", "").strip()
        if rationale:
            lines.append("Rationale:")
            lines.append(rationale)
        lines.append("")
    return "\n".join(lines).strip()


async def moderator_synthesise_stream(
    user_query: str,
    verdicts: List[PersonaVerdict],
    portfolio_brief: str = "",
) -> AsyncIterator[str]:
    """Yield the moderator's closing synthesis token-by-token."""
    # Slightly more room for synthesis since portfolio-aware runs carry
    # more structural content (agreement + disagreement + takeaway + caveat).
    llm = build_chat_model(temperature=0.2, max_tokens=900)
    transcript = _format_verdicts_for_synthesis(verdicts)
    portfolio_line = (
        f"\nPortfolio snapshot: {portfolio_brief}\n"
        if portfolio_brief
        else ""
    )
    messages = [
        SystemMessage(content=_SYNTH_SYSTEM),
        HumanMessage(
            content=(
                f"User's original question: {user_query}\n"
                f"{portfolio_line}\n"
                f"Panel views so far:\n\n{transcript}\n\n"
                "Produce the closing briefing now, following the structure in "
                "the system prompt exactly."
            )
        ),
    ]
    async for chunk in llm.astream(messages):
        text = getattr(chunk, "content", None)
        if text:
            yield text


async def moderator_synthesise(
    user_query: str,
    verdicts: List[PersonaVerdict],
    portfolio_brief: str = "",
) -> str:
    """Non-streaming wrapper around :func:`moderator_synthesise_stream`."""
    parts: List[str] = []
    async for chunk in moderator_synthesise_stream(
        user_query, verdicts, portfolio_brief
    ):
        parts.append(chunk)
    return "".join(parts).strip()
