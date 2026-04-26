"""Portfolio Analysis flow - two modes, gated on ``decision.want_panel``.

Picked by the dispatcher when the router classifies a query as
``portfolio_analysis`` (i.e. the user asked about their own holdings).

Two output shapes, consistent with :mod:`src.core.flows.stock_research`:

**want_panel=False (default)** — short, data-first briefing, ~25-30 s:

    # Portfolio Analysis
        ## 1. Portfolio Overview        (Portfolio Agent handoffs)
        ## 2. Market Snapshot           (Stock + Research Agent handoffs)
        ## 3. Portfolio Analyst Summary (single GPT-OSS-120B call)

**want_panel=True** — full investor panel debate, ~50-60 s:

    # Portfolio Analysis
        ## 1. Portfolio Overview        (Portfolio Agent handoffs)
        ## 2. Market Snapshot           (Stock + Research Agent handoffs)
        ## 3. Investor Panel Debate     (Buffett -> Wood -> Graham, parallel)
        ## 4. Closing Brief             (moderator synthesis)

This matches the stock_research pattern: ``want_panel`` is the single
knob that turns the persona debate on or off, so the visible
"Panel requested?" value on the classification card always agrees with
what actually runs below it.

The heavy lifting (``PortfolioContext``, ``_orchestrator_fetch_portfolio``,
``_stream_persona_events``, etc.) lives in :mod:`src.core.panel`; this
module only wires those pieces together in the portfolio-specific order.

MCP tool registration and the final disclaimer are handled by
:mod:`src.core.dispatcher`.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.personas.base import build_chat_model
from src.agents.personas.moderator import moderator_open_stream
from src.core.debate import PanelScratchpad, run_debate_loop
from src.core.panel import (
    PanelEvent,
    PortfolioContext,
    _orchestrator_fetch_portfolio,
)
from src.core.resilient_stream import stream_llm_resilient
from src.core.router import RouteDecision


log = logging.getLogger("finai.flows.portfolio")


_ANALYST_SYSTEM = (
    "You are a careful, neutral portfolio analyst. You have just been "
    "handed a structured snapshot of a user's portfolio: holdings with "
    "weights, sector mix, concentration flags, a diversification score, "
    "live fundamentals for the top holdings, and any recent catalysts.\n\n"
    "Write a short portfolio briefing in this exact structure:\n\n"
    "**Overall read** — 2-3 sentences summarising the shape of the "
    "portfolio and its key strengths / risks.\n"
    "**What's working** — 2-3 bullets, each citing a specific holding, "
    "metric, or sector weight from the snapshot.\n"
    "**What to watch** — 2-3 bullets highlighting the concentration "
    "flags, any stretched valuations, and live catalysts visible in the "
    "snapshot.\n"
    "**Suggested next questions** — 2-3 bullets like 'Ask: what do the "
    "panel personas think about my top holdings?' that nudge the user "
    "toward the panel or a single-stock deep dive.\n\n"
    "Rules:\n"
    "- Aim for 220-340 words total.\n"
    "- Cite numbers from the snapshot; do NOT invent any.\n"
    "- Neutral tone. No buy/sell calls, no price targets.\n"
    "- Do NOT end with a disclaimer; one is appended separately.\n"
    "- Do NOT include the investor panel's personas' names in this "
    "summary - this section is the data-first alternative to the panel."
)


async def run(
    query: str,
    decision: Optional[RouteDecision] = None,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Run the portfolio analysis, panel or no panel based on ``decision``.

    Fix 3 layout (matches stock_research):

    * Inline mode (default) — a brief italic chat status line, then the
      whole analysis (overview, market snapshot, optional panel debate,
      summary / closing brief) streams directly into the chat.
    * Artifact mode (``decision["wants_artifact"]=True``) — same status
      line in chat, then the structured content is wrapped in a
      ``:::artifact{}:::`` block for LibreChat's side pane, plus a
      one-line chat summary at the end.

    The H1 banner ``# 📋 FinAI Portfolio Analysis`` and the
    "This response is organised in three parts:" tour-guide preamble
    that previous versions emitted have been removed - they were
    dev-style narration the user never asked for.
    """
    from src.core.artifacts import (
        artifact_body,
        chat_text,
        close_artifact,
        open_artifact,
        safe_id,
        status,
    )

    decision = decision or {}
    want_panel = bool(decision.get("want_panel"))
    wants_artifact = bool(decision.get("wants_artifact"))

    # 1) One brief inline status line (more come from the panel.py
    # orchestrator helpers below). No banner, no TOC paragraph.
    if want_panel:
        yield status(
            "Pulling your portfolio for full investor panel debate "
            "(~60-180s)..."
        )
    else:
        yield status("Pulling your portfolio holdings + analytics...")

    # 2) Lazy artifact open: in artifact mode the wrapper opens on
    # first content event; in inline mode it never opens and content
    # flows straight to chat.
    artifact_title = (
        "Portfolio Analysis + Panel Debate"
        if want_panel
        else "Portfolio Analysis"
    )
    artifact_id = safe_id(f"portfolio-analysis-{user_id}")
    artifact_open = False

    def _ensure_artifact_open() -> List[PanelEvent]:
        nonlocal artifact_open
        if artifact_open or not wants_artifact:
            return []
        artifact_open = True
        return [
            open_artifact(identifier=artifact_id, title=artifact_title),
            artifact_body(f"# {artifact_title}\n\n"),
        ]

    def _route(ev: PanelEvent) -> List[PanelEvent]:
        """Route a single flow event - open the artifact if needed."""
        return _ensure_artifact_open() + [ev]

    # 3) Orchestrator: portfolio fetch + analytics + market snapshot.
    # Forwards every event from panel.py except the structural
    # _portfolio_ready handoff.
    portfolio_ctx: Optional[PortfolioContext] = None
    async for ev in _orchestrator_fetch_portfolio(user_id=user_id):
        if ev.get("type") == "_portfolio_ready":
            portfolio_ctx = ev.get("ctx")  # type: ignore[assignment]
            continue
        for piece in _route(ev):
            yield piece

    if not portfolio_ctx or not portfolio_ctx.has_data():
        # No artifact to close (the failure happened before any content
        # was rendered). Just emit an inline error.
        if artifact_open:
            yield close_artifact()
        yield {
            "type": "error",
            "text": (
                "Portfolio data is empty; skipping the rest of the analysis. "
                "This usually means the Portfolio Agent could not find the "
                "requested user in its fixture."
            ),
        }
        return

    # 4) Branch on want_panel
    if want_panel:
        async for ev in _run_panel_branch(query, portfolio_ctx, user_id=user_id):
            for piece in _route(ev):
                yield piece
    else:
        async for ev in _run_analyst_summary_branch(
            query, portfolio_ctx, user_id=user_id
        ):
            for piece in _route(ev):
                yield piece

    # 5) Close artifact + emit chat summary
    if artifact_open:
        yield close_artifact()
    if wants_artifact:
        yield chat_text(
            f"Portfolio analysis ready. Full report in the artifact pane →\n"
        )


# ---------------------------------------------------------------------------
# No-panel branch: single-analyst briefing grounded in the snapshot
# ---------------------------------------------------------------------------
async def _run_analyst_summary_branch(
    query: str,
    ctx: PortfolioContext,
    *,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Emit the single-analyst portfolio summary (no persona debate).

    Fix 3: section header only — the dev-style "_No panel was
    requested, so we're running a single neutral LLM synthesis..._"
    italic was removed. Tells the user nothing they care about.
    """
    yield {
        "type": "text",
        "text": "\n## Analyst Summary\n\n",
        "persona": "orchestrator",
    }

    snapshot_md = _format_snapshot_for_llm(ctx)
    messages = [
        SystemMessage(content=_ANALYST_SYSTEM),
        HumanMessage(
            content=(
                f"User's question: {query}\n\n"
                f"Portfolio snapshot (already computed by the orchestrator):\n\n"
                f"{snapshot_md}\n\n"
                "Write the briefing now, following the system prompt structure."
            )
        ),
    ]

    async def _stream() -> AsyncIterator[str]:
        llm = build_chat_model(temperature=0.2, max_tokens=1600, streaming=True)
        async for chunk in llm.astream(messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text

    async for chunk in stream_llm_resilient(
        stream_factory=_stream,
        user_id=user_id,
        query=query,
        flow_name="portfolio_no_panel",
        cache_agent="analyst_summary",
        cache_agent_title="Portfolio Analyst Summary",
        retries=1,
        error_label="analyst summary",
    ):
        yield {"type": "text", "text": chunk, "persona": "moderator"}

    yield {"type": "text", "text": "\n\n", "persona": "moderator"}


def _format_snapshot_for_llm(ctx: PortfolioContext) -> str:
    """Markdown-ish bundle of the portfolio data for the analyst prompt."""
    parts: List[str] = []
    if ctx.summary:
        total = ctx.summary.get("total_value_usd", 0)
        parts.append(f"Total value: ${total:,.2f} (USD)")
        parts.append(f"Holding count: {ctx.summary.get('holding_count', len(ctx.holdings))}")
        geo = ctx.summary.get("geographic_split") or {}
        if geo:
            parts.append("Geographic split: " + ", ".join(f"{k} {v}%" for k, v in geo.items()))
    parts.append("")
    parts.append("All holdings:")
    parts.append(ctx.top_holdings_table_md())
    parts.append("")
    parts.append("Sector allocation:")
    parts.append(ctx.sector_summary_md() or "(n/a)")
    parts.append("")
    parts.append("Concentration flags:")
    parts.append(ctx.risks_summary_md())
    parts.append("")
    parts.append(ctx.score_summary_md() or "(no diversification score)")
    snap_md = ctx.market_snapshot_md()
    if snap_md:
        parts.append("")
        parts.append("Live fundamentals snapshot (every holding):")
        parts.append(snap_md)
    cat_md = ctx.catalysts_md()
    if cat_md:
        parts.append("")
        parts.append("Recent catalysts:")
        parts.append(cat_md)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# With-panel branch: moderator + multi-round sequential debate + closing brief
# ---------------------------------------------------------------------------
_DEBATE_SYNTH_SYSTEM = (
    "You are the moderator of the FinAI Investor Panel. The three "
    "analyst personas have just completed a multi-round sequential "
    "debate on the user's portfolio, with a shared scratchpad visible "
    "to every speaker. Your job is to write the **Closing Brief** "
    "section, grounded in both the live portfolio snapshot and the "
    "full debate transcript you will be shown.\n\n"
    "Follow this structure exactly:\n\n"
    "**Where the panel converged:** 1-2 bullets on points the three "
    "analysts now agree on. If the panel never converged, say so.\n"
    "**Where the panel remained divergent:** 1-2 bullets naming who is "
    "on each side and citing a specific claim from the transcript.\n"
    "**How stances evolved:** 2-3 sentences summarising movements "
    "across rounds. Call out any persona who shifted stance and the "
    "argument that moved them.\n"
    "**What it means for your portfolio:** 2-3 sentences connecting "
    "the (resolved or persisting) disagreement back to specific "
    "holdings or concentration flags from the snapshot.\n"
    "**What to watch next:** 2-4 bullets with educational indicators "
    "the user should monitor. No buy/sell calls, no price targets.\n\n"
    "Rules:\n"
    "- Aim for 260-380 words total.\n"
    "- Cite panelists by name when referencing a claim.\n"
    "- Quote specific numbers from the portfolio snapshot where useful.\n"
    "- Do NOT invent new metrics.\n"
    "- Do NOT include a caveat - a standalone disclaimer follows."
)


def _format_scratchpad_for_moderator(scratchpad: PanelScratchpad) -> str:
    """Render the full debate transcript for the moderator synthesis prompt."""
    lines: List[str] = [f"Query: {scratchpad.query}", ""]
    rounds_used = sorted({e.round for e in scratchpad.entries})
    for r in rounds_used:
        lines.append(f"=== Round {r} ===")
        for entry in scratchpad.entries_for_round(r):
            lines.append(
                f"\n### {entry.persona_title} — stance: {entry.stance} "
                f"({entry.confidence} confidence)"
            )
            if entry.one_liner:
                lines.append(f"One-liner: {entry.one_liner}")
            if entry.content:
                lines.append(entry.content)
            lines.append("")
    evolution = scratchpad.stance_evolution_md()
    if evolution:
        lines.append("Stance evolution:")
        lines.append(evolution)
    return "\n".join(lines)


async def _run_panel_branch(
    query: str,
    ctx: PortfolioContext,
    *,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Emit the multi-round debate + closing brief.

    Fix 3: header simplified to ``## Investor Panel Debate``; the
    "(3-Round Sequential)" parenthetical and "_Warming up the panel..._"
    explanatory paragraph were dev-narration that the user already
    sees in action via the round-by-round persona stream below.
    """
    yield {
        "type": "text",
        "text": "\n## Investor Panel Debate\n\n### Moderator — Opening\n\n",
        "persona": "orchestrator",
    }
    mod_ctx_block = ctx.moderator_context_block()

    async def _moderator_open_stream() -> AsyncIterator[str]:
        async for chunk in moderator_open_stream(query, mod_ctx_block):
            yield chunk

    async for chunk in stream_llm_resilient(
        stream_factory=_moderator_open_stream,
        user_id=user_id,
        query=query,
        flow_name="portfolio_panel",
        cache_agent="moderator_opening",
        cache_agent_title="Moderator Opening",
        retries=1,
        error_label="moderator opening",
    ):
        yield {"type": "text", "text": chunk, "persona": "moderator"}
    yield {"type": "text", "text": "\n\n---\n\n", "persona": "moderator"}

    # 4) Run the sequential multi-round debate loop
    scratchpad: Optional[PanelScratchpad] = None
    async for ev in run_debate_loop(
        query,
        portfolio_ctx=ctx,
        user_id=user_id,
        flow_name="portfolio_panel",
    ):
        if ev.get("type") == "_debate_done":
            scratchpad = ev.get("scratchpad")  # type: ignore[assignment]
            continue
        yield ev

    if scratchpad is None:
        yield {"type": "error", "text": "Debate loop finished without a scratchpad."}
        return

    # 5) Closing Brief - moderator synthesis over the FULL transcript.
    # Fix 3: dropped the "_Moderator synthesising..._" italic dev-note;
    # users see the closing brief streaming in real time so the heads-up
    # was redundant.
    yield {
        "type": "text",
        "text": "\n## Closing Brief\n\n",
        "persona": "orchestrator",
    }
    transcript = _format_scratchpad_for_moderator(scratchpad)
    # Feed the moderator the FULL persona context (holdings table, market
    # snapshot, moat / growth / defensive / catalysts) so the closing
    # brief can reference the same numbers and news items the personas
    # debated - not just the one-line summary in ``mod_ctx_block``.
    full_ctx = ctx.persona_context_block()
    synth_messages = [
        SystemMessage(content=_DEBATE_SYNTH_SYSTEM),
        HumanMessage(
            content=(
                f"Portfolio one-liner: {mod_ctx_block}\n\n"
                f"Full portfolio context (identical to what the personas saw):\n\n"
                f"{full_ctx}\n\n"
                f"Full debate transcript:\n\n{transcript}\n\n"
                "Write the Closing Brief now, following the system prompt structure."
            )
        ),
    ]

    async def _synth_stream() -> AsyncIterator[str]:
        llm = build_chat_model(temperature=0.2, max_tokens=1100, streaming=True)
        async for chunk in llm.astream(synth_messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text

    async for chunk in stream_llm_resilient(
        stream_factory=_synth_stream,
        user_id=user_id,
        query=query,
        flow_name="portfolio_panel",
        cache_agent="moderator_synthesis",
        cache_agent_title="Moderator Closing Brief",
        retries=1,
        error_label="moderator synthesis",
    ):
        yield {"type": "text", "text": chunk, "persona": "moderator"}
