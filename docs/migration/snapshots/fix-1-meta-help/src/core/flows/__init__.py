"""Per-intent flow modules dispatched by :mod:`src.core.dispatcher`.

Each flow is an async generator yielding :class:`src.core.panel.PanelEvent`
dicts. Flows share the supporting machinery (tool calling, persona
streaming, etc.) defined in :mod:`src.core.panel`, but decide on their
own which agents to invoke and what the final report should look like.

Flows exposed here:

* ``portfolio_analysis.run``    -- full investor panel over the user's portfolio
* ``stock_research.run``        -- focused deep dive on 1-N tickers
* ``deep_stock_research.run``   -- batch-mode multi-step deep agent with
                                   claim-tracking (SEC filings + historical
                                   news + forward-claim extraction + diff)
* ``topic_research.run``        -- open-ended web research on a theme/macro
* ``educational.run``           -- concept explanation; no agents, no tools
* ``meta_help.run``             -- curated FinAI capabilities answer; zero
                                   LLM calls (static markdown only)

All expose a coroutine with the signature
``run(query: str, decision: RouteDecision, user_id: str = "demo")``
and yield events compatible with the SSE renderer in
:mod:`src.core.streaming`.
"""
from . import (
    deep_stock_research,
    educational,
    meta_help,
    portfolio_analysis,
    stock_research,
    topic_research,
)

__all__ = [
    "deep_stock_research",
    "educational",
    "meta_help",
    "portfolio_analysis",
    "stock_research",
    "topic_research",
]
