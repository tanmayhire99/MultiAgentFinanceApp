"""Top-level routing dispatcher for FinAI requests.

Every user query goes through this single entry point, which:

1. Runs the **intent router** (a small LLM call, see :mod:`src.core.router`).
2. Emits a visible **Classification** section at the top of the response so
   the audience sees which flow was picked and why. This is the single
   biggest change that makes the system legible as a real LangGraph
   router instead of a hardcoded pipeline.
3. Initialises the MCP workers (cached; no-op after the first call).
4. Dispatches to one of six flows in :mod:`src.core.flows`:

        portfolio_analysis   - full investor panel over the user's portfolio
        stock_research       - focused deep dive on specific ticker(s)
        deep_stock_research  - multi-step claim-tracking + SEC + historical news
        topic_research       - web research on a macro / sector question
        educational          - direct LLM explanation of a finance concept
        meta_help            - curated capabilities answer (zero LLM calls)

5. Emits a universal disclaimer footer + terminal ``panel_done`` event.

Keeping the orchestration logic in one module means only one place has
to care about telemetry, error handling, or disclaimer wording.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from src.agents.personas.base import register_tools
from src.config import mcp_servers
from src.core.flows import (
    deep_stock_research,
    educational,
    meta_help,
    portfolio_analysis,
    stock_research,
    topic_research,
)
from src.core.panel import (
    PanelEvent,
    install_tool_cache_wrappers,
)
from src.core.router import RouteDecision, classify_query, render_decision_card


log = logging.getLogger("finai.dispatcher")


_FLOW_MAP = {
    "portfolio_analysis": portfolio_analysis.run,
    "stock_research": stock_research.run,
    "deep_stock_research": deep_stock_research.run,
    "topic_research": topic_research.run,
    "educational": educational.run,
    "meta_help": meta_help.run,
}


_DISCLAIMER_TEXT = (
    "\n\n---\n"
    "⚠️ **Disclaimer** — This response is an educational, multi-agent "
    "analysis produced by a demo system (LLMs + curated data + live "
    "search). It does not constitute personalised investment advice, a "
    "buy/sell recommendation, or a price target. Consult a "
    "SEBI-registered (or locally licensed) advisor before making "
    "investment decisions.\n"
)


async def run_analysis(
    query: str,
    user_id: str = "demo",
    history: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[PanelEvent]:
    """Classify ``query`` and dispatch to the appropriate flow.

    ``history`` is the full OpenAI-style messages list (including the
    current user turn); it is forwarded to the router so follow-up
    queries can chain onto the previous classification.

    Yields :class:`PanelEvent` dicts compatible with the SSE renderer in
    :mod:`src.core.streaming`. Safe to call concurrently for different
    users thanks to the shared MCP cache + per-persona API key slots.
    """
    # 1) Intent classification - a small, visible step.
    has_history = bool(history) and len(history) > 1
    if has_history:
        yield {
            "type": "text",
            "text": (
                "_Routing your query through the intent classifier "
                "(with prior-turn context)…_\n\n"
            ),
            "persona": "orchestrator",
        }
    else:
        yield {
            "type": "text",
            "text": "_Routing your query through the intent classifier…_\n\n",
            "persona": "orchestrator",
        }
    try:
        decision: RouteDecision = await classify_query(query, history=history)
    except Exception as e:
        log.exception("Router dispatch failure")
        # Should be unreachable - classify_query catches its own errors
        # and returns a fallback decision - but keep a belt-and-braces guard.
        decision = {
            "intent": "educational",
            "tickers": [],
            "topic": query[:60],
            "want_panel": False,
            "rationale": f"Classifier errored; defaulting to educational. ({e})",
        }

    yield {
        "type": "text",
        "text": render_decision_card(decision, query),
        "persona": "orchestrator",
    }

    # 2) Pre-warm / refresh the MCP tool pool. This is cached across
    #    requests so the first call pays subprocess spawn cost and
    #    subsequent calls are instant. We also push the tool list into
    #    the persona registry so ReAct agents discover their tools.
    #
    #    CRITICAL: langchain-mcp-adapters 0.2.x spawns a fresh MCP
    #    subprocess for every tool.ainvoke() call - which kills the
    #    intended in-subprocess TTL caches. We wrap every tool with a
    #    main-process TTL cache so repeat calls with the same args
    #    (orchestrator + personas all asking for WDC fundamentals)
    #    skip the subprocess round-trip entirely.
    try:
        tools = await mcp_servers.get_tools()
        install_tool_cache_wrappers(tools)
        register_tools(tools)
    except Exception as e:
        log.exception("MCP initialisation failed inside dispatcher")
        yield {
            "type": "error",
            "text": f"Failed to initialise MCP workers: {e}",
        }
        yield {"type": "panel_done"}
        return

    # 3) Dispatch to the selected flow.
    intent = decision.get("intent", "educational")
    flow = _FLOW_MAP.get(intent)
    if flow is None:
        log.warning("Dispatcher received unknown intent %r; using educational", intent)
        flow = _FLOW_MAP["educational"]

    try:
        async for ev in flow(query, decision, user_id):
            yield ev
    except Exception as e:
        log.exception("Flow %s crashed", intent)
        yield {
            "type": "error",
            "text": f"Flow `{intent}` failed: {e}",
        }

    # 4) Universal disclaimer (one place, same wording for every flow).
    yield {
        "type": "text",
        "text": _DISCLAIMER_TEXT,
        "persona": "orchestrator",
    }
    yield {"type": "panel_done"}
