"""Top-level routing dispatcher for FinAI requests.

Every user query goes through this single entry point, which:

1. Runs the **intent router** (a small LLM call, see :mod:`src.core.router`).
2. Optionally emits a **Classification** section showing which flow was
   picked and why - hidden by default for end users, but useful during
   demos. Toggle via:

   * ``FINAI_VERBOSE_TRACE=1`` env var (global default)
   * ``/trace `` prefix on a single message (one-shot override, even when
     the env var is off)

3. Initialises the MCP workers (cached; no-op after the first call).
4. Dispatches ALL queries through the planner-first pipeline
   :func:`src.core.flows.planner_pipeline.run` — every agent is a
   standalone :class:`ScopedAgent` called only when the planner puts
   its name in a plan step. Two fast-path exceptions:

        smalltalk  — short conversational reply (zero LLM calls)
        meta_help  — curated capabilities answer (zero LLM calls)

5. Emits a regulatory disclaimer footer **only** for flows that produced
   real financial analysis (portfolio / stock / deep / topic). Concept
   explanations, capability listings, and chitchat get no disclaimer -
   the intent is informational, not advisory.
"""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from src.personas.base import register_tools
from src.config import mcp_servers
from src.core.flows import (
    meta_help,
    planner_pipeline,
    smalltalk,
)
from src.core.panel import (
    PanelEvent,
    install_tool_cache_wrappers,
)
from src.core.router import RouteDecision, classify_query, render_decision_card


log = logging.getLogger("finai.dispatcher")


_FAST_PATH = {
    "smalltalk": smalltalk.run,
    "meta_help": meta_help.run,
}

# Intents whose responses constitute "financial analysis" and therefore
# need the regulatory disclaimer. Everything else (concept explanations,
# capability listings, casual chitchat) skips the disclaimer.
_FINANCE_FLOW_INTENTS = frozenset({
    "portfolio_analysis",
    "stock_research",
    "deep_stock_research",
    "topic_research",
})


_DISCLAIMER_TEXT = (
    "\n\n---\n"
    "⚠️ **Disclaimer** — This response is an educational, multi-agent "
    "analysis produced by a demo system (LLMs + curated data + live "
    "search). It does not constitute personalised investment advice, a "
    "buy/sell recommendation, or a price target. Consult a "
    "SEBI-registered (or locally licensed) advisor before making "
    "investment decisions.\n"
)


# ---------------------------------------------------------------------------
# Verbose-trace toggle
# ---------------------------------------------------------------------------
def _env_verbose_trace() -> bool:
    """Read the ``FINAI_VERBOSE_TRACE`` env var. Truthy values: 1, true, yes."""
    raw = (os.getenv("FINAI_VERBOSE_TRACE") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _strip_trace_prefix(query: str) -> Tuple[str, bool]:
    """Extract a leading ``/trace `` prefix from ``query``.

    Returns ``(stripped_query, force_trace)``. If the user prefixed the
    message with ``/trace `` (case-insensitive), we set ``force_trace``
    to True for THIS request only and remove the prefix from the query
    so the router sees the real intent.

    The prefix is "/trace " with a trailing space so ``/tracehello`` is
    not mistaken for a trace request. Plain ``/trace`` (no payload) is
    also accepted - the rest of the line is the query.
    """
    q = query.lstrip()
    lower = q.lower()
    for prefix in ("/trace ", "/trace\t"):
        if lower.startswith(prefix):
            return q[len(prefix):].lstrip(), True
    if lower.rstrip() == "/trace":
        # Empty payload: nothing to ask, but flip the flag anyway in
        # case the user simply wants to see the routing for an empty
        # follow-up. classify_query will treat the empty string as
        # "default" and return a sensible fallback.
        return "", True
    return query, False


# ---------------------------------------------------------------------------
# Artifact-mode detection
# ---------------------------------------------------------------------------
# Default UX is "stream everything inline in the chat, like Claude does
# for casual replies". The artifact pane (right side panel) is opt-in,
# triggered by either:
#   1. A leading slash command:  ``/artifact <query>`` or ``/report <query>``
#   2. A natural-language ask elsewhere in the query: "generate report",
#      "show in artifact", "as artifact", "make a report", etc.
#
# We deliberately keep the regex narrow — false positives would put a
# heavy dense report into the side pane when the user just wanted an
# inline answer, which is harder to recover from than the opposite.
import re as _re

_ARTIFACT_PREFIXES = ("/artifact ", "/report ", "/artifact\t", "/report\t")
_ARTIFACT_BARE = {"/artifact", "/report"}
# Match natural-language asks for a report / artifact / document. The
# verb / noun split allows up to 3 adjective-ish words in between
# ("generate **detailed** report", "make a **comprehensive financial**
# report") so common phrasings work without being so loose that
# unrelated text triggers it.
_ARTIFACT_PHRASES = _re.compile(
    r"\b(?:"
    # action verb + (optional article / pronoun) + (up to 3 word slots)
    # + report / artifact / document
    r"(?:generate|make|create|give\s+me|show\s+me|produce|write|draft|"
    r"build|prepare)"
    r"(?:\s+(?:a|an|the|me|us|some|that|this))?"
    r"(?:\s+\w+){0,3}?"
    r"\s+(?:report|artifact|document)|"
    # explicit pane / artifact references
    r"show\s+(?:in|as)(?:\s+the)?\s+artifact|"
    r"in\s+(?:the\s+)?(?:artifact|side)\s+pane|"
    r"as\s+(?:an?\s+)?(?:artifact|report|document)"
    r")\b",
    _re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Planner-pipeline — default router
# ---------------------------------------------------------------------------
# All non-trivial queries route through the planner-first pipeline
# :func:`src.core.flows.planner_pipeline.run`. The `/planner` prefix is
# retained for backwards compatibility (treated as a no-op — it no longer
# gates anything). The only fast-path intents that skip the planner are
# ``smalltalk`` and ``meta_help`` (zero LLM calls).
_PLANNER_PREFIXES = ("/planner ", "/planner\t")
_PLANNER_BARE = "/planner"


def _strip_planner_prefix(query: str) -> str:
    """Strip the legacy ``/planner`` prefix (now a no-op — all queries
    go through the planner pipeline by default). Retained so old
    slash-command habits don't break anything."""
    q = query.lstrip()
    lower = q.lower()
    for prefix in _PLANNER_PREFIXES:
        if lower.startswith(prefix):
            return q[len(prefix):].lstrip()
    if lower.rstrip() == _PLANNER_BARE:
        return ""
    return query


def _strip_artifact_prefix(query: str) -> Tuple[str, bool]:
    """Detect ``wants_artifact`` and strip any slash-command prefix.

    Returns ``(stripped_query, wants_artifact)``. wants_artifact is True
    if either:

    * the message has a ``/artifact `` or ``/report `` prefix
      (case-insensitive), OR
    * the message body matches one of :data:`_ARTIFACT_PHRASES`
      ("generate report", "as artifact", "in the side pane", etc.)

    Slash-command prefixes are stripped from the returned query so the
    router sees the real intent (e.g. ``/report tell me about WDC``
    classifies as stock_research with the right ticker).
    """
    q = query.lstrip()
    lower = q.lower()

    for prefix in _ARTIFACT_PREFIXES:
        if lower.startswith(prefix):
            return q[len(prefix):].lstrip(), True
    if lower.rstrip() in _ARTIFACT_BARE:
        return "", True

    # Natural-language phrase elsewhere in the message
    if _ARTIFACT_PHRASES.search(q):
        return q, True

    return query, False


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
    # 0) Detect /trace, /planner, and /artifact prefixes on the user
    #    message. Each is one-shot:
    #      /trace    — toggles the developer routing card on for this
    #                   one request (independent of the env flag)
    #      /planner  — legacy prefix, now a no-op; all queries route
    #                   through the planner by default
    #      /artifact — opts INTO the LibreChat artifact pane (otherwise
    #                   everything streams inline in the chat,
    #                   Claude-style)
    #    Strip them before the classifier so it sees the real intent.
    query, force_trace = _strip_trace_prefix(query)
    verbose_trace = force_trace or _env_verbose_trace()
    query = _strip_planner_prefix(query)
    query, wants_artifact = _strip_artifact_prefix(query)

    # 1) Intent classification. If verbose_trace is on, we narrate the
    #    routing step so the audience can see the agent graph at work.
    #    Otherwise the classification is silent (still computed, still
    #    logged - just not streamed to the user).
    has_history = bool(history) and len(history) > 1
    if verbose_trace:
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

    # Stash the artifact-mode flag on the decision so the planner
    # pipeline can decide whether to emit ``:::artifact{}:::`` wrappers.
    # The smalltalk/meta_help fast paths ignore it (they're already
    # short and inline).
    decision["wants_artifact"] = wants_artifact  # type: ignore[typeddict-unknown-key]

    log.info(
        "route: query=%r intent=%s tickers=%s wants_artifact=%s rationale=%s",
        query[:80], decision.get("intent"),
        decision.get("tickers"), wants_artifact,
        decision.get("rationale"),
    )

    if verbose_trace:
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

    # 3) Dispatch. All intents route through the planner-first pipeline
    #    by default. The only exceptions are smalltalk (greetings) and
    #    meta_help (capabilities/help) — those are deterministic,
    #    zero-LLM paths that don't need a plan.
    intent = decision.get("intent", "educational")
    flow = _FAST_PATH.get(intent, planner_pipeline.run)
    if intent in _FAST_PATH:
        log.info("dispatcher: fast-path intent=%s (no LLM needed)", intent)
    else:
        log.info("dispatcher: routing intent=%s through planner-first pipeline", intent)

    # Per-tool narration events (`tool_call`, `tool_result`) and
    # ``header`` banner events are dev-facing trace. We strip them from
    # the user-visible stream unless verbose_trace is on. The flows
    # also stop emitting many of these in Fix 3, but we filter
    # centrally so the gating is uniform across every flow - including
    # the persona panel and deepagents harness which we're not
    # rewriting.
    _DEV_TRACE_EVENT_TYPES = frozenset({"tool_call", "tool_result"})
    try:
        async for ev in flow(query, decision, user_id):
            if not verbose_trace and ev.get("type") in _DEV_TRACE_EVENT_TYPES:
                continue
            yield ev
    except Exception as e:
        log.exception("Flow %s crashed", intent)
        yield {
            "type": "error",
            "text": f"Flow `{intent}` failed: {e}",
        }

    # 4) Disclaimer footer - finance flows only. A "hi" or a "what is
    #    EBITDA?" doesn't need a SEBI advisor warning, but a portfolio
    #    panel verdict does. The disclaimer is decoupled from the
    #    verbose-trace flag because it's a regulatory concern, not a
    #    UX concern - it appears whenever advisory-shaped content was
    #    actually produced.
    if intent in _FINANCE_FLOW_INTENTS:
        yield {
            "type": "text",
            "text": _DISCLAIMER_TEXT,
            "persona": "orchestrator",
        }
    yield {"type": "panel_done"}
