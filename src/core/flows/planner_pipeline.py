"""Flow that routes a query through :func:`src.core.pipeline.run_pipeline`.

This is the **canonical execution path** for ALL non-trivial FinAI
queries. The dispatcher routes here by default; only ``smalltalk``
and ``meta_help`` take the deterministic fast-path.

Event handling
--------------
* ``_status`` events are turned into italic chat-visible progress
  lines so the user sees per-step progress in real time.
* The synthesizer's ``text`` event wraps in a LibreChat artifact
  when ``decision["wants_artifact"]`` is True; otherwise inline.
* All step-level events (``step_content``, ``step_tool_call``,
  ``step_tool_result``) are forwarded to the chat so the user sees
  agent reasoning in real time.
* Debate events (``header``, ``text`` from personas, etc.) pass
  through as-is.

Intent-flag derivation
----------------------
The classifier emits a coarse ``intent`` enum, not the granular
``intent_flags`` vocabulary the registry uses. ``_derive_intent_flags``
provides a deterministic mapping so the registry's policy gates
fire correctly.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Optional

from src.config import mcp_servers
from src.core.artifacts import (
    close_artifact,
    open_artifact,
    safe_id,
    status,
)
from src.core.panel import PanelEvent
from src.core.pipeline import run_pipeline
from src.core.router import RouteDecision
from src.core.types import KNOWN_INTENT_FLAGS


log = logging.getLogger("finai.flows.planner_pipeline")


def _derive_intent_flags(decision: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Translate a classifier RouteDecision into the new flag vocab.

    Mapping (best-effort, deterministic — refines as the classifier
    is upgraded):

    * ``deep_stock_research`` → claim-tracking + filings + historical
      news + deep_research.
    * ``portfolio_analysis`` → portfolio data + panel debate.
    * ``stock_research`` → no flags by default; ``want_panel`` adds
      the panel-debate flag.
    * ``topic_research`` / ``educational`` / ``meta_help`` /
      ``smalltalk`` → no flags. The planner falls back to
      research_agent + synthesizer.
    """
    flags = {f: False for f in KNOWN_INTENT_FLAGS}
    if not decision:
        return flags

    intent = (decision.get("intent") or "").strip()
    want_panel = bool(decision.get("want_panel", False))

    if intent == "deep_stock_research":
        flags["wants_claim_tracking"] = True
        flags["wants_filings"] = True
        flags["wants_historical_news"] = True
        flags["wants_deep_research"] = True
        flags["wants_panel_debate"] = True
    elif intent == "portfolio_analysis":
        flags["wants_portfolio_data"] = True
        flags["wants_panel_debate"] = True
    elif intent == "stock_research" and want_panel:
        flags["wants_panel_debate"] = True
    # topic_research, educational, meta_help, smalltalk:
    # leave all flags False; the planner's catalog gives it research_agent
    # by default and the synthesizer is always allowed.

    if want_panel:
        flags["wants_panel_debate"] = True

    return flags


async def run(
    query: str,
    decision: Optional[RouteDecision] = None,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Bridge :func:`run_pipeline` to the dispatcher's PanelEvent stream.

    Behaviour:
    * Forwards every ``_status`` event as an italic chat progress line.
    * Forwards all step-level events (content, tool calls, results)
      so the user sees agent reasoning in real time.
    * If ``decision["wants_artifact"]`` is True, wraps the synthesizer's
      report in a LibreChat artifact.
    """
    intent_flags = _derive_intent_flags(decision or {})
    wants_artifact = bool((decision or {}).get("wants_artifact", False))
    topic = ((decision or {}).get("topic") or "").strip() or query.strip()[:60]

    try:
        tools = await mcp_servers.get_tools()
    except Exception as e:
        log.exception("planner_pipeline: MCP tool fetch failed")
        yield {
            "type": "error",
            "text": f"Could not load MCP tools for planner pipeline: {e}",
        }
        return

    artifact_open = False
    artifact_id = safe_id(query, prefix="finai-planner")
    artifact_title = f"Multi-agent: {topic}" if topic else "Multi-agent investigation"

    async for ev in run_pipeline(
        query,
        intent_flags=intent_flags,
        all_mcp_tools=tools,
        user_id=user_id,
    ):
        etype = ev.get("type")

        if etype == "_status":
            yield status(ev.get("text", ""))
            continue

        if etype == "step_content":
            text = ev.get("text", "")
            if text:
                yield {"type": "text", "text": text, "persona": ev.get("persona", "agent")}
            continue

        if etype == "step_tool_call":
            tool_name = ev.get("tool", "?")
            args = ev.get("args", {})
            args_str = ""
            if args:
                try:
                    import json
                    args_str = " " + json.dumps(args, default=str)[:60]
                except Exception:
                    pass
            yield {
                "type": "text",
                "text": f"_Calling {tool_name}…{args_str}_\n\n",
                "persona": ev.get("persona", "agent"),
            }
            continue

        if etype == "step_tool_result":
            tool_name = ev.get("tool", "?")
            preview = ev.get("result_preview", "")
            yield {
                "type": "text",
                "text": f"_→ {tool_name}: {preview}_\n\n",
                "persona": ev.get("persona", "agent"),
            }
            continue

        if etype == "text" and ev.get("persona") == "synthesizer":
            text = ev.get("text", "")
            if not text:
                continue
            if wants_artifact and not artifact_open:
                yield open_artifact(
                    identifier=artifact_id,
                    title=artifact_title,
                )
                artifact_open = True
            yield {"type": "text", "text": text, "persona": "synthesizer"}
            continue

        # Debate events (header, text from personas, tool_call,
        # tool_result, persona_verdict, etc.) pass through as-is.
        if etype in ("header", "tool_call", "tool_result",
                     "persona_verdict", "panel_done"):
            yield ev
            continue

        # error / other → forward as-is
        yield ev

    if artifact_open:
        yield close_artifact()


__all__ = [
    "_derive_intent_flags",
    "run",
]
