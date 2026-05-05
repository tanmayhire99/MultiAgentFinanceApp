"""Flow that routes a query through :func:`src.core.pipeline.run_pipeline`.

This is the **bridge** between the dispatcher and the new planner-first
pipeline (Day 6-7 slice engine):

* The dispatcher routes here when the user opts in via the ``/planner``
  prefix (Day 10 slice). After the demo we'll layer in an env-var
  auto-route table; for now the `/planner` prefix is the only entry
  point so the existing static flows keep handling regular traffic
  unchanged.
* This wrapper takes care of turning the pipeline's
  :class:`PanelEvent` stream into something the dispatcher's existing
  rendering logic understands - specifically, converting ``_status``
  events into italic chat-visible text and (optionally) wrapping the
  synthesizer's report in a LibreChat artifact.

Event handling
--------------

* ``_status`` events from :mod:`src.core.executor` /
  :mod:`src.core.pipeline` get rewritten into italic chat lines via
  :func:`src.core.artifacts.status` (matches the static flows' Fix 3
  status-line UX so the user sees per-step progress in real time).
* The synthesizer's single ``text`` event (with
  ``persona="synthesizer"``) carries the user-visible report. When
  ``decision["wants_artifact"]`` is True we open a LibreChat artifact
  block first so the report renders in the side pane; otherwise it
  streams inline in the chat.
* Other events (``error`` from the pipeline, anything we don't
  recognise) flow through unchanged so nothing gets dropped.

Intent-flag derivation
----------------------
The current classifier emits a coarse ``intent`` enum + ``want_panel``
boolean, NOT the granular ``intent_flags`` vocabulary the new
:mod:`src.core.agents.registry` policy gates use. Until the
classifier is upgraded (post-demo), :func:`_derive_intent_flags`
provides a deterministic mapping based on the existing
:class:`RouteDecision` so the registry's gates can fire correctly.

When the dispatcher is updated to call the upgraded classifier
directly, this helper becomes a one-line passthrough.
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
    elif intent == "portfolio_analysis":
        flags["wants_portfolio_data"] = True
        flags["wants_panel_debate"] = True
    # stock_research, topic_research, educational, meta_help, smalltalk:
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

    * Always emits a brief inline header so the user knows the planner
      pipeline (rather than the static flow) is handling this turn.
    * Forwards every ``_status`` event from the pipeline as an italic
      chat line via :func:`src.core.artifacts.status`.
    * If ``decision["wants_artifact"]`` is True, lazily opens a
      LibreChat artifact on the synthesizer's first text emission so
      the report lands in the side pane.
    * Forwards ``error`` events unchanged so the dispatcher's existing
      error rendering kicks in.
    """
    intent_flags = _derive_intent_flags(decision or {})
    wants_artifact = bool((decision or {}).get("wants_artifact", False))
    topic = ((decision or {}).get("topic") or "").strip() or query.strip()[:60]

    # Inline header so demo audiences see the planner-first slice
    # explicitly. Keep it short — the pipeline's first _status event
    # already says "Planning a multi-agent investigation for...".
    yield {
        "type": "text",
        "text": (
            "_Routing through the planner-first pipeline "
            "(`/planner` opt-in)._\n\n"
        ),
        "persona": "orchestrator",
    }

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
