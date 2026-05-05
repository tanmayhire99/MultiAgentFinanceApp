"""``PanelScopedAgent`` — special-cased ScopedAgent for ``panel_agent``.

The panel agent is qualitatively different from every other agent in
the registry:

* In the registry it owns **zero MCP tools** (``tools=()``). Its work
  is done via three persona sub-agents (Buffett / Wood / Graham)
  orchestrated by the existing :mod:`src.core.debate` and
  :mod:`src.core.panel` modules.
* It is **policy-gated** — the registry rejects any plan that names
  ``panel_agent`` unless the classifier set
  ``intent_flags["wants_panel_debate"] = True``.
* Its ``run()`` is **NOT** a plain ReAct loop. The standard
  :class:`ScopedAgent` base class would compile a ReAct graph with no
  MCP tools and an empty system prompt — at best the LLM would emit a
  dry one-shot answer; at worst it would hallucinate a panel by
  inventing dialogue without invoking the actual personas.

To preserve the architecture (ScopedAgent is the unit of execution
the executor consumes) while delegating the actual debate to the
existing machinery, we define :class:`PanelScopedAgent` as a subclass
of :class:`ScopedAgent` that overrides :meth:`run`. The base
constructor still runs (registry validation, policy gate, prompt
assembly) so the agent shows up in tests and audit logs identically
to any other ScopedAgent.

What ``run()`` does
-------------------
1. **Build a portfolio context.** Walks the step's declared
   dependencies (``step.depends_on``). For each prior step that has a
   completed result, attempt to extract:

   * holdings + summary + sector allocation + concentration risks +
     diversification score (from a ``portfolio_agent`` step), OR
   * per-ticker fundamentals (from ``us_stock_agent`` /
     ``indian_stock_agent`` steps), and
   * per-ticker catalysts / news (from a ``research_agent`` step).

   If at least one source was found, populate a
   :class:`~src.core.panel.PortfolioContext`. Otherwise the debate
   runs ungrounded; that's a degraded mode but still valid.

2. **Run the multi-round sequential debate** via
   :func:`src.core.debate.run_debate_loop`. We drain the
   ``AsyncIterator`` of :class:`~src.core.panel.PanelEvent` dicts,
   appending the renderable ones (``text`` / ``header``) to a
   transcript-markdown buffer and capturing the final
   :class:`~src.core.debate.PanelScratchpad`.

3. **Synthesize the closing brief.** A single LLM call (moderator
   voice) over the transcript + portfolio context produces the
   user-visible "Closing Brief" section. We reuse the system prompt
   from :mod:`src.core.flows.portfolio_analysis` so output style
   matches the existing static-flow path.

4. **Return a** :class:`~src.core.types.StepResult` with the
   combined transcript + closing brief in ``output["text"]`` and
   structured metadata (verdicts, stance evolution,
   convergence-round) for downstream consumers.

Failure modes are caught at the boundary of each phase: if the
debate loop crashes we still return a ``failed`` StepResult so the
executor can mark the step terminal and the synthesizer can either
produce a partial report or surface an error.

The Stage-4 build deliberately does the **minimum end-to-end**
delivery: a panel can run from a planner-emitted plan even if the
plan only hands the panel agent a query. Stage 5 refines:

* the **dispatcher** so ``/planner panel ...`` queries actually
  route through the new pipeline (they currently fail at the
  factory layer in Stage 3);
* the **planner's worked examples** so the LLM knows the canonical
  shape of a "panel" plan (currently it has examples for stock
  research and claim tracking only).
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.agents._base import ScopedAgent
from src.core.panel import PanelEvent
from src.core.types import StepResult


# ``build_panel_agent`` lives in :mod:`src.core.agents._factories` so it
# shares the patched ``build_chat_model`` reference with every other
# factory function (the test suite patches that single module-level
# reference). This module exposes :class:`PanelScopedAgent` and the
# :meth:`run` override; the factory there just chooses model params and
# instantiates it.


log = logging.getLogger("finai.panel_agent")


# Reused verbatim so the planner-pipeline closing brief reads the same as
# the existing static-flow path. Imported lazily inside ``run()`` to keep
# this module's import graph cheap (no LangChain / persona machinery
# pulled in just to compile the class).
_DEBATE_SYNTH_SYSTEM_FALLBACK = (
    "You are the moderator of the FinAI Investor Panel. The three "
    "analyst personas have just completed a multi-round sequential "
    "debate on the user's question. Your job is to write a tight "
    "closing brief grounded in the transcript.\n\n"
    "Structure: Where the panel converged · Where the panel remained "
    "divergent · How stances evolved · What it means for the user · "
    "What to watch next. 220-380 words. No buy/sell calls; no price "
    "targets; no disclaimer."
)


class PanelScopedAgent(ScopedAgent):
    """ScopedAgent variant that orchestrates the multi-round panel debate.

    Inherits the full :class:`ScopedAgent` constructor (registry
    validation, policy-gate enforcement, system-prompt assembly) so
    that ``REGISTRY.validate_step`` running on a ``panel_agent`` step
    behaves identically to any other agent's. Only :meth:`run` is
    overridden.
    """

    async def run(self) -> AsyncIterator[PanelEvent]:  # type: ignore[override]
        """Run the debate, streaming events, then yield a terminal _step_result.

        Yields ``PanelEvent`` dicts from the debate loop (``header``,
        ``text``, ``tool_call``, ``tool_result``, etc.) in real time,
        then a terminal ``_step_result`` event carrying the
        :class:`StepResult` — matching the base class's streaming
        contract.

        Catches every exception inside the orchestration so that a
        crashed persona / failed LLM call surfaces as a ``failed``
        StepResult — never propagates to the executor.
        """
        started_at = time.time()
        try:
            from src.agents.personas.base import build_chat_model
            from src.core.debate import (
                PanelScratchpad,
                run_debate_loop,
            )
            from src.core.panel import PortfolioContext
        except Exception as exc: # pragma: no cover - import-time only
            log.exception("panel agent: failed to import debate machinery")
            yield {
                "type": "_step_result",
                "result": StepResult(
                    step_id=self.step.id,
                    status="failed",
                    output=None,
                    error=f"panel agent boot error: {exc}",
                    error_type=type(exc).__name__,
                    started_at=started_at,
                    completed_at=time.time(),
                    tools_used=[],
                ),
            }
            return

        try:
            ctx = self._build_portfolio_context(PortfolioContext)
        except Exception as exc:
            log.warning(
                "panel agent: building portfolio context failed: %s; "
                "running debate ungrounded", exc,
            )
            ctx = None

        query = self._panel_query()

        # Run the debate, forwarding events in real time
        transcript_lines: List[str] = []
        final_scratchpad = None

        try:
            async for ev in run_debate_loop(
                query,
                portfolio_ctx=ctx,
                user_id="demo",
                flow_name="planner_panel",
            ):
                ev_type = ev.get("type") if isinstance(ev, dict) else None
                if ev_type == "_debate_done":
                    final_scratchpad = ev.get("scratchpad")
                    continue
                if ev_type in ("header", "text"):
                    text = ev.get("text") if isinstance(ev, dict) else None
                    if isinstance(text, str) and text:
                        transcript_lines.append(text)
                # Forward ALL debate events to the pipeline (including
                # tool_call, tool_result, persona_verdict, etc.)
                yield ev
        except Exception as exc:
            log.exception("panel agent: debate loop crashed")
            yield {
                "type": "_step_result",
                "result": StepResult(
                    step_id=self.step.id,
                    status="failed",
                    output=None,
                    error=f"debate loop failed: {exc}",
                    error_type=type(exc).__name__,
                    started_at=started_at,
                    completed_at=time.time(),
                    tools_used=[],
                ),
            }
            return

        scratchpad = final_scratchpad
        transcript_md = "".join(transcript_lines).strip()

        # The conversation content is already streamed during the debate,
        # so we don't need to emit the full transcript again here.
        # This avoids duplication while keeping the streamed content visible.
        if transcript_md:
            pass  # No duplicate transcript yield - content already streamed

        try:
            closing_brief = await self._write_closing_brief(
                query=query,
                portfolio_ctx=ctx,
                scratchpad=scratchpad,
                build_chat_model=build_chat_model,
            )
        except Exception as exc:
            log.exception("panel agent: closing-brief synthesis crashed")
            closing_brief = (
                f"_(closing brief unavailable: {exc}; the debate "
                "transcript above is the full panel output.)_"
            )

        # Emit closing brief as a streamed text event too
        yield {
            "type": "text",
            "text": "\n\n## Panel Conversation Summary\n\n" + closing_brief + "\n",
            "persona": "moderator",
        }

        full_text = self._render_full_text(
            transcript_md=transcript_md,
            closing_brief=closing_brief,
        )

        verdicts = self._extract_verdicts(scratchpad)
        consensus_round = self._consensus_round(scratchpad)
        evolution_md = (
            scratchpad.stance_evolution_md() if scratchpad is not None else ""
        )

        yield {
            "type": "_step_result",
            "result": StepResult(
                step_id=self.step.id,
                status="complete",
                output={
                    "text": full_text,
                    "verdicts": verdicts,
                    "consensus_round": consensus_round,
                    "stance_evolution_md": evolution_md,
                    "rounds": len({e.round for e in scratchpad.entries})
                    if scratchpad is not None else 0,
                },
                tools_used=["panel_debate_loop", "moderator_synthesis"],
                started_at=started_at,
                completed_at=time.time(),
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _panel_query(self) -> str:
        """Pick the user-visible debate topic.

        The dispatcher hands the original query down to the pipeline,
        which puts it in ``Scratchpad.query``. The planner is also
        encouraged (via the system prompt) to put a brief debate
        topic in ``step.description``. We prefer the description if
        present (more concise) and fall back to the original query.
        """
        if self.step.description and self.step.description.strip():
            return self.step.description.strip()
        return self.scratchpad.query

    def _build_portfolio_context(
        self,
        PortfolioContext: type,
    ) -> Optional[Any]:
        """Best-effort recovery of a :class:`PortfolioContext` from deps.

        Walks ``step.depends_on`` and inspects each completed prior
        step's output. For:

        * a ``portfolio_agent`` step → adopt its holdings / summary /
          allocation / risks / score directly.
        * a ``us_stock_agent`` / ``indian_stock_agent`` step → record
          its fundamentals dict under the relevant ticker in
          ``market_snapshot``.
        * a ``research_agent`` step → record any catalyst lists under
          their tickers.

        If the result is a free-form LLM string (which is the
        ScopedAgent's default ``output["text"]`` on completion), we
        skip it — the panel personas will still see the user's query
        and can reason without snapshot data.

        Returns ``None`` if no usable data was extracted, which means
        the debate runs ungrounded (still valid, just less rich).
        """
        if not self.step.depends_on:
            return None

        ctx = PortfolioContext(user_id="demo")
        any_data = False

        for dep_id in self.step.depends_on:
            result = self.scratchpad.get(dep_id)
            if result is None or result.status != "complete":
                continue
            output = result.output
            if not isinstance(output, dict):
                continue

            # Heuristic 1 - portfolio shaped output
            if "holdings" in output and isinstance(
                output["holdings"], list
            ):
                ctx.holdings = output["holdings"]
                any_data = True
            if "summary" in output and isinstance(output["summary"], dict):
                ctx.summary = output["summary"]
                any_data = True
            if "allocation" in output and isinstance(output["allocation"], dict):
                ctx.allocation = output["allocation"]
                any_data = True
            if "risks" in output and isinstance(output["risks"], dict):
                ctx.risks = output["risks"]
                any_data = True
            if "score" in output and isinstance(output["score"], dict):
                ctx.score = output["score"]
                any_data = True

            # Heuristic 2 - per-ticker fundamentals from a stock agent
            ticker = output.get("ticker")
            if ticker and "fundamentals" in output:
                ctx.market_snapshot[str(ticker)] = output["fundamentals"]
                any_data = True
            if ticker and "growth_metrics" in output:
                ctx.growth[str(ticker)] = output["growth_metrics"]
                any_data = True
            if ticker and "defensive_metrics" in output:
                ctx.defensive[str(ticker)] = output["defensive_metrics"]
                any_data = True
            if ticker and "moat_signals" in output:
                signals = output["moat_signals"]
                if isinstance(signals, list):
                    ctx.moat_signals[str(ticker)] = signals
                    any_data = True

            # Heuristic 3 - catalysts from research_agent
            catalysts = output.get("catalysts")
            if isinstance(catalysts, dict):
                ctx.catalysts.update(catalysts)
                any_data = True

        if not any_data:
            return None
        return ctx

    async def _write_closing_brief(
        self,
        *,
        query: str,
        portfolio_ctx: Optional[Any],
        scratchpad: Any,
        build_chat_model: Any,
    ) -> str:
        """One-shot moderator synthesis over the full debate transcript."""
        if scratchpad is None:
            return (
                "_The debate did not produce a scratchpad; closing brief "
                "skipped._"
            )

        # Prefer the existing portfolio_analysis system prompt for
        # style parity. Falls back to an inlined version if that
        # module isn't importable (keeps tests light-weight).
        try:
            from src.core.flows.portfolio_analysis import (
                _DEBATE_SYNTH_SYSTEM,
                _format_scratchpad_for_moderator,
            )

            synth_system = _DEBATE_SYNTH_SYSTEM
            transcript = _format_scratchpad_for_moderator(scratchpad)
        except Exception:
            log.debug(
                "panel agent: falling back to inline closing-brief prompt"
            )
            synth_system = _DEBATE_SYNTH_SYSTEM_FALLBACK
            transcript = self._format_scratchpad_inline(scratchpad)

        ctx_block = ""
        if portfolio_ctx is not None:
            try:
                ctx_block = portfolio_ctx.persona_context_block()
            except Exception:
                ctx_block = ""

        user_message = (
            f"User's question: {query}\n\n"
            f"Portfolio / debate context:\n\n{ctx_block or '(no portfolio context)'}\n\n"
            f"Full debate transcript:\n\n{transcript}\n\n"
            "Write the Closing Brief now."
        )
        llm = build_chat_model(
            temperature=0.2, max_tokens=1100, streaming=False,
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=synth_system),
                HumanMessage(content=user_message),
            ]
        )
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for blk in content:
                if isinstance(blk, dict) and "text" in blk:
                    parts.append(str(blk["text"]))
                elif isinstance(blk, str):
                    parts.append(blk)
            return "\n".join(p for p in parts if p).strip()
        return str(content or "").strip()

    @staticmethod
    def _format_scratchpad_inline(scratchpad: Any) -> str:
        """Minimal scratchpad rendering when portfolio_analysis isn't available."""
        try:
            entries = list(getattr(scratchpad, "entries", []) or [])
        except Exception:
            return ""
        rounds_used = sorted({getattr(e, "round", 0) for e in entries})
        lines: List[str] = []
        for r in rounds_used:
            lines.append(f"=== Round {r} ===")
            for e in entries:
                if getattr(e, "round", None) != r:
                    continue
                lines.append(
                    f"\n### {getattr(e, 'persona_title', e.persona)} — "
                    f"stance: {getattr(e, 'stance', '?')} "
                    f"({getattr(e, 'confidence', '?')} confidence)"
                )
                if getattr(e, "one_liner", ""):
                    lines.append(f"One-liner: {e.one_liner}")
                if getattr(e, "content", ""):
                    lines.append(e.content)
                lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_full_text(*, transcript_md: str, closing_brief: str) -> str:
        """Render the panel agent's user-visible markdown.

        Sections match the static-flow path so the synthesizer step
        receives content the user is already familiar with.
        """
        # The conversation content is already streamed during execution,
        # so we only include the closing brief in the final output
        parts: List[str] = ["## Panel Conversation Summary", ""]
        parts.append(closing_brief or "_(no closing brief was produced)_")
        return "\n".join(parts).strip() + "\n"

    @staticmethod
    def _extract_verdicts(scratchpad: Any) -> List[Dict[str, Any]]:
        """Final-round verdict per persona, in PERSONA_ORDER."""
        if scratchpad is None:
            return []
        try:
            entries = list(getattr(scratchpad, "entries", []) or [])
        except Exception:
            return []
        if not entries:
            return []
        # Take each persona's LATEST entry (highest round)
        latest: Dict[str, Any] = {}
        for e in entries:
            persona = getattr(e, "persona", None)
            if persona is None:
                continue
            cur = latest.get(persona)
            if cur is None or getattr(e, "round", 0) > getattr(cur, "round", 0):
                latest[persona] = e
        out: List[Dict[str, Any]] = []
        for persona, entry in latest.items():
            out.append({
                "persona": persona,
                "title": getattr(entry, "persona_title", persona),
                "stance": getattr(entry, "stance", "neutral"),
                "one_liner": getattr(entry, "one_liner", ""),
                "confidence": getattr(entry, "confidence", "low"),
                "round": getattr(entry, "round", 0),
            })
        return out

    @staticmethod
    def _consensus_round(scratchpad: Any) -> Optional[int]:
        """Round at which all personas converged on the same stance.

        ``None`` if the panel never converged (still valid output —
        divergence is itself a useful signal).
        """
        if scratchpad is None:
            return None
        try:
            entries = list(getattr(scratchpad, "entries", []) or [])
        except Exception:
            return None
        if not entries:
            return None
        rounds = sorted({getattr(e, "round", 0) for e in entries})
        for r in rounds:
            stances = {
                getattr(e, "stance", None)
                for e in entries
                if getattr(e, "round", 0) == r
            }
            stances.discard(None)
            if len(stances) == 1:
                return r
        return None


__all__ = [
    "PanelScopedAgent",
]
