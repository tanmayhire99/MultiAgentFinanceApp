"""Pipeline — query → plan → execute → stream user-visible report.

This is the top-level coroutine that wires the Phase 1 / 2 / 3 / 5
pieces of the planner-first architecture together:

1. **Phase 1** (already done by the dispatcher) — classifier emits
   ``intent_flags``.
2. **Phase 2** — :func:`src.core.planner.plan` produces a validated
   :class:`~src.core.types.Plan`.
3. **Phase 3** — :func:`src.core.executor.execute` runs each step
   via :class:`~src.core.agents._base.ScopedAgent`, populating the
   :class:`~src.core.types.Scratchpad`.
4. **Phase 5** — the terminal step's report reaches the user. For a
   ``synthesizer`` terminal that's its buffered
   ``StepResult.output["text"]``, streamed back in :class:`PanelEvent`
   form. For a ``panel_agent`` terminal (panel debates are
   self-contained) the debate + closing brief already streamed live
   during Phase 3, so Phase 5 emits nothing extra.

We deliberately skip Phase 4 (the joiner / replan) in this v0. If
the synthesizer step succeeded, we emit its output. If it failed
or was skipped, we emit an honest error message. Replanning lives
in the next iteration; the architecture supports it (every step
result is in the scratchpad with ``status`` and ``error``), we just
haven't wired the replan loop yet.

PanelEvent shape
----------------
The pipeline yields the same events the static flows do, so the
dispatcher's filtering (verbose_trace / artifact wrapper / disclaimer)
applies uniformly:

* ``_status`` events — chat-visible italic progress lines
* ``text`` events — the synthesizer's streamed markdown
* ``error`` events — construction or run failures
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.executor import execute as execute_plan
from src.core.panel import PanelEvent
from src.core.planner import PlannerError, plan as build_plan
from src.core.types import Plan, Scratchpad, StepResult


log = logging.getLogger("finai.pipeline")


# Default budget for the planner LLM call. Real models on NIM finish
# well under this even with retries; the cap is so a stuck request
# doesn't hold up the whole pipeline.
DEFAULT_PLANNER_TIMEOUT = 45.0


async def run_pipeline(
    query: str,
    *,
    intent_flags: Dict[str, bool],
    all_mcp_tools: Sequence[BaseTool],
    registry: AgentRegistry = REGISTRY,
    history_summary: Optional[str] = None,
) -> AsyncIterator[PanelEvent]:
    """End-to-end planner-first pipeline for ``query``.

    Yields ``_status`` lines + the synthesizer's streamed markdown +
    any ``error`` events. The dispatcher routes ``_status`` events
    into chat (or skips them when verbose_trace is off — though
    these are user-relevant progress, not dev-trace, so they always
    show).

    Caller responsibilities:

    * The dispatcher already classified the query and built
      ``intent_flags``.
    * The dispatcher already loaded ``all_mcp_tools`` from
      :func:`src.config.mcp_servers.get_tools`.
    * The dispatcher decides whether to wrap the output in an
      artifact (Fix 3 ``wants_artifact`` plumbing); the pipeline
      just yields the events.
    """
    yield _status(f"Planning a multi-agent investigation for: _{query.strip()}_")
    started = time.time()

    # 1) Phase 2: Plan
    try:
        plan_obj: Plan = await build_plan(
            query,
            intent_flags=intent_flags,
            registry=registry,
            history_summary=history_summary,
            timeout_seconds=DEFAULT_PLANNER_TIMEOUT,
        )
    except PlannerError as e:
        log.exception("Planner failed: %s", e)
        yield _err(
            f"Planner failed: {e}. "
            "Falling back to the deterministic flow path is not yet "
            "wired in this slice — re-run with a different phrasing "
            "or set FINAI_PLANNER_ENABLED=0 to use the static flows."
        )
        return

    yield _status(
        f"Plan ready: {len(plan_obj.steps)} step(s) "
        f"({_describe_plan_brief(plan_obj)})"
    )

    # 2) Phase 3: Execute
    scratchpad = Scratchpad(query=query)
    async for ev in execute_plan(
        plan=plan_obj,
        scratchpad=scratchpad,
        intent_flags=intent_flags,
        all_mcp_tools=all_mcp_tools,
        registry=registry,
    ):
        yield ev

    # 3) Phase 5: surface the user-visible report.
    #    Normally the terminal ``synthesizer`` step's output is the
    #    report. Panel plans are different: ``panel_agent`` is the
    #    terminal step and ALREADY streamed its debate + closing brief
    #    live during execution (PanelScopedAgent.run_streaming), so
    #    there is nothing to re-emit here — doing so would duplicate the
    #    entire transcript.
    report_step, report_kind = _find_report_step(plan_obj)
    if report_step is None:
        yield _err(
            "Plan had no synthesizer or panel_agent step — the planner "
            "produced an incomplete DAG (the final step must be a "
            "synthesizer, or a panel_agent for panel debates)."
        )
        return

    report_result = scratchpad.get(report_step.id)
    if report_result is None:
        yield _err(
            f"Terminal step {report_step.id} ({report_step.agent}) never "
            "ran — executor returned without producing a result."
        )
        return

    if report_result.status != "complete":
        # Surface what we have. The panel path already streamed its
        # (partial) transcript live, so only the synthesizer path has
        # un-emitted buffered text to surface here.
        partial = (report_result.output or {}).get("text", "") if isinstance(
            report_result.output, dict
        ) else ""
        if partial and report_kind == "synthesizer":
            yield {"type": "text", "text": partial, "persona": "synthesizer"}
        yield _err(
            f"{report_step.agent.capitalize()} step finished with status="
            f"{report_result.status}: {report_result.error or '(no error)'}. "
            "The report above (if any) is partial."
        )
        return

    if report_kind == "synthesizer":
        # Happy path: emit the synthesizer's buffered text. The
        # dispatcher decides whether to wrap it in :::artifact{}:::
        # based on the decision["wants_artifact"] flag the user set.
        final_text = ""
        output = report_result.output
        if isinstance(output, dict):
            final_text = str(output.get("text", "") or "")
        elif isinstance(output, str):
            final_text = output

        if not final_text:
            yield _err(
                "Synthesizer step completed but produced empty text. "
                "Check the synthesizer system prompt and the deps it "
                f"received: {report_step.depends_on}"
            )
            return

        yield {"type": "text", "text": final_text, "persona": "synthesizer"}
    # else: report_kind == "panel" — the panel_agent streamed its debate
    # and closing brief live during Phase 3, so the report is already in
    # the user's stream. Nothing to emit.

    # Closing telemetry
    duration = time.time() - started
    yield _status(
        f"Pipeline complete: {len(plan_obj.steps)} step(s) in "
        f"{duration:.1f}s"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _status(text: str) -> PanelEvent:
    return {"type": "_status", "text": text}


def _err(text: str) -> PanelEvent:
    return {"type": "error", "text": text}


def _find_report_step(plan: Plan) -> tuple[Optional[Any], Optional[str]]:
    """Return the ``(step, kind)`` that produces the user-visible report.

    Prefers a ``synthesizer`` step (the normal terminal, ``kind ==
    "synthesizer"``). If the plan has no synthesizer but ends with a
    ``panel_agent`` (panel debates are self-contained — the panel
    writes its own debate + closing brief), that panel step is the
    report (``kind == "panel"``). Returns ``(None, None)`` if the plan
    has neither, which the pipeline surfaces as an error.

    When multiple candidates of the chosen kind exist, the highest
    step id wins (by convention the report step is last).
    """
    synth = [s for s in plan.steps if s.agent == "synthesizer"]
    if synth:
        return sorted(synth, key=lambda s: -s.id)[0], "synthesizer"
    panel = [s for s in plan.steps if s.agent == "panel_agent"]
    if panel:
        return sorted(panel, key=lambda s: -s.id)[0], "panel"
    return None, None


def _describe_plan_brief(plan: Plan) -> str:
    """One-line description of the plan shape for status emit."""
    agent_seq = " → ".join(s.agent for s in plan.steps)
    if len(agent_seq) > 80:
        agent_seq = agent_seq[:77] + "..."
    return agent_seq


__all__ = [
    "DEFAULT_PLANNER_TIMEOUT",
    "run_pipeline",
]
