"""Pipeline — query → plan → execute → join → (replan loop) → stream report.

This is the top-level coroutine that wires the phases of the
planner-first architecture together:

1. **Phase 1** (already done by the dispatcher) — classifier emits
   ``intent_flags``.
2. **Phase 2** — :func:`src.core.planner.plan` produces a validated
   :class:`~src.core.types.Plan`.
3. **Phase 3** — :func:`src.core.executor.execute` runs each step
   via :class:`~src.core.agents._base.ScopedAgent`, populating the
   :class:`~src.core.types.Scratchpad`.
4. **Phase 4** — :func:`src.core.joiner.decide` inspects the scratchpad
   and returns a :class:`~src.core.types.JoinDecision`. If the action
   is ``replan``, additional steps are merged into the plan and Phase 3
   re-enters. If ``finish``, we surface the synthesizer output. If
   ``abort``, we emit an honest error.
5. **Phase 5** — the synthesizer step's ``StepResult.output["text"]``
   is the user-visible markdown report. The pipeline streams it back
   in :class:`PanelEvent` form so the dispatcher / SSE renderer
   handle it identically to the static flows.

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
from src.core.joiner import decide as joiner_decide
from src.core.panel import PanelEvent
from src.core.planner import PlannerError, plan as build_plan
from src.core.types import ExecutionState, Plan, Scratchpad, StepResult
from src.core.verify_numbers import verify_numbers

log = logging.getLogger("finai.pipeline")


DEFAULT_PLANNER_TIMEOUT = 45.0


async def run_pipeline(
    query: str,
    *,
    intent_flags: Dict[str, bool],
    all_mcp_tools: Sequence[BaseTool],
    registry: AgentRegistry = REGISTRY,
    history_summary: Optional[str] = None,
    max_replans: int = 2,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """End-to-end planner-first pipeline for ``query``.

    Yields ``_status`` lines + the synthesizer's streamed markdown +
    any ``error`` events. The dispatcher routes ``_status`` events
    into chat (or skips them when verbose_trace is off — though
    these are user-relevant progress, not dev-trace, so they always
    show).

    After each executor run the joiner decides whether the results are
    sufficient (``finish``), gaps need filling (``replan``), or we
    should bail out (``abort``). On ``replan``, the additional steps
    are merged into the plan and the executor re-enters Phase 3.

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

    # 2) Phase 3 + 4: Execute then join, looping on replan
    scratchpad = Scratchpad(query=query)
    state = ExecutionState(
        query=query,
        plan=plan_obj,
        scratchpad=scratchpad,
        max_replans=max_replans,
        user_id=user_id,
    )

    while True:
        async for ev in execute_plan(
            plan=state.plan,
            scratchpad=state.scratchpad,
            intent_flags=intent_flags,
            all_mcp_tools=all_mcp_tools,
            registry=registry,
        ):
            yield ev

        decision = joiner_decide(state)

        if decision.action == "finish":
            break

        if decision.action == "abort":
            yield _status(f"Joiner: {decision.reasoning}")
            yield _err(decision.reasoning)
            return

        if decision.action == "replan":
            yield _status(f"Joiner: {decision.reasoning}")
            for ns in decision.additional_steps:
                state.plan.steps.append(ns)
            state.replan_count += 1
            continue

    # 3) Phase 5: Surface the synthesizer's output.
    synth_step = _find_synth_step(state.plan)
    if synth_step is None:
        yield _err(
            "Plan had no synthesizer step — the planner produced an "
            "incomplete DAG (this is a planner bug; the system prompt "
            "explicitly requires a synthesizer step)."
        )
        return

    synth_result = state.scratchpad.get(synth_step.id)
    if synth_result is None:
        yield _err(
            f"Synthesizer step {synth_step.id} never ran — "
            "executor returned without producing a result."
        )
        return

    if synth_result.status != "complete":
        partial = (synth_result.output or {}).get("text", "") if isinstance(
            synth_result.output, dict
        ) else ""
        if partial:
            yield {"type": "text", "text": partial, "persona": "synthesizer"}
        yield _err(
            f"Synthesizer step finished with status="
            f"{synth_result.status}: {synth_result.error or '(no error)'}. "
            "The report above (if any) is partial."
        )
        return

    final_text = ""
    output = synth_result.output
    if isinstance(output, dict):
        final_text = str(output.get("text", "") or "")
    elif isinstance(output, str):
        final_text = output

    if not final_text:
        yield _err(
            "Synthesizer step completed but produced empty text. "
            "Check the synthesizer system prompt and the deps it "
            f"received: {synth_step.depends_on}"
        )
        return

    # Phase 5.5: verify numerical claims against scratchpad data
    try:
        verify_result = verify_numbers(final_text, scratchpad)
        if verify_result.has_flags:
            log.warning(
                "verify_numbers: %d flagged claims in synthesizer output",
                verify_result.flagged_count,
            )
        yield _status(
            f"Number verification: {verify_result.verified_count} ✓, "
            f"{verify_result.unverified_count} unverified, "
            f"{verify_result.flagged_count} ✗"
        )
        final_text = verify_result.annotated_text
    except Exception:
        log.exception("verify_numbers failed; using unverified text")

    yield {"type": "text", "text": final_text, "persona": "synthesizer"}

    duration = time.time() - started
    replan_info = ""
    if state.replan_count > 0:
        replan_info = f" ({state.replan_count} replan(s))"
    yield _status(
        f"Pipeline complete: {len(state.plan.steps)} step(s) in "
        f"{duration:.1f}s{replan_info}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _status(text: str) -> PanelEvent:
    return {"type": "_status", "text": text}


def _err(text: str) -> PanelEvent:
    return {"type": "error", "text": text}


def _find_synth_step(plan: Plan) -> Optional[Any]:
    """Return the synthesizer step (last one if multiple, by step id)."""
    candidates = [s for s in plan.steps if s.agent == "synthesizer"]
    if not candidates:
        return None
    return sorted(candidates, key=lambda s: -s.id)[0]


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
