"""DAG executor — runs a :class:`Plan` step-by-step, populating the scratchpad.

This is **Phase 3** of the planner-first pipeline (Phase 1 = classifier,
Phase 2 = planner). The executor:

1. Iterates the plan's ready steps (those whose ``depends_on`` are all
   complete) until every step is in a terminal state.
2. For each ready step, builds a :class:`ScopedAgent` via
   :func:`build_scoped_agent_for_step`, runs it, and writes the
   resulting :class:`StepResult` to the shared :class:`Scratchpad`.
3. Yields :class:`PanelEvent`-shaped status events the pipeline can
   forward to the user as live progress.

Sequential v0
-------------
The first version runs steps **sequentially** in topological order
(no ``asyncio.gather``). For the claim-tracker slice this is fine —
the typical plan has 5 steps and most have data dependencies on
prior steps anyway. Parallel execution of independent ready steps
is a Day-7 enhancement; the entry point's signature is parallel-
ready so the upgrade is a swap, not a rewrite.

Failure handling
----------------
* If a step's ScopedAgent construction fails (ScopedAgentError —
  e.g. policy gate violation, unknown agent, tool not owned), we
  emit a structured error event and write a ``failed`` StepResult
  to the scratchpad. The DAG walk continues with whatever ready
  steps remain — descendants that depend on the failed step end
  up as ``skipped`` (not run, not retried).
* If the agent's ReAct loop crashes, ScopedAgent.run already
  catches the exception and returns ``StepResult(status="failed")``.
  We forward that result to the scratchpad and continue.
* If the executor itself crashes (programming bug), the exception
  propagates to the pipeline.

The executor does NOT call the joiner. After all steps reach a
terminal state it returns. The pipeline is responsible for then
running the synthesizer step or the joiner replan loop.
"""
from __future__ import annotations

import logging
import time
from typing import AsyncIterator, Dict, List, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.agents._base import ScopedAgentError
from src.core.agents._factories import build_scoped_agent_for_step
from src.core.panel import PanelEvent
from src.core.types import Plan, PlanStep, Scratchpad, StepResult


log = logging.getLogger("finai.executor")


# Default per-step recursion limit that propagates into ScopedAgent.
# Per-step in seconds is unenforced for now — long-running steps
# (e.g. claim_agent making per-claim LLM calls) just take their time.
DEFAULT_STEP_RECURSION_LIMIT = 25


# ---------------------------------------------------------------------------
# Helpers — short status-line emitters for the pipeline
# ---------------------------------------------------------------------------
def _status(text: str) -> PanelEvent:
    """Brief italic status line (chat-pane visible in the pipeline)."""
    return {
        "type": "_status",
        "text": text,
    }


def _err(text: str) -> PanelEvent:
    return {"type": "error", "text": text}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def execute(
    *,
    plan: Plan,
    scratchpad: Scratchpad,
    intent_flags: Dict[str, bool],
    all_mcp_tools: Sequence[BaseTool],
    registry: AgentRegistry = REGISTRY,
    recursion_limit: int = DEFAULT_STEP_RECURSION_LIMIT,
) -> AsyncIterator[PanelEvent]:
    """Run ``plan`` step-by-step against the shared ``scratchpad``.

    Yields ``_status`` and ``error`` PanelEvents the pipeline can
    forward to the user. ``StepResult`` objects are appended to the
    scratchpad as side-effects (the scratchpad is the caller's
    accumulator; the executor mutates it but doesn't return it).

    Sequential v0: runs ready steps one at a time in topological
    order. Independent ready steps are NOT yet parallelised.
    """
    total_steps = len(plan.steps)
    started = time.time()
    yield _status(
        f"Plan: {total_steps} step(s) targeting agents "
        f"{sorted(plan.all_agents())}"
    )

    # The walker uses ``terminal_ids`` (complete ∪ failed ∪ skipped)
    # so descendants of a failed step are NOT considered "ready"
    # forever. They get skipped explicitly below.
    iterations = 0
    max_iterations = total_steps + 5  # safety: prevent infinite loops
    while iterations < max_iterations:
        iterations += 1
        terminal = scratchpad.terminal_ids()
        if len(terminal) >= total_steps:
            break

        # A step is RUNNABLE iff:
        #   - it has not already reached a terminal state, AND
        #   - every one of its declared deps is in COMPLETED (not just
        #     terminal — a failed dep means the step must be skipped,
        #     not run).
        completed = scratchpad.completed_ids()
        ready = [
            s for s in plan.steps
            if s.id not in terminal
            and all(dep in completed for dep in s.depends_on)
        ]

        if not ready:
            # No runnable step but non-terminal steps remain — that
            # means every remaining step has at least one failed or
            # skipped ancestor. Mark them all skipped and exit.
            remaining = [s for s in plan.steps if s.id not in terminal]
            for s in remaining:
                yield _status(
                    f"Step {s.id} ({s.agent}) skipped — depends on "
                    f"a failed/skipped ancestor"
                )
                scratchpad.add(StepResult(
                    step_id=s.id,
                    status="skipped",
                    output=None,
                    started_at=time.time(),
                    completed_at=time.time(),
                ))
            break

        # Sequential: take the FIRST runnable step. ``plan.steps`` is
        # in topological order (Plan validation ensures this), so the
        # first runnable step is always the earliest one we can do.
        # Day-7 will swap this for asyncio.gather over the full
        # ``ready`` list.
        step = ready[0]
        async for ev in _run_one_step(
            step=step,
            scratchpad=scratchpad,
            intent_flags=intent_flags,
            all_mcp_tools=all_mcp_tools,
            registry=registry,
            recursion_limit=recursion_limit,
        ):
            yield ev

    duration = time.time() - started
    completed = len(scratchpad.completed_ids())
    failed = sum(
        1 for r in scratchpad.results.values() if r.status == "failed"
    )
    skipped = sum(
        1 for r in scratchpad.results.values() if r.status == "skipped"
    )
    yield _status(
        f"Plan execution complete: {completed} ok, {failed} failed, "
        f"{skipped} skipped, in {duration:.1f}s"
    )


# ---------------------------------------------------------------------------
# Per-step runner
# ---------------------------------------------------------------------------
async def _run_one_step(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    intent_flags: Dict[str, bool],
    all_mcp_tools: Sequence[BaseTool],
    registry: AgentRegistry,
    recursion_limit: int,
) -> AsyncIterator[PanelEvent]:
    """Build the step's ScopedAgent, run it, commit the result.

    Yields ``_status`` events for the step's start / end (so the
    pipeline can render them in the chat) and ``error`` events for
    construction failures. The step's actual ReAct trajectory is
    NOT yielded by this function in v0 - just the summary of the
    final StepResult.
    """
    desc_short = step.description.strip().split("\n", 1)[0][:80]
    yield _status(f"Step {step.id}: {step.agent} — {desc_short}...")

    # Build the ScopedAgent. ScopedAgentError is the only construction
    # failure mode we expect (registry / policy-gate / tool-owner
    # violations). Anything else propagates.
    try:
        agent = build_scoped_agent_for_step(
            step=step,
            scratchpad=scratchpad,
            all_mcp_tools=all_mcp_tools,
            intent_flags=intent_flags,
            registry=registry,
            recursion_limit=recursion_limit,
        )
    except ScopedAgentError as e:
        yield _err(f"Step {step.id} construction failed: {e}")
        scratchpad.add(StepResult(
            step_id=step.id,
            status="failed",
            output=None,
            error=str(e),
            error_type="ScopedAgentError",
            started_at=time.time(),
            completed_at=time.time(),
        ))
        return

    # Run the ScopedAgent. Its run() catches ReAct-loop exceptions
    # internally and returns a failed StepResult — so a raise here
    # would be a programming error, not a tool failure.
    try:
        result = await agent.run()
    except Exception as e:
        log.exception("Unexpected exception running step %d", step.id)
        result = StepResult(
            step_id=step.id,
            status="failed",
            output=None,
            error=str(e),
            error_type=type(e).__name__,
            started_at=time.time(),
            completed_at=time.time(),
        )

    scratchpad.add(result)

    if result.status == "complete":
        tool_count = len(result.tools_used or [])
        yield _status(
            f"Step {step.id} ✓ ({result.duration_s or 0:.1f}s, "
            f"{tool_count} tool call{'s' if tool_count != 1 else ''})"
        )
    elif result.status == "failed":
        yield _status(
            f"Step {step.id} ✗ failed: "
            f"{(result.error or 'unknown error')[:120]}"
        )
    else:
        yield _status(f"Step {step.id} status={result.status}")


__all__ = [
    "DEFAULT_STEP_RECURSION_LIMIT",
    "execute",
]
