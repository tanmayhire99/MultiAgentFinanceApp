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

Parallel execution
------------------
Independent ready steps (those whose ``depends_on`` are all satisfied)
are dispatched concurrently via ``asyncio.gather``. Steps that share
no dependency edge run in parallel; steps that depend on prior results
wait for those results before starting. Event streams from concurrent
steps are multiplexed into the single output ``AsyncIterator`` in
arrival order.

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
* If a step exceeds the per-step wall-clock timeout
  (``FINAI_STEP_TIMEOUT_S``, default 180s), it is cancelled and
  committed as a ``failed`` StepResult (``error_type="StepTimeout"``)
  so one wedged agent can't block the whole pipeline.
* If the executor itself crashes (programming bug), the exception
  propagates to the pipeline.

The executor does NOT call the joiner. After all steps reach a
terminal state it returns. The pipeline is responsible for then
running the synthesizer step or the joiner replan loop.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.agents._base import ScopedAgentError
from src.core.agents.factory_dispatch import build_scoped_agent_for_step
from src.core.panel import PanelEvent
from src.core.types import Plan, PlanStep, Scratchpad, StepResult


log = logging.getLogger("finai.executor")


# Default per-step recursion limit that propagates into ScopedAgent.
DEFAULT_STEP_RECURSION_LIMIT = 25

# Per-step wall-clock safety timeout (seconds). A stuck agent (hung MCP
# tool, wedged LLM stream) would otherwise block the whole pipeline
# indefinitely. On timeout the step is committed as a ``failed``
# StepResult so the DAG walk skips its descendants and the joiner can
# replan. Override via ``FINAI_STEP_TIMEOUT_S``; set <= 0 to disable.
DEFAULT_STEP_TIMEOUT_S = 180.0

# Sentinel so _run_one_step can tell "caller passed None to disable" apart
# from "caller didn't specify → read the env default".
_UNSET = object()


def _default_step_timeout_s() -> Optional[float]:
    """Resolve the per-step timeout from ``FINAI_STEP_TIMEOUT_S`` (or default).

    Returns ``None`` (disabled) when the env var parses to <= 0.
    """
    raw = os.environ.get("FINAI_STEP_TIMEOUT_S", "").strip()
    if not raw:
        return DEFAULT_STEP_TIMEOUT_S
    try:
        val = float(raw)
    except ValueError:
        log.warning("Invalid FINAI_STEP_TIMEOUT_S=%r; using default", raw)
        return DEFAULT_STEP_TIMEOUT_S
    return val if val > 0 else None


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
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Run ``plan`` step-by-step against the shared ``scratchpad``.

    Yields ``_status`` and ``error`` PanelEvents the pipeline can
    forward to the user. ``StepResult`` objects are appended to the
    scratchpad as side-effects (the scratchpad is the caller's
    accumulator; the executor mutates it but doesn't return it).

    Independent ready steps are dispatched concurrently via
    ``asyncio.gather``; their event streams are multiplexed into
    the single output iterator in arrival order.
    """
    total_steps = len(plan.steps)
    started = time.time()
    yield _status(
        f"Plan: {total_steps} step(s) targeting agents "
        f"{sorted(plan.all_agents())}"
    )

    iterations = 0
    max_iterations = total_steps + 5
    while iterations < max_iterations:
        iterations += 1
        terminal = scratchpad.terminal_ids()
        if len(terminal) >= total_steps:
            break

        completed = scratchpad.completed_ids()
        ready = [
            s for s in plan.steps
            if s.id not in terminal
            and all(dep in completed for dep in s.depends_on)
        ]

        if not ready:
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

        if len(ready) == 1:
            async for ev in _run_one_step(
                step=ready[0],
                scratchpad=scratchpad,
                intent_flags=intent_flags,
                all_mcp_tools=all_mcp_tools,
                registry=registry,
                recursion_limit=recursion_limit,
                user_id=user_id,
            ):
                yield ev
        else:
            async for ev in _run_parallel_steps(
                steps=ready,
                scratchpad=scratchpad,
                intent_flags=intent_flags,
                all_mcp_tools=all_mcp_tools,
                registry=registry,
                recursion_limit=recursion_limit,
                user_id=user_id,
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

        # The walker uses ``terminal_ids`` (complete ∪ failed ∪ skipped)
        # so descendants of a failed step are NOT considered "ready"
        # forever. They get skipped explicitly below.
        iterations = 0
        max_iterations = total_steps + 5 # safety: prevent infinite loops
        while iterations < max_iterations:
            iterations += 1
            terminal = scratchpad.terminal_ids()
            if len(terminal) >= total_steps:
                break

            # A step is RUNNABLE iff:
            # - it has not already reached a terminal state, AND
            # - every one of its declared deps is in COMPLETED (not just
            # terminal — a failed dep means the step must be skipped,
            # not run).
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
                user_id=user_id,
            ):
                yield ev
        else:
            async for ev in _run_parallel_steps(
                steps=ready,
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
# Parallel step runner
# ---------------------------------------------------------------------------
async def _run_parallel_steps(
    *,
    steps: List[PlanStep],
    scratchpad: Scratchpad,
    intent_flags: Dict[str, bool],
    all_mcp_tools: Sequence[BaseTool],
    registry: AgentRegistry,
    recursion_limit: int,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Run multiple independent steps concurrently, multiplexing events.

    Each step's event stream is collected into a per-step list, then
    yielded in step-id order after all steps finish. This ensures
    deterministic output ordering while getting the latency benefit
    of concurrent execution.

    Steps that fail do not block others — all steps run to completion
    (or failure) regardless of sibling outcomes.
    """
    step_events: Dict[int, List[PanelEvent]] = {s.id: [] for s in steps}

    async def _collect_step(step: PlanStep) -> None:
        async for ev in _run_one_step(
            step=step,
            scratchpad=scratchpad,
            intent_flags=intent_flags,
            all_mcp_tools=all_mcp_tools,
            registry=registry,
            recursion_limit=recursion_limit,
            user_id=user_id,
        ):
            step_events[step.id].append(ev)

    await asyncio.gather(
        *(_collect_step(s) for s in steps),
        return_exceptions=False,
    )

    for step in steps:
        for ev in step_events[step.id]:
            yield ev


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
    user_id: str = "demo",
    step_timeout_s: Any = _UNSET,
) -> AsyncIterator[PanelEvent]:
    """Build the step's ScopedAgent, run it (streaming), commit the result.

    Yields ``_status`` events for the step's start / end (so the
    pipeline can render them in the chat), ``error`` events for
    construction failures, and forwards all intermediate events
    (``step_content``, ``step_tool_call``, ``step_tool_result``)
    from the agent's streaming ``run()`` method so the user sees
    live progress per step.

    The agent's terminal ``_step_result`` event is consumed here
    (not yielded) — its ``StepResult`` is committed to the
    scratchpad and a summary status line is emitted instead.
    """
    desc_short = step.description.strip().split("\n", 1)[0][:80]
    yield _status(f"Step {step.id}: {step.agent} — {desc_short}...")

    try:
        agent = build_scoped_agent_for_step(
            step=step,
            scratchpad=scratchpad,
            all_mcp_tools=all_mcp_tools,
            intent_flags=intent_flags,
            registry=registry,
            recursion_limit=recursion_limit,
            user_id=user_id,
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

    # Resolve the per-step safety timeout (env default unless caller overrides).
    timeout_s = (
        _default_step_timeout_s() if step_timeout_s is _UNSET else step_timeout_s
    )
    step_started = time.time()

    # The agent's run() is now an AsyncIterator yielding
    # PanelEvents, with a terminal _step_result event.
    result: Optional[StepResult] = None

    async def _drive() -> AsyncIterator[PanelEvent]:
        nonlocal result
        async for ev in agent.run():
            etype = ev.get("type")

            if etype == "_step_result":
                result = ev.get("result")
                continue

            # Forward step-level content events upstream so the
            # pipeline / planner_pipeline can render them in the
            # chat as they arrive.
            if etype in ("step_content", "step_tool_call", "step_tool_result"):
                yield ev
                continue

            # Other events (e.g. from PanelScopedAgent forwarding
            # debate events like "header", "text", "tool_call",
            # "tool_result", "persona_verdict") pass through as-is.
            if etype not in ("_status", "error"):
                yield ev

    try:
        if timeout_s is not None:
            async with asyncio.timeout(timeout_s):
                async for ev in _drive():
                    yield ev
        else:
            async for ev in _drive():
                yield ev
    except asyncio.TimeoutError:
        log.warning(
            "Step %d (%s) exceeded %.0fs timeout — marking failed",
            step.id, step.agent, timeout_s or 0,
        )
        result = StepResult(
            step_id=step.id,
            status="failed",
            output=None,
            error=f"step exceeded {timeout_s:.0f}s timeout",
            error_type="StepTimeout",
            started_at=step_started,
            completed_at=time.time(),
        )
    except Exception as e:
        log.exception("Unexpected exception running step %d", step.id)
        result = StepResult(
            step_id=step.id,
            status="failed",
            output=None,
            error=str(e),
            error_type=type(e).__name__,
            started_at=step_started,
            completed_at=time.time(),
        )

    if result is None:
        # Agent didn't yield _step_result — treat as failed
        log.error("Step %d agent.run() completed without _step_result", step.id)
        result = StepResult(
            step_id=step.id,
            status="failed",
            output=None,
            error="agent.run() did not yield _step_result",
            error_type="ExecutorError",
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
