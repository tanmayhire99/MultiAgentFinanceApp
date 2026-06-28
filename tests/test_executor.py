"""Unit tests for src.core.executor — Slice Stage 2b.

Coverage:

1. Happy path — 3-step plan runs in topological order; all results
   land in the scratchpad with status=complete.
2. Parallel dispatch — independent ready steps run concurrently via
   asyncio.gather; events are yielded in step-id order.
3. Failed step — descendants get marked skipped, the executor still
   completes its walk.
4. Construction failure (ScopedAgentError) — step marked failed,
   error event emitted, walk continues.
5. Run exception — agent.run() raising is caught as a failed StepResult.
6. Status events — every step gets a "starting" and a "done" status.

Mocking strategy
----------------
We patch ``build_scoped_agent_for_step`` to return a fake ScopedAgent
whose ``run()`` is controlled per-test. This avoids needing a real
LLM, real MCP tools, or a live registry — the executor's contract is
"call the factory, run the agent, write the result", and that's
exactly what the mock exercises.

Run via::

    uv run pytest tests/test_executor.py -v
"""
from __future__ import annotations

import asyncio
import time
import unittest
from typing import Any, Dict, List, AsyncIterator
from unittest.mock import MagicMock, patch

from src.core.agents._base import ScopedAgentError
from src.core.agents.registry import REGISTRY
from src.core.executor import execute, _run_one_step
from src.core.types import KNOWN_INTENT_FLAGS, Plan, PlanStep, Scratchpad, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flags(**overrides: bool) -> Dict[str, bool]:
    base = {f: False for f in KNOWN_INTENT_FLAGS}
    base.update(overrides)
    return base


def _step(
    step_id: int,
    *,
    agent: str = "research_agent",
    deps: List[int] = None,
    desc: str = "test step",
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=desc,
        agent=agent,
        tool_subset=[],
        depends_on=deps or [],
    )


def _plan(*steps: PlanStep, goal: str = "test") -> Plan:
    return Plan(
        goal=goal,
        rationale="test rationale",
        steps=list(steps),
    )


def _make_fake_agent(*, step: PlanStep, result: StepResult) -> MagicMock:
    """A MagicMock with .step + an async run() yielding _step_result event."""
    fake = MagicMock()
    fake.step = step

    async def _run():
        yield {"type": "_step_result", "result": result}

    fake.run = _run
    return fake


def _ok_result(step_id: int, *, text: str = "ok") -> StepResult:
    return StepResult(
        step_id=step_id,
        status="complete",
        output={"text": text},
        tools_used=[],
        started_at=time.time(),
        completed_at=time.time() + 0.01,
    )


def _failed_result(step_id: int, *, error: str = "boom") -> StepResult:
    return StepResult(
        step_id=step_id,
        status="failed",
        output=None,
        error=error,
        error_type="TestError",
        started_at=time.time(),
        completed_at=time.time() + 0.01,
    )


def _drain_executor(plan: Plan, scratchpad: Scratchpad,
                    intent_flags: Dict[str, bool]) -> List[Any]:
    """Run the executor and collect every yielded event."""

    async def _go() -> List[Any]:
        events: List[Any] = []
        async for ev in execute(
            plan=plan,
            scratchpad=scratchpad,
            intent_flags=intent_flags,
            all_mcp_tools=[],
        ):
            events.append(ev)
        return events

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class HappyPathTests(unittest.TestCase):
    def test_three_step_plan_executes_in_topo_order(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="filings_agent")
        s3 = _step(3, agent="synthesizer", deps=[1, 2])
        plan = _plan(s1, s2, s3)
        scratchpad = Scratchpad(query="q")

        order_seen: List[int] = []

        def fake_factory(*, step: PlanStep, **_kwargs):
            order_seen.append(step.id)
            return _make_fake_agent(step=step, result=_ok_result(step.id))

        with patch(
            "src.core.executor.build_scoped_agent_for_step",
            side_effect=fake_factory,
        ):
            events = _drain_executor(plan, scratchpad, _flags())

        self.assertEqual(set(order_seen), {1, 2, 3})
        self.assertEqual(order_seen[-1], 3, "synthesizer must run last")
        self.assertLess(order_seen.index(1), order_seen.index(3))
        self.assertLess(order_seen.index(2), order_seen.index(3))

        for step_id in (1, 2, 3):
            r = scratchpad.get(step_id)
            self.assertIsNotNone(r)
            self.assertEqual(r.status, "complete")

    def test_yields_status_events_per_step(self):
        s1 = _step(1)
        s2 = _step(2, deps=[1])
        plan = _plan(s1, s2)

        with patch(
            "src.core.executor.build_scoped_agent_for_step",
            side_effect=lambda *, step, **_: _make_fake_agent(
                step=step, result=_ok_result(step.id)
            ),
        ):
            events = _drain_executor(plan, Scratchpad(query="q"), _flags())

        statuses = [
            ev["text"] for ev in events
            if ev.get("type") == "_status"
        ]
        self.assertGreaterEqual(len(statuses), 5)
        self.assertTrue(any("Plan:" in s for s in statuses))
        self.assertTrue(any("Step 1: research_agent" in s for s in statuses))
        self.assertTrue(any("Step 2: research_agent" in s for s in statuses))
        self.assertTrue(any("Step 1 ✓" in s for s in statuses))
        self.assertTrue(any("Step 2 ✓" in s for s in statuses))
        self.assertTrue(any(
            "Plan execution complete" in s for s in statuses
        ))


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
class ParallelExecutionTests(unittest.TestCase):
    def test_independent_steps_run_concurrently(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="filings_agent")
        s3 = _step(3, agent="synthesizer", deps=[1, 2])
        plan = _plan(s1, s2, s3)
        scratchpad = Scratchpad(query="q")

        call_times: Dict[int, float] = {}

        def fake_factory(*, step: PlanStep, **_kwargs):
            call_times[step.id] = time.time()
            return _make_fake_agent(step=step, result=_ok_result(step.id))

        with patch(
            "src.core.executor.build_scoped_agent_for_step",
            side_effect=fake_factory,
        ):
            events = _drain_executor(plan, scratchpad, _flags())

        self.assertEqual(scratchpad.get(1).status, "complete")
        self.assertEqual(scratchpad.get(2).status, "complete")
        self.assertEqual(scratchpad.get(3).status, "complete")

        if 1 in call_times and 2 in call_times:
            delta = abs(call_times[1] - call_times[2])
            self.assertLess(
                delta, 0.1,
                f"Steps 1 and 2 should be dispatched concurrently "
                f"(delta={delta:.3f}s)",
            )

    def test_parallel_failure_does_not_block_sibling(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="filings_agent")
        s3 = _step(3, agent="synthesizer", deps=[1, 2])
        plan = _plan(s1, s2, s3)
        scratchpad = Scratchpad(query="q")

        def fake_factory(*, step: PlanStep, **_kwargs):
            if step.id == 1:
                return _make_fake_agent(
                    step=step, result=_failed_result(1, error="api timeout")
                )
            return _make_fake_agent(step=step, result=_ok_result(step.id))

        with patch(
            "src.core.executor.build_scoped_agent_for_step",
            side_effect=fake_factory,
        ):
            events = _drain_executor(plan, scratchpad, _flags())

        self.assertEqual(scratchpad.get(1).status, "failed")
        self.assertEqual(scratchpad.get(2).status, "complete")
        self.assertEqual(scratchpad.get(3).status, "skipped")

    def test_single_ready_step_uses_single_dispatch(self):
        s1 = _step(1)
        s2 = _step(2, deps=[1])
        plan = _plan(s1, s2)
        scratchpad = Scratchpad(query="q")

        with patch(
            "src.core.executor.build_scoped_agent_for_step",
            side_effect=lambda *, step, **_: _make_fake_agent(
                step=step, result=_ok_result(step.id)
            ),
        ):
            events = _drain_executor(plan, scratchpad, _flags())

        self.assertEqual(scratchpad.get(1).status, "complete")
        self.assertEqual(scratchpad.get(2).status, "complete")


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
class FailureTests(unittest.TestCase):
    def test_failed_step_marks_descendants_as_skipped(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="filings_agent", deps=[1])
        s3 = _step(3, agent="synthesizer", deps=[2])
        plan = _plan(s1, s2, s3)
        scratchpad = Scratchpad(query="q")

        def fake_factory(*, step: PlanStep, **_kwargs):
            if step.id == 1:
                return _make_fake_agent(
                    step=step, result=_failed_result(1, error="api timeout")
                )
            return _make_fake_agent(step=step, result=_ok_result(step.id))

        with patch(
            "src.core.executor.build_scoped_agent_for_step",
            side_effect=fake_factory,
        ):
            events = _drain_executor(plan, scratchpad, _flags())

        self.assertEqual(scratchpad.get(1).status, "failed")
        r2 = scratchpad.get(2)
        self.assertIsNotNone(r2)
        self.assertEqual(r2.status, "skipped")
        r3 = scratchpad.get(3)
        self.assertIsNotNone(r3)
        self.assertEqual(r3.status, "skipped")

        statuses = [ev["text"] for ev in events if ev.get("type") == "_status"]
        self.assertTrue(any("Step 1 ✗ failed" in s for s in statuses))
        self.assertTrue(any("Step 2" in s and "skipped" in s for s in statuses))

    def test_construction_failure_marks_step_failed_and_continues(self):
        s1 = _step(1)
        s2 = _step(2)
        plan = _plan(s1, s2)
        scratchpad = Scratchpad(query="q")

        def fake_factory(*, step: PlanStep, **_kwargs):
            if step.id == 1:
                raise ScopedAgentError("policy-gated agent without flag")
            return _make_fake_agent(step=step, result=_ok_result(step.id))

        with patch(
            "src.core.executor.build_scoped_agent_for_step",
            side_effect=fake_factory,
        ):
            events = _drain_executor(plan, scratchpad, _flags())

        r1 = scratchpad.get(1)
        self.assertEqual(r1.status, "failed")
        self.assertIn("policy-gated", r1.error or "")
        self.assertEqual(r1.error_type, "ScopedAgentError")
        r2 = scratchpad.get(2)
        self.assertEqual(r2.status, "complete")

        errors = [ev for ev in events if ev.get("type") == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("Step 1 construction failed", errors[0]["text"])

    def test_run_exception_is_caught_as_failed(self):
        s1 = _step(1)
        plan = _plan(s1)
        scratchpad = Scratchpad(query="q")

        fake_agent = MagicMock()
        fake_agent.step = s1

        async def _run_raises():
            raise RuntimeError("kaboom")
            yield  # noqa: unreachable — makes this an async generator

        fake_agent.run = _run_raises

        with patch(
            "src.core.executor.build_scoped_agent_for_step",
            return_value=fake_agent,
        ):
            _drain_executor(plan, scratchpad, _flags())

        r = scratchpad.get(1)
        self.assertEqual(r.status, "failed")
        self.assertEqual(r.error_type, "RuntimeError")
        self.assertIn("kaboom", r.error)


# ---------------------------------------------------------------------------
# Per-step timeout
# ---------------------------------------------------------------------------
class StepTimeoutTests(unittest.TestCase):
    def test_step_exceeding_timeout_is_marked_failed(self):
        s1 = _step(1)
        scratchpad = Scratchpad(query="q")

        fake = MagicMock()
        fake.step = s1

        async def _slow_run():
            await asyncio.sleep(5)  # far longer than the test timeout
            yield {"type": "_step_result", "result": _ok_result(1)}

        fake.run = _slow_run

        async def _go():
            with patch(
                "src.core.executor.build_scoped_agent_for_step",
                return_value=fake,
            ):
                async for _ev in _run_one_step(
                    step=s1,
                    scratchpad=scratchpad,
                    intent_flags=_flags(),
                    all_mcp_tools=[],
                    registry=REGISTRY,
                    recursion_limit=5,
                    step_timeout_s=0.05,
                ):
                    pass

        asyncio.run(_go())

        r = scratchpad.get(1)
        self.assertIsNotNone(r)
        self.assertEqual(r.status, "failed")
        self.assertEqual(r.error_type, "StepTimeout")
        self.assertIn("timeout", (r.error or "").lower())

    def test_fast_step_unaffected_by_generous_timeout(self):
        s1 = _step(1)
        scratchpad = Scratchpad(query="q")
        fake = _make_fake_agent(step=s1, result=_ok_result(1))

        async def _go():
            with patch(
                "src.core.executor.build_scoped_agent_for_step",
                return_value=fake,
            ):
                async for _ev in _run_one_step(
                    step=s1,
                    scratchpad=scratchpad,
                    intent_flags=_flags(),
                    all_mcp_tools=[],
                    registry=REGISTRY,
                    recursion_limit=5,
                    step_timeout_s=5.0,
                ):
                    pass

        asyncio.run(_go())
        self.assertEqual(scratchpad.get(1).status, "complete")

    def test_timeout_disabled_with_none(self):
        s1 = _step(1)
        scratchpad = Scratchpad(query="q")
        fake = _make_fake_agent(step=s1, result=_ok_result(1))

        async def _go():
            with patch(
                "src.core.executor.build_scoped_agent_for_step",
                return_value=fake,
            ):
                async for _ev in _run_one_step(
                    step=s1,
                    scratchpad=scratchpad,
                    intent_flags=_flags(),
                    all_mcp_tools=[],
                    registry=REGISTRY,
                    recursion_limit=5,
                    step_timeout_s=None,  # explicitly disabled
                ):
                    pass

        asyncio.run(_go())
        self.assertEqual(scratchpad.get(1).status, "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
