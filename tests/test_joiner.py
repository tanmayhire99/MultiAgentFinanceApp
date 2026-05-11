"""Unit tests for src.core.joiner — the rule-based joiner / replan decider.

Coverage:

1. finish — synthesizer completed successfully.
2. replan for unmet dependencies — scratchpad has unmet_dependencies.
3. replan for failed steps — a non-synth step failed with replan budget.
4. abort — replan budget exhausted.
5. abort — synthesizer failed/skipped with no replan budget.

Run via::

    uv run pytest tests/test_joiner.py -v
"""
from __future__ import annotations

import time
import unittest

from src.core.joiner import decide
from src.core.types import (
    ExecutionState,
    JoinDecision,
    Plan,
    PlanStep,
    Scratchpad,
    StepResult,
    UnmetDependency,
)


def _step(
    step_id: int,
    *,
    agent: str = "research_agent",
    deps: list[int] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=f"test step {step_id}",
        agent=agent,
        tool_subset=[],
        depends_on=deps or [],
    )


def _plan(*steps: PlanStep) -> Plan:
    return Plan(
        goal="test goal",
        rationale="test rationale",
        steps=list(steps),
    )


def _ok_result(step_id: int) -> StepResult:
    return StepResult(
        step_id=step_id,
        status="complete",
        output={"text": f"output {step_id}"},
        tools_used=[],
        started_at=time.time(),
        completed_at=time.time() + 0.1,
    )


def _failed_result(step_id: int) -> StepResult:
    return StepResult(
        step_id=step_id,
        status="failed",
        output=None,
        error="something went wrong",
        error_type="TestError",
        started_at=time.time(),
        completed_at=time.time() + 0.1,
    )


def _state(
    plan: Plan,
    scratchpad: Scratchpad | None = None,
    *,
    replan_count: int = 0,
    max_replans: int = 2,
) -> ExecutionState:
    return ExecutionState(
        query="test",
        plan=plan,
        scratchpad=scratchpad or Scratchpad(query="test"),
        replan_count=replan_count,
        max_replans=max_replans,
    )


class FinishTests(unittest.TestCase):
    def test_synth_complete_returns_finish(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="synthesizer", deps=[1])
        plan = _plan(s1, s2)
        pad = Scratchpad(query="test")
        pad.add(_ok_result(1))
        pad.add(_ok_result(2))
        state = _state(plan, pad)

        decision = decide(state)
        self.assertEqual(decision.action, "finish")
        self.assertEqual(decision.additional_steps, [])

    def test_synth_complete_with_prior_failure_returns_finish(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="filings_agent")
        s3 = _step(3, agent="synthesizer", deps=[1, 2])
        plan = _plan(s1, s2, s3)
        pad = Scratchpad(query="test")
        pad.add(_ok_result(1))
        pad.add(_failed_result(2))
        pad.add(_ok_result(3))
        state = _state(plan, pad)

        decision = decide(state)
        self.assertEqual(decision.action, "finish")


class ReplanForUnmetTests(unittest.TestCase):
    def test_unmet_dependency_triggers_replan(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="synthesizer", deps=[1])
        plan = _plan(s1, s2)
        pad = Scratchpad(query="test")
        pad.add(_ok_result(1))
        pad.add(_ok_result(2))
        pad.add_unmet_dependency(
            requested_by_step_id=1,
            target_agent="filings_agent",
            reason="Need SEC filings data for analysis",
        )
        state = _state(plan, pad)

        decision = decide(state)
        self.assertEqual(decision.action, "replan")
        self.assertGreater(len(decision.additional_steps), 0)

        agents_in_replan = [s.agent for s in decision.additional_steps]
        self.assertIn("filings_agent", agents_in_replan)
        self.assertIn("synthesizer", agents_in_replan)

    def test_unmet_dependency_exhausted_budget_aborts(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="synthesizer", deps=[1])
        plan = _plan(s1, s2)
        pad = Scratchpad(query="test")
        pad.add(_ok_result(1))
        pad.add(_ok_result(2))
        pad.add_unmet_dependency(
            requested_by_step_id=1,
            target_agent="filings_agent",
            reason="Need SEC filings data for analysis",
        )
        state = _state(plan, pad, replan_count=2, max_replans=2)

        decision = decide(state)
        self.assertEqual(decision.action, "abort")


class ReplanForFailuresTests(unittest.TestCase):
    def test_failed_non_synth_triggers_replan(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="synthesizer", deps=[1])
        plan = _plan(s1, s2)
        pad = Scratchpad(query="test")
        pad.add(_failed_result(1))
        state = _state(plan, pad)

        decision = decide(state)
        self.assertEqual(decision.action, "replan")
        self.assertGreater(len(decision.additional_steps), 0)

    def test_failed_synth_no_replan_budget_aborts(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="synthesizer", deps=[1])
        plan = _plan(s1, s2)
        pad = Scratchpad(query="test")
        pad.add(_ok_result(1))
        pad.add(_failed_result(2))
        state = _state(plan, pad, replan_count=2, max_replans=2)

        decision = decide(state)
        self.assertEqual(decision.action, "abort")


class AbortTests(unittest.TestCase):
    def test_no_synth_step_aborts(self):
        s1 = _step(1, agent="research_agent")
        plan = _plan(s1)
        pad = Scratchpad(query="test")
        pad.add(_ok_result(1))
        state = _state(plan, pad, replan_count=2, max_replans=2)

        decision = decide(state)
        self.assertEqual(decision.action, "abort")

    def test_all_ok_but_no_synth_with_no_replan_budget(self):
        s1 = _step(1, agent="research_agent")
        plan = _plan(s1)
        pad = Scratchpad(query="test")
        pad.add(_ok_result(1))
        state = _state(plan, pad, replan_count=0, max_replans=0)

        decision = decide(state)
        self.assertEqual(decision.action, "abort")


class ReplanStepIdTests(unittest.TestCase):
    def test_replan_steps_get_unique_ids(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="synthesizer", deps=[1])
        plan = _plan(s1, s2)
        pad = Scratchpad(query="test")
        pad.add(_failed_result(1))
        state = _state(plan, pad)

        decision = decide(state)
        self.assertEqual(decision.action, "replan")
        ids = [s.id for s in decision.additional_steps]
        self.assertEqual(len(ids), len(set(ids)), "replan step IDs must be unique")

    def test_replan_steps_do_not_collide_with_existing(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="synthesizer", deps=[1])
        plan = _plan(s1, s2)
        pad = Scratchpad(query="test")
        pad.add(_failed_result(1))
        state = _state(plan, pad)

        decision = decide(state)
        existing_ids = {s.id for s in plan.steps}
        for ns in decision.additional_steps:
            self.assertNotIn(ns.id, existing_ids)

    def test_replan_steps_have_correct_replan_round(self):
        s1 = _step(1, agent="research_agent")
        s2 = _step(2, agent="synthesizer", deps=[1])
        plan = _plan(s1, s2)
        pad = Scratchpad(query="test")
        pad.add(_failed_result(1))
        state = _state(plan, pad, replan_count=0)

        decision = decide(state)
        for ns in decision.additional_steps:
            self.assertEqual(ns.replan_round, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
