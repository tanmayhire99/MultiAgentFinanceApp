"""Unit tests for src.core.types — the planner-first foundation.

Stdlib unittest only so this runs in any environment that can import the
project, no pytest dependency required. Run via::

    docker exec finai-api python -m unittest tests.test_types -v
    # or, on the host with the project's environment:
    python -m unittest tests.test_types -v

Coverage target
---------------
* PlanStep field validation (ranges, required, forbid-extra)
* Plan DAG validation (unique IDs, missing refs, self-loops, cycles)
* Plan helpers (topological_order, ready_steps, all_tools, all_agents)
* Variable-reference parsing + resolution (#3, #3.url, #3.items[0].url)
* Scratchpad relevant_results_for_step (scoped context)
* ExecutionState replan tracking + duration
* JoinDecision action/additional_steps consistency rules
"""
from __future__ import annotations

import time
import unittest

from pydantic import ValidationError

from src.core.types import (
    ExecutionState,
    JoinDecision,
    Plan,
    PlanStep,
    Scratchpad,
    StepResult,
    UnmetDependency,
    parse_var_ref,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _step(
    step_id: int,
    *,
    agent: str = "research_agent",
    tools: list | None = None,
    deps: list | None = None,
    inputs: dict | None = None,
    desc: str = "test step",
) -> PlanStep:
    """Concise PlanStep factory for tests."""
    return PlanStep(
        id=step_id,
        description=desc,
        agent=agent,
        tool_subset=tools or [],
        inputs=inputs or {},
        depends_on=deps or [],
    )


def _plan(*steps: PlanStep, goal: str = "test goal") -> Plan:
    return Plan(
        goal=goal,
        rationale="Test plan rationale.",
        steps=list(steps),
    )


# ---------------------------------------------------------------------------
# PlanStep validation
# ---------------------------------------------------------------------------
class PlanStepValidationTests(unittest.TestCase):
    def test_minimal_valid_step(self):
        s = PlanStep(
            id=1,
            description="Fetch news",
            agent="research_agent",
        )
        self.assertEqual(s.tool_subset, [])
        self.assertEqual(s.depends_on, [])
        self.assertEqual(s.replan_round, 0)
        self.assertEqual(s.max_tool_calls, 20)

    def test_id_must_be_positive(self):
        with self.assertRaises(ValidationError):
            PlanStep(id=0, description="x", agent="research_agent")
        with self.assertRaises(ValidationError):
            PlanStep(id=-3, description="x", agent="research_agent")

    def test_description_length_bounds(self):
        with self.assertRaises(ValidationError):
            PlanStep(id=1, description="abc", agent="research_agent")  # too short
        with self.assertRaises(ValidationError):
            PlanStep(id=1, description="x" * 401, agent="research_agent")  # too long

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            PlanStep(
                id=1,
                description="ok desc",
                agent="research_agent",
                this_field_does_not_exist=42,  # type: ignore[call-arg]
            )

    def test_max_tool_calls_bounds(self):
        with self.assertRaises(ValidationError):
            PlanStep(id=1, description="ok desc", agent="x", max_tool_calls=0)
        with self.assertRaises(ValidationError):
            PlanStep(id=1, description="ok desc", agent="x", max_tool_calls=999)

    def test_tool_subset_rejects_blank_strings(self):
        with self.assertRaises(ValidationError):
            PlanStep(
                id=1,
                description="ok desc",
                agent="research_agent",
                tool_subset=["search_news", ""],
            )

    def test_depends_on_no_duplicates(self):
        with self.assertRaises(ValidationError):
            PlanStep(
                id=3,
                description="ok desc",
                agent="research_agent",
                depends_on=[1, 1, 2],
            )


# ---------------------------------------------------------------------------
# Plan DAG validation
# ---------------------------------------------------------------------------
class PlanDAGValidationTests(unittest.TestCase):
    def test_unique_step_ids(self):
        with self.assertRaisesRegex(ValidationError, "unique within the plan"):
            _plan(_step(1), _step(1))

    def test_dependency_must_exist(self):
        with self.assertRaisesRegex(ValidationError, "non-existent step"):
            _plan(_step(1), _step(2, deps=[42]))

    def test_self_loop_rejected(self):
        with self.assertRaisesRegex(ValidationError, "depends on itself"):
            # Trick: validator runs at Plan-level, so we set deps in PlanStep
            _plan(_step(1, deps=[1]))

    def test_cycle_rejected(self):
        # 1 -> 2 -> 3 -> 1
        with self.assertRaisesRegex(ValidationError, "dependency cycle"):
            _plan(
                _step(1, deps=[3]),
                _step(2, deps=[1]),
                _step(3, deps=[2]),
            )

    def test_valid_dag_topological_order(self):
        p = _plan(
            _step(1, deps=[]),
            _step(2, deps=[1]),
            _step(3, deps=[1]),
            _step(4, deps=[2, 3]),
        )
        order = [s.id for s in p.topological_order()]
        # 1 must come before 2, 3; 4 must come last
        self.assertEqual(order[0], 1)
        self.assertEqual(order[-1], 4)
        self.assertIn(order[1:3], ([2, 3], [3, 2]))

    def test_ready_steps(self):
        p = _plan(
            _step(1, deps=[]),
            _step(2, deps=[1]),
            _step(3, deps=[1]),
            _step(4, deps=[2, 3]),
        )
        # Initially only step 1 is ready
        ready = [s.id for s in p.ready_steps(set())]
        self.assertEqual(ready, [1])
        # After 1 completes, both 2 and 3 are ready
        ready = sorted(s.id for s in p.ready_steps({1}))
        self.assertEqual(ready, [2, 3])
        # After 1, 2 done — only 3 (4 still waiting on 3)
        ready = [s.id for s in p.ready_steps({1, 2})]
        self.assertEqual(ready, [3])
        # All deps satisfied -> 4 ready
        ready = [s.id for s in p.ready_steps({1, 2, 3})]
        self.assertEqual(ready, [4])

    def test_all_tools_and_all_agents_aggregation(self):
        p = _plan(
            _step(1, agent="research_agent", tools=["search_news"]),
            _step(2, agent="us_stock_agent", tools=["get_quote", "get_fundamentals"]),
            _step(3, agent="synthesizer", tools=[], deps=[1, 2]),
        )
        self.assertEqual(
            p.all_tools(),
            {"search_news", "get_quote", "get_fundamentals"},
        )
        self.assertEqual(
            p.all_agents(),
            {"research_agent", "us_stock_agent", "synthesizer"},
        )


# ---------------------------------------------------------------------------
# Variable-reference parsing + resolution
# ---------------------------------------------------------------------------
class VariableRefTests(unittest.TestCase):
    def test_parse_bare_ref(self):
        self.assertEqual(parse_var_ref("#3"), (3, None))

    def test_parse_dotted_ref(self):
        self.assertEqual(parse_var_ref("#3.url"), (3, "url"))

    def test_parse_indexed_ref(self):
        self.assertEqual(parse_var_ref("#3.items[0].url"), (3, "items[0].url"))

    def test_parse_non_ref_returns_none(self):
        self.assertIsNone(parse_var_ref("hello"))
        self.assertIsNone(parse_var_ref("#abc"))
        self.assertIsNone(parse_var_ref(""))

    def test_resolve_bare_ref(self):
        sp = Scratchpad(query="q")
        sp.add(StepResult(
            step_id=1,
            status="complete",
            output={"items": [{"url": "https://a.test"}, {"url": "https://b.test"}]},
            completed_at=time.time(),
        ))
        self.assertEqual(
            sp.resolve_value("#1"),
            {"items": [{"url": "https://a.test"}, {"url": "https://b.test"}]},
        )

    def test_resolve_dotted_ref(self):
        sp = Scratchpad(query="q")
        sp.add(StepResult(
            step_id=1, status="complete",
            output={"items": [{"url": "https://a.test"}], "count": 1},
            completed_at=time.time(),
        ))
        self.assertEqual(sp.resolve_value("#1.count"), 1)

    def test_resolve_indexed_ref(self):
        sp = Scratchpad(query="q")
        sp.add(StepResult(
            step_id=1, status="complete",
            output={"items": [{"url": "https://a.test"}, {"url": "https://b.test"}]},
            completed_at=time.time(),
        ))
        self.assertEqual(sp.resolve_value("#1.items[0].url"), "https://a.test")
        self.assertEqual(sp.resolve_value("#1.items[1].url"), "https://b.test")

    def test_resolve_recurses_into_dicts_and_lists(self):
        sp = Scratchpad(query="q")
        sp.add(StepResult(
            step_id=1, status="complete",
            output="resolved-value",
            completed_at=time.time(),
        ))
        nested = {
            "tool_arg": "#1",
            "static": "literal",
            "list_arg": ["#1", "another literal", {"deep": "#1"}],
        }
        out = sp.resolve_value(nested)
        self.assertEqual(out["tool_arg"], "resolved-value")
        self.assertEqual(out["static"], "literal")
        self.assertEqual(out["list_arg"][0], "resolved-value")
        self.assertEqual(out["list_arg"][1], "another literal")
        self.assertEqual(out["list_arg"][2]["deep"], "resolved-value")

    def test_resolve_missing_step_returns_none(self):
        sp = Scratchpad(query="q")
        # No step 5 in scratchpad
        self.assertIsNone(sp.resolve_value("#5"))
        self.assertIsNone(sp.resolve_value("#5.field"))

    def test_resolve_missing_path_returns_none(self):
        sp = Scratchpad(query="q")
        sp.add(StepResult(
            step_id=1, status="complete",
            output={"items": []},
            completed_at=time.time(),
        ))
        self.assertIsNone(sp.resolve_value("#1.items[0]"))  # empty list
        self.assertIsNone(sp.resolve_value("#1.nonexistent"))

    def test_resolve_failed_step_returns_none(self):
        sp = Scratchpad(query="q")
        sp.add(StepResult(
            step_id=1, status="failed", output=None, error="boom",
            completed_at=time.time(),
        ))
        self.assertIsNone(sp.resolve_value("#1"))


# ---------------------------------------------------------------------------
# Scratchpad scoped-context behaviour
# ---------------------------------------------------------------------------
class ScratchpadScopedContextTests(unittest.TestCase):
    def setUp(self):
        self.sp = Scratchpad(query="q")
        for sid, content in [(1, "A"), (2, "B"), (3, "C"), (4, "D")]:
            self.sp.add(StepResult(
                step_id=sid, status="complete", output=content,
                completed_at=time.time(),
            ))

    def test_relevant_results_only_returns_declared_deps(self):
        step5 = _step(5, deps=[1, 3])
        scoped = self.sp.relevant_results_for_step(step5)
        self.assertEqual(set(scoped.keys()), {1, 3})
        self.assertEqual(scoped[1].output, "A")
        self.assertEqual(scoped[3].output, "C")
        self.assertNotIn(2, scoped)
        self.assertNotIn(4, scoped)

    def test_relevant_results_handles_missing_dep(self):
        step5 = _step(5, deps=[1, 99])  # 99 doesn't exist in scratchpad yet
        scoped = self.sp.relevant_results_for_step(step5)
        self.assertEqual(set(scoped.keys()), {1})  # 99 silently dropped

    def test_completed_ids_excludes_failed_and_skipped(self):
        sp = Scratchpad(query="q")
        sp.add(StepResult(step_id=1, status="complete", output="ok",
                          completed_at=time.time()))
        sp.add(StepResult(step_id=2, status="failed", error="boom",
                          completed_at=time.time()))
        sp.add(StepResult(step_id=3, status="skipped", completed_at=time.time()))
        self.assertEqual(sp.completed_ids(), {1})
        self.assertEqual(sp.terminal_ids(), {1, 2, 3})
        self.assertTrue(sp.has_failures())


# ---------------------------------------------------------------------------
# StepResult / ExecutionState
# ---------------------------------------------------------------------------
class StepResultAndExecutionStateTests(unittest.TestCase):
    def test_step_result_duration(self):
        r = StepResult(
            step_id=1,
            status="complete",
            output="ok",
            started_at=100.0,
            completed_at=103.5,
        )
        self.assertAlmostEqual(r.duration_s, 3.5)
        self.assertTrue(r.is_terminal)

    def test_running_step_has_no_duration(self):
        r = StepResult(step_id=1, status="running", started_at=100.0)
        self.assertIsNone(r.duration_s)
        self.assertFalse(r.is_terminal)

    def test_execution_state_replan_budget(self):
        plan = _plan(_step(1))
        state = ExecutionState(
            query="q",
            plan=plan,
            scratchpad=Scratchpad(query="q"),
            max_replans=2,
        )
        self.assertTrue(state.can_replan)
        state.replan_count = 1
        self.assertTrue(state.can_replan)
        state.replan_count = 2
        self.assertFalse(state.can_replan)

    def test_all_steps_terminal(self):
        plan = _plan(_step(1), _step(2))
        sp = Scratchpad(query="q")
        sp.add(StepResult(step_id=1, status="complete", output="a",
                          completed_at=time.time()))
        state = ExecutionState(query="q", plan=plan, scratchpad=sp)
        self.assertFalse(state.all_steps_terminal())
        sp.add(StepResult(step_id=2, status="failed", error="x",
                          completed_at=time.time()))
        self.assertTrue(state.all_steps_terminal())  # failed counts as terminal


# ---------------------------------------------------------------------------
# JoinDecision consistency rules
# ---------------------------------------------------------------------------
class JoinDecisionConsistencyTests(unittest.TestCase):
    def test_finish_no_additional_steps(self):
        d = JoinDecision(action="finish", reasoning="all good")
        self.assertEqual(d.action, "finish")
        self.assertEqual(d.additional_steps, [])

    def test_replan_requires_additional_steps(self):
        with self.assertRaisesRegex(
            ValidationError, "requires at least one PlanStep"
        ):
            JoinDecision(action="replan", reasoning="need more")

    def test_replan_with_additional_steps(self):
        d = JoinDecision(
            action="replan",
            reasoning="need claim extraction",
            additional_steps=[_step(99, agent="claim_agent")],
        )
        self.assertEqual(len(d.additional_steps), 1)

    def test_finish_must_not_have_additional_steps(self):
        with self.assertRaisesRegex(
            ValidationError, "additional_steps must be empty"
        ):
            JoinDecision(
                action="finish",
                reasoning="results sufficient",
                additional_steps=[_step(99)],
            )

    def test_abort_must_not_have_additional_steps(self):
        with self.assertRaisesRegex(
            ValidationError, "additional_steps must be empty"
        ):
            JoinDecision(
                action="abort",
                reasoning="unrecoverable",
                additional_steps=[_step(99)],
            )


# ---------------------------------------------------------------------------
# UnmetDependency model + Scratchpad.add_unmet_dependency
# ---------------------------------------------------------------------------
class UnmetDependencyTests(unittest.TestCase):
    """Day 3 addition: scoped agents emit these when blocked.

    The model is intentionally strict (extra='forbid', length bounds on
    ``reason``) so a buggy synthetic-tool implementation can't silently
    write garbage that the joiner then tries to use.
    """

    def test_minimal_valid(self):
        d = UnmetDependency(
            requested_by_step_id=2,
            target_agent="us_stock_agent",
            reason="Need current quote for the synthesis step.",
        )
        self.assertEqual(d.requested_by_step_id, 2)
        self.assertEqual(d.target_agent, "us_stock_agent")
        self.assertGreater(d.raised_at, 0)  # auto-set timestamp

    def test_step_id_must_be_positive(self):
        with self.assertRaises(ValidationError):
            UnmetDependency(
                requested_by_step_id=0,
                target_agent="x",
                reason="needs at least 10 characters",
            )
        with self.assertRaises(ValidationError):
            UnmetDependency(
                requested_by_step_id=-1,
                target_agent="x",
                reason="needs at least 10 characters",
            )

    def test_reason_length_bounds(self):
        # Too short
        with self.assertRaises(ValidationError):
            UnmetDependency(
                requested_by_step_id=1,
                target_agent="x",
                reason="too short",  # 9 chars
            )
        # Too long
        with self.assertRaises(ValidationError):
            UnmetDependency(
                requested_by_step_id=1,
                target_agent="x",
                reason="a" * 501,  # 501 chars
            )

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            UnmetDependency(
                requested_by_step_id=1,
                target_agent="x",
                reason="long enough reason here",
                bogus_field=42,  # type: ignore[call-arg]
            )

    def test_scratchpad_starts_with_no_unmet_dependencies(self):
        sp = Scratchpad(query="q")
        self.assertEqual(sp.unmet_dependencies, [])

    def test_add_unmet_dependency_appends_and_returns(self):
        sp = Scratchpad(query="q")
        d1 = sp.add_unmet_dependency(
            requested_by_step_id=2,
            target_agent="us_stock_agent",
            reason="need current quote for downstream step",
        )
        d2 = sp.add_unmet_dependency(
            requested_by_step_id=3,
            target_agent="filings_agent",
            reason="need 10-K text to validate the management claim",
        )
        self.assertEqual(len(sp.unmet_dependencies), 2)
        self.assertIs(sp.unmet_dependencies[0], d1)
        self.assertIs(sp.unmet_dependencies[1], d2)
        # Helper does NOT collapse duplicates - the planner needs to see
        # repeated requests to gauge urgency.
        sp.add_unmet_dependency(
            requested_by_step_id=2,
            target_agent="us_stock_agent",
            reason="need current quote for downstream step",
        )
        self.assertEqual(len(sp.unmet_dependencies), 3)


# ---------------------------------------------------------------------------
# JSON-schema generation (used by the planner LLM's response_format)
# ---------------------------------------------------------------------------
class JSONSchemaTests(unittest.TestCase):
    """The planner LLM is given Plan.model_json_schema() as its response_format.

    These tests pin the schema's shape so a planner change doesn't silently
    break the LLM contract.
    """

    def test_plan_schema_has_required_top_level_fields(self):
        schema = Plan.model_json_schema()
        # Pydantic v2 keeps the title and lists required at the top level
        required = set(schema.get("required", []))
        # ``goal`` and ``rationale`` are required (no defaults). ``steps``,
        # ``schema_version``, and ``estimated_complexity`` have defaults so
        # may or may not appear in ``required`` depending on Pydantic v2
        # version - we only enforce the must-haves.
        self.assertIn("goal", required)
        self.assertIn("rationale", required)
        # All five fields exist as properties
        for key in ("goal", "rationale", "steps", "schema_version",
                    "estimated_complexity"):
            self.assertIn(key, schema["properties"])

    def test_plan_step_schema_field_descriptions_present(self):
        schema = PlanStep.model_json_schema()
        # Field descriptions matter because the planner LLM reads them
        for key in ("id", "agent", "tool_subset", "depends_on", "max_tool_calls"):
            self.assertIn("description", schema["properties"][key])
            self.assertGreater(len(schema["properties"][key]["description"]), 10)

    def test_plan_step_extra_forbid(self):
        schema = PlanStep.model_json_schema()
        # additionalProperties False (or "false") signals extra=forbid.
        self.assertFalse(schema.get("additionalProperties", True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
