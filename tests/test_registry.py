"""Unit tests for src.core.agents.registry — Day 2 of the migration.

Coverage:

1. PolicyGate matching: intent-flag boolean checks (no regex, no text matching)
2. PolicyGate validation: unknown flag names rejected at construction
3. AgentDefinition validation: name format, field bounds
4. AgentRegistry construction: duplicate names, duplicate tool ownership
5. Tool ownership lookup
6. validate_step / validate_plan: unknown agent, tool not owned, gate
   unsatisfied, valid step (all parameterised by ``intent_flags``)
7. The headline behaviour: claim_agent / panel_agent are blocked unless the
   classifier set the corresponding intent flag
8. The canonical REGISTRY: 37-tool count, no orphan tools, gated set,
   planner_catalog_text shape

Run via::

    docker exec finai-api python -m unittest tests.test_registry -v
"""
from __future__ import annotations

import unittest
from typing import Dict, List

from pydantic import ValidationError

from src.core.agents.registry import (
    AgentDefinition,
    AgentRegistry,
    CLAIM_AGENT,
    FILINGS_AGENT,
    INDIAN_STOCK_AGENT,
    PANEL_AGENT,
    PolicyGate,
    PORTFOLIO_AGENT,
    REGISTRY,
    RESEARCH_AGENT,
    SYNTHESIZER,
    US_STOCK_AGENT,
)
from src.core.types import KNOWN_INTENT_FLAGS, Plan, PlanStep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _step(
    step_id: int = 1,
    *,
    agent: str = "research_agent",
    tool_subset: List[str] | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description="test step",
        agent=agent,
        tool_subset=tool_subset or [],
    )


def _flags(**overrides: bool) -> Dict[str, bool]:
    """Build a complete intent_flags dict with only the named flags True.

    Mirrors what the classifier would produce. Keeping this helper in
    the test module so a future flag addition surfaces here first.
    """
    base = {f: False for f in KNOWN_INTENT_FLAGS}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# PolicyGate
# ---------------------------------------------------------------------------
class PolicyGateTests(unittest.TestCase):
    def test_open_gate_with_no_flags_always_matches(self):
        # An empty required_intent_flags is a vacuously open gate.
        gate = PolicyGate()
        self.assertTrue(gate.matches(_flags()))
        self.assertTrue(gate.matches(_flags(wants_claim_tracking=True)))

    def test_single_flag_gate_matches_when_flag_true(self):
        gate = PolicyGate(required_intent_flags=["wants_claim_tracking"])
        self.assertTrue(gate.matches(_flags(wants_claim_tracking=True)))

    def test_single_flag_gate_blocks_when_flag_false(self):
        gate = PolicyGate(required_intent_flags=["wants_claim_tracking"])
        self.assertFalse(gate.matches(_flags()))
        self.assertFalse(gate.matches(_flags(wants_claim_tracking=False)))

    def test_multi_flag_gate_matches_on_any(self):
        # Logical OR: any flag True satisfies the gate.
        gate = PolicyGate(
            required_intent_flags=["wants_claim_tracking", "wants_filings"],
        )
        self.assertTrue(gate.matches(_flags(wants_claim_tracking=True)))
        self.assertTrue(gate.matches(_flags(wants_filings=True)))
        self.assertTrue(gate.matches(
            _flags(wants_claim_tracking=True, wants_filings=True)
        ))
        self.assertFalse(gate.matches(_flags()))

    def test_missing_flag_treated_as_false(self):
        gate = PolicyGate(required_intent_flags=["wants_claim_tracking"])
        self.assertFalse(gate.matches({}))  # totally empty dict
        self.assertFalse(gate.matches({"wants_panel_debate": True}))

    def test_unknown_flag_name_rejected_at_construction(self):
        # Catches typos at import time, not at runtime.
        with self.assertRaisesRegex(ValidationError, "Unknown intent flag"):
            PolicyGate(required_intent_flags=["wants_claim_tracking_typo"])
        with self.assertRaisesRegex(ValidationError, "Unknown intent flag"):
            PolicyGate(required_intent_flags=["panel"])  # close miss

    def test_explain_string_format(self):
        gate = PolicyGate(
            description="Claim tracking is opt-in.",
            required_intent_flags=["wants_claim_tracking"],
        )
        explained = gate.explain()
        self.assertIn("opt-in", explained)
        self.assertIn("wants_claim_tracking", explained)
        self.assertIn("HARD-BLOCK", explained)

    def test_explain_advisory_mode(self):
        gate = PolicyGate(
            required_intent_flags=["wants_claim_tracking"],
            hard_block_unless_match=False,
        )
        self.assertIn("advisory", gate.explain())


# ---------------------------------------------------------------------------
# AgentDefinition
# ---------------------------------------------------------------------------
class AgentDefinitionTests(unittest.TestCase):
    def test_valid_agent(self):
        a = AgentDefinition(
            name="research_agent",
            title="Research Agent",
            description="Does research over the web.",
            tools=("research__search_news",),
            role_hint="any web-search query",
        )
        self.assertEqual(a.name, "research_agent")
        self.assertEqual(a.tools, ("research__search_news",))
        self.assertIsNone(a.policy_gate)

    def test_name_must_be_snake_case(self):
        for bad in ["ResearchAgent", "research-agent", "research agent",
                    "1research", "_research", "RESEARCH_AGENT"]:
            with self.subTest(name=bad):
                with self.assertRaisesRegex(ValidationError, "snake_case"):
                    AgentDefinition(
                        name=bad,
                        title="x",
                        description="x" * 20,
                        tools=(),
                    )

    def test_description_min_length(self):
        with self.assertRaises(ValidationError):
            AgentDefinition(
                name="x_agent",
                title="x",
                description="short",  # < 10 chars
                tools=(),
            )

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            AgentDefinition(
                name="x_agent",
                title="x",
                description="x" * 20,
                tools=(),
                bogus_field=42,  # type: ignore[call-arg]
            )

    def test_frozen(self):
        a = AgentDefinition(
            name="x_agent", title="x", description="x" * 20, tools=()
        )
        with self.assertRaises(ValidationError):
            a.name = "y_agent"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AgentRegistry construction
# ---------------------------------------------------------------------------
class AgentRegistryConstructionTests(unittest.TestCase):
    def test_duplicate_agent_names_rejected(self):
        a1 = AgentDefinition(name="x_agent", title="x",
                             description="x" * 20, tools=("ns__a",))
        a2 = AgentDefinition(name="x_agent", title="x2",
                             description="y" * 20, tools=("ns__b",))
        with self.assertRaisesRegex(ValueError, "Duplicate agent name"):
            AgentRegistry([a1, a2])

    def test_duplicate_tool_ownership_rejected(self):
        a1 = AgentDefinition(name="a_agent", title="a",
                             description="a" * 20, tools=("shared__tool",))
        a2 = AgentDefinition(name="b_agent", title="b",
                             description="b" * 20, tools=("shared__tool",))
        with self.assertRaisesRegex(ValueError, "claimed by both"):
            AgentRegistry([a1, a2])

    def test_minimal_registry(self):
        a = AgentDefinition(
            name="x_agent", title="x", description="x" * 20,
            tools=("ns__t1", "ns__t2"),
        )
        r = AgentRegistry([a])
        self.assertEqual(len(r), 1)
        self.assertEqual(r.names(), ["x_agent"])
        self.assertEqual(r.tool_owner("ns__t1"), "x_agent")
        self.assertIsNone(r.tool_owner("not_registered"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class RegistryValidationTests(unittest.TestCase):
    def test_unknown_agent_in_step(self):
        s = _step(agent="not_a_real_agent")
        errs = REGISTRY.validate_step(s, _flags())
        self.assertEqual(len(errs), 1)
        self.assertIn("unknown agent", errs[0])

    def test_tool_not_owned_by_agent(self):
        # research_agent does NOT own us_stock__get_quote
        s = _step(agent="research_agent",
                  tool_subset=["us_stock__get_quote"])
        errs = REGISTRY.validate_step(s, _flags())
        self.assertTrue(any("not owned by agent" in e for e in errs))

    def test_valid_step_passes(self):
        s = _step(
            agent="research_agent",
            tool_subset=["research__search_news",
                         "research__get_company_brief"],
        )
        errs = REGISTRY.validate_step(s, _flags())
        self.assertEqual(errs, [])

    def test_synthesizer_valid_with_no_tools(self):
        s = _step(agent="synthesizer", tool_subset=[])
        errs = REGISTRY.validate_step(s, _flags())
        self.assertEqual(errs, [])

    def test_validate_plan_aggregates_errors(self):
        bad_plan = Plan(
            goal="test goal",
            rationale="test rationale",
            steps=[
                _step(1, agent="research_agent",
                      tool_subset=["research__search_news"]),  # ok
                _step(2, agent="bogus_agent"),                  # bad
                _step(3, agent="us_stock_agent",
                      tool_subset=["research__search_news"]),   # bad: not owned
            ],
        )
        errs = REGISTRY.validate_plan(bad_plan, _flags())
        self.assertEqual(len(errs), 2)
        self.assertTrue(any("bogus_agent" in e for e in errs))
        self.assertTrue(any("not owned by agent" in e for e in errs))


# ---------------------------------------------------------------------------
# Policy gates on the canonical agents — the headline behaviour
#
# THE WHOLE POINT: claim_agent / panel_agent are gate-blocked unless the
# CLASSIFIER (a real LLM, in Phase 1) set the corresponding intent flag.
# These tests exercise the registry's structural check, not the
# classifier's semantic understanding. The classifier's job is to set
# the flags correctly given a natural-language query - that lives in
# tests/test_classifier.py (Day 6 work) and is mocked here.
# ---------------------------------------------------------------------------
class CanonicalPolicyGateTests(unittest.TestCase):
    def test_claim_agent_blocked_when_flag_false(self):
        s = _step(agent="claim_agent",
                  tool_subset=["research__extract_forward_claims"])
        errs = REGISTRY.validate_step(s, _flags(wants_claim_tracking=False))
        self.assertTrue(any("policy-gated" in e for e in errs),
                        f"Expected gate block, got: {errs}")

    def test_claim_agent_blocked_when_flag_missing(self):
        # Treating missing flags as False is critical so a buggy
        # classifier output doesn't accidentally enable gated agents.
        s = _step(agent="claim_agent",
                  tool_subset=["research__extract_forward_claims"])
        errs = REGISTRY.validate_step(s, {})
        self.assertTrue(any("policy-gated" in e for e in errs))

    def test_claim_agent_allowed_when_flag_true(self):
        s = _step(agent="claim_agent",
                  tool_subset=["research__extract_forward_claims"])
        errs = REGISTRY.validate_step(
            s, _flags(wants_claim_tracking=True)
        )
        self.assertEqual(errs, [])

    def test_claim_agent_unaffected_by_unrelated_flags(self):
        # A different flag being true shouldn't open the gate.
        s = _step(agent="claim_agent",
                  tool_subset=["research__extract_forward_claims"])
        errs = REGISTRY.validate_step(
            s, _flags(wants_panel_debate=True, wants_filings=True),
        )
        self.assertTrue(any("policy-gated" in e for e in errs))

    def test_panel_agent_blocked_when_flag_false(self):
        s = _step(agent="panel_agent")
        errs = REGISTRY.validate_step(s, _flags(wants_panel_debate=False))
        self.assertTrue(any("policy-gated" in e for e in errs))

    def test_panel_agent_allowed_when_flag_true(self):
        s = _step(agent="panel_agent")
        errs = REGISTRY.validate_step(s, _flags(wants_panel_debate=True))
        self.assertEqual(errs, [])

    def test_both_gates_independent(self):
        plan = Plan(
            goal="dual-gate plan",
            rationale="exercises both gates simultaneously",
            steps=[
                _step(1, agent="claim_agent",
                      tool_subset=["research__extract_forward_claims"]),
                _step(2, agent="panel_agent"),
            ],
        )
        # Neither flag set → both blocked
        errs = REGISTRY.validate_plan(plan, _flags())
        self.assertEqual(
            sum(1 for e in errs if "policy-gated" in e), 2,
            f"Expected both gates blocked; errors: {errs}",
        )
        # Only claim flag set → only panel blocked
        errs = REGISTRY.validate_plan(
            plan, _flags(wants_claim_tracking=True),
        )
        gated = [e for e in errs if "policy-gated" in e]
        self.assertEqual(len(gated), 1)
        self.assertIn("panel_agent", gated[0])
        # Both flags set → no errors
        errs = REGISTRY.validate_plan(
            plan,
            _flags(wants_claim_tracking=True, wants_panel_debate=True),
        )
        self.assertEqual(errs, [])


# ---------------------------------------------------------------------------
# The canonical REGISTRY — invariants that must hold for the demo to work
# ---------------------------------------------------------------------------
class CanonicalRegistryInvariantTests(unittest.TestCase):
    def test_tool_count_matches_mcp_namespacing(self):
        # The 8 agents collectively own exactly 37 tools — the same total
        # as what mcp_servers.get_tools() returns. If this fails the
        # registry got out of sync with the MCP workers.
        all_tools = set()
        for a in REGISTRY:
            all_tools.update(a.tools)
        self.assertEqual(
            len(all_tools), 37,
            f"Expected 37 namespaced MCP tools across the registry, "
            f"got {len(all_tools)}. Tools: {sorted(all_tools)}",
        )

    def test_every_agent_has_unique_tools(self):
        # Already enforced at construction time, but pin it as a test so a
        # future edit to the catalog can't accidentally allow overlaps.
        seen, dupes = set(), []
        for a in REGISTRY:
            for t in a.tools:
                if t in seen:
                    dupes.append(t)
                seen.add(t)
        self.assertEqual(dupes, [], f"Duplicate tool ownership: {dupes}")

    def test_claim_tools_owned_by_claim_agent_only(self):
        for tool in ("research__extract_forward_claims",
                     "research__compare_claim_to_reality"):
            self.assertEqual(
                REGISTRY.tool_owner(tool), "claim_agent",
                f"{tool} must be owned by claim_agent (gated), "
                f"got {REGISTRY.tool_owner(tool)!r}",
            )

    def test_gated_set_is_exactly_two(self):
        gated = [a.name for a in REGISTRY.gated_agents()]
        self.assertEqual(set(gated), {"claim_agent", "panel_agent"})

    def test_claim_agent_required_flag(self):
        self.assertEqual(
            CLAIM_AGENT.policy_gate.required_intent_flags,  # type: ignore[union-attr]
            ["wants_claim_tracking"],
        )

    def test_panel_agent_required_flag(self):
        self.assertEqual(
            PANEL_AGENT.policy_gate.required_intent_flags,  # type: ignore[union-attr]
            ["wants_panel_debate"],
        )

    def test_synthesizer_has_no_tools(self):
        self.assertEqual(REGISTRY.get("synthesizer").tools, ())  # type: ignore[union-attr]

    def test_panel_agent_has_no_tools(self):
        # Panel runs sub-personas internally; from the planner's view it's
        # tool-less so the planner can't try to call panel-internal tools.
        self.assertEqual(REGISTRY.get("panel_agent").tools, ())  # type: ignore[union-attr]

    def test_planner_catalog_text_includes_every_agent(self):
        text = REGISTRY.planner_catalog_text()
        for a in REGISTRY:
            self.assertIn(f"`{a.name}`", text,
                          f"agent {a.name!r} missing from planner catalog text")

    def test_planner_catalog_text_marks_gates(self):
        text = REGISTRY.planner_catalog_text()
        # Gated agents must have the warning marker
        self.assertIn("POLICY GATE", text)
        # And the warning marker should appear at most twice (the two gated
        # agents). This catches accidental gating of an extra agent.
        self.assertEqual(text.count("POLICY GATE"), 2)

    def test_planner_catalog_text_lists_ungated_first(self):
        text = REGISTRY.planner_catalog_text()
        # Sanity: the first agent listed should be ungated
        first_agent_line = next(
            line for line in text.splitlines()
            if line.startswith("### `")
        )
        first_name = first_agent_line.split("`")[1]
        self.assertIsNone(REGISTRY.get(first_name).policy_gate,  # type: ignore[union-attr]
                          f"First agent in catalog ({first_name}) should not be gated")

    def test_planner_catalog_text_mentions_intent_flags(self):
        # The planner LLM must see WHICH flag gates each agent so it can
        # stop including them when the classifier left the flag unset.
        text = REGISTRY.planner_catalog_text()
        self.assertIn("wants_claim_tracking", text)
        self.assertIn("wants_panel_debate", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
