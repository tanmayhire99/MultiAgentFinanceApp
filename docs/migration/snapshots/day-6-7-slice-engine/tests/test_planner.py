"""Unit tests for src.core.planner — Slice Stage 2a.

Coverage:

1. Happy path — LLM returns a valid Plan JSON, planner returns the
   parsed Plan object.
2. JSON parse failure with code-fence — `_try_repair_json` strips the
   fence and the planner accepts on first try.
3. JSON parse failure → repair retry → success.
4. Schema validation failure → repair retry with errors echoed.
5. Policy gate violation → repair retry with errors echoed.
6. Exhausted retries → :class:`PlannerError` raised with diagnostic
   info attached.

Mocking strategy
----------------
``build_chat_model`` is patched to return a fake chat-model whose
``ainvoke`` is configured per-test to return canned ``AIMessage``
content. We don't actually hit NIM. Tests verify the planner's
parse / validate / repair-retry logic against these canned outputs.

Run via::

    docker exec finai-api python -m unittest tests.test_planner -v
"""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.core.planner import (
    PlannerError,
    _build_system_prompt,
    _try_repair_json,
    plan as run_planner,
)
from src.core.types import KNOWN_INTENT_FLAGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PATCH_TARGET = "src.core.planner.build_chat_model"


def _flags(**overrides: bool) -> Dict[str, bool]:
    base = {f: False for f in KNOWN_INTENT_FLAGS}
    base.update(overrides)
    return base


def _good_plan_json(*, with_claim: bool = False) -> str:
    """A valid Plan JSON that passes both schema and registry validation."""
    if with_claim:
        plan = {
            "schema_version": "1.0",
            "goal": "Verify Tesla FSD claims",
            "rationale": "claim tracking on Tesla",
            "estimated_complexity": "high",
            "steps": [
                {
                    "id": 1,
                    "description": "Fetch Tesla SEC filings",
                    "agent": "filings_agent",
                    "tool_subset": [
                        "research__get_sec_filings",
                        "research__fetch_sec_document",
                    ],
                    "depends_on": [],
                    "max_tool_calls": 5,
                },
                {
                    "id": 2,
                    "description": "Search historical news on Tesla FSD",
                    "agent": "research_agent",
                    "tool_subset": ["research__search_historical_news"],
                    "depends_on": [],
                    "max_tool_calls": 4,
                },
                {
                    "id": 3,
                    "description": "Extract forward claims",
                    "agent": "claim_agent",
                    "tool_subset": ["research__extract_forward_claims"],
                    "depends_on": [1],
                    "max_tool_calls": 3,
                },
                {
                    "id": 4,
                    "description": "Compare claims to actuals",
                    "agent": "claim_agent",
                    "tool_subset": ["research__compare_claim_to_reality"],
                    "depends_on": [3, 2],
                    "max_tool_calls": 5,
                },
                {
                    "id": 5,
                    "description": "Synthesize verdict report",
                    "agent": "synthesizer",
                    "tool_subset": [],
                    "depends_on": [3, 4],
                    "max_tool_calls": 0,
                },
            ],
        }
    else:
        plan = {
            "schema_version": "1.0",
            "goal": "Topic research on Indian IT outlook",
            "rationale": "topic research with research_agent + synthesizer",
            "estimated_complexity": "medium",
            "steps": [
                {
                    "id": 1,
                    "description": "Pull recent news on Indian IT sector",
                    "agent": "research_agent",
                    "tool_subset": [
                        "research__search_news",
                        "research__search_web",
                    ],
                    "depends_on": [],
                    "max_tool_calls": 6,
                },
                {
                    "id": 2,
                    "description": "Write the structured topic brief",
                    "agent": "synthesizer",
                    "tool_subset": [],
                    "depends_on": [1],
                    "max_tool_calls": 0,
                },
            ],
        }
    return json.dumps(plan)


def _fake_llm_returning(*responses: str) -> MagicMock:
    """Build a fake chat model whose ainvoke returns the given strings in turn.

    Each successive call to ``llm.ainvoke(...)`` returns the next
    response. Once exhausted, the last response is repeated (so a test
    that expects one call doesn't fail on a second call by accident).
    """
    fake = MagicMock()
    iterator = iter(responses)
    last = ""

    async def _ainvoke(*_args, **_kwargs):
        nonlocal last
        try:
            last = next(iterator)
        except StopIteration:
            pass  # repeat last
        return AIMessage(content=last)

    fake.ainvoke = _ainvoke
    return fake


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------
class JSONRepairTests(unittest.TestCase):
    def test_pure_json_parses(self):
        raw = '{"a": 1, "b": [2, 3]}'
        self.assertEqual(_try_repair_json(raw), {"a": 1, "b": [2, 3]})

    def test_strips_markdown_fence(self):
        raw = '```json\n{"a": 1}\n```'
        self.assertEqual(_try_repair_json(raw), {"a": 1})

    def test_strips_bare_fence(self):
        raw = '```\n{"a": 1}\n```'
        self.assertEqual(_try_repair_json(raw), {"a": 1})

    def test_carves_out_first_object_after_prose(self):
        raw = 'Here is the plan you asked for:\n\n{"goal": "x"}\n\nLet me know!'
        self.assertEqual(_try_repair_json(raw), {"goal": "x"})

    def test_returns_none_for_unbalanced(self):
        self.assertIsNone(_try_repair_json("{"))

    def test_returns_none_for_empty(self):
        self.assertIsNone(_try_repair_json(""))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
class SystemPromptTests(unittest.TestCase):
    def test_prompt_contains_full_agent_catalog(self):
        from src.core.agents.registry import REGISTRY
        prompt = _build_system_prompt(_flags(), REGISTRY)
        for agent in REGISTRY:
            self.assertIn(f"`{agent.name}`", prompt,
                          f"agent {agent.name!r} missing from planner prompt")

    def test_prompt_lists_intent_flags(self):
        prompt = _build_system_prompt(
            _flags(wants_claim_tracking=True),
            __import__("src.core.agents.registry").core.agents.registry.REGISTRY,
        )
        self.assertIn("wants_claim_tracking", prompt)
        self.assertIn("True", prompt)
        # And the False flags appear too
        self.assertIn("wants_panel_debate", prompt)

    def test_prompt_includes_plan_schema(self):
        from src.core.agents.registry import REGISTRY
        prompt = _build_system_prompt(_flags(), REGISTRY)
        self.assertIn("Plan JSON schema", prompt)
        # Pydantic-generated schema should mention "PlanStep" anywhere
        self.assertIn("PlanStep", prompt)

    def test_prompt_includes_examples(self):
        from src.core.agents.registry import REGISTRY
        prompt = _build_system_prompt(_flags(), REGISTRY)
        self.assertIn("Example A", prompt)
        self.assertIn("Example B", prompt)
        self.assertIn("synthesizer", prompt)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class HappyPathTests(unittest.TestCase):
    def test_returns_valid_plan_on_first_try(self):
        with patch(
            _PATCH_TARGET,
            return_value=_fake_llm_returning(_good_plan_json()),
        ):
            plan = asyncio.run(run_planner(
                "Outlook for Indian IT?",
                intent_flags=_flags(),
            ))
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.steps[0].agent, "research_agent")
        self.assertEqual(plan.steps[-1].agent, "synthesizer")

    def test_strips_fenced_json(self):
        # Some models wrap the response in ```json ``` despite the prompt.
        fenced = "```json\n" + _good_plan_json() + "\n```"
        with patch(
            _PATCH_TARGET,
            return_value=_fake_llm_returning(fenced),
        ):
            plan = asyncio.run(run_planner(
                "Outlook for Indian IT?",
                intent_flags=_flags(),
            ))
        self.assertEqual(len(plan.steps), 2)

    def test_claim_tracking_plan_with_intent_flag_set(self):
        with patch(
            _PATCH_TARGET,
            return_value=_fake_llm_returning(_good_plan_json(with_claim=True)),
        ):
            plan = asyncio.run(run_planner(
                "Did Tesla deliver on FSD?",
                intent_flags=_flags(
                    wants_claim_tracking=True,
                    wants_filings=True,
                    wants_historical_news=True,
                ),
            ))
        self.assertEqual(len(plan.steps), 5)
        agents = [s.agent for s in plan.steps]
        self.assertIn("claim_agent", agents)
        self.assertEqual(agents[-1], "synthesizer")


# ---------------------------------------------------------------------------
# Retry / repair behaviour
# ---------------------------------------------------------------------------
class RetryTests(unittest.TestCase):
    def test_retries_on_invalid_json_then_succeeds(self):
        with patch(
            _PATCH_TARGET,
            return_value=_fake_llm_returning(
                "this is not json",          # 1st try fails parse
                _good_plan_json(),           # 2nd try succeeds
            ),
        ):
            plan = asyncio.run(run_planner(
                "Outlook for Indian IT?",
                intent_flags=_flags(),
            ))
        self.assertEqual(len(plan.steps), 2)

    def test_retries_on_schema_error_then_succeeds(self):
        # First response: a JSON object that doesn't match the Plan schema
        bad = json.dumps({"goal": "x", "rationale": "y"})  # missing steps
        with patch(
            _PATCH_TARGET,
            return_value=_fake_llm_returning(bad, _good_plan_json()),
        ):
            plan = asyncio.run(run_planner(
                "Outlook for Indian IT?",
                intent_flags=_flags(),
            ))
        self.assertEqual(len(plan.steps), 2)

    def test_retries_on_registry_error_then_succeeds(self):
        # First response: a plan that uses an unknown agent
        bad_plan = json.dumps({
            "schema_version": "1.0",
            "goal": "test",
            "rationale": "test",
            "estimated_complexity": "medium",
            "steps": [
                {
                    "id": 1,
                    "description": "do something",
                    "agent": "not_a_real_agent",  # registry rejects
                    "tool_subset": [],
                    "depends_on": [],
                    "max_tool_calls": 0,
                },
            ],
        })
        with patch(
            _PATCH_TARGET,
            return_value=_fake_llm_returning(bad_plan, _good_plan_json()),
        ):
            plan = asyncio.run(run_planner(
                "Outlook for Indian IT?",
                intent_flags=_flags(),
            ))
        self.assertEqual(len(plan.steps), 2)

    def test_raises_planner_error_after_exhausted_retries(self):
        # Every retry returns invalid JSON
        with patch(
            _PATCH_TARGET,
            return_value=_fake_llm_returning("garbage 1", "garbage 2", "garbage 3"),
        ):
            with self.assertRaises(PlannerError) as ctx:
                asyncio.run(run_planner(
                    "Outlook for Indian IT?",
                    intent_flags=_flags(),
                    retries=1,
                ))
        self.assertIn("did not return valid JSON", str(ctx.exception))
        self.assertEqual(ctx.exception.raw_output, "garbage 2")

    def test_planner_error_carries_validation_errors(self):
        # Schema-error every retry → exhausted → PlannerError with errors
        bad = json.dumps({"goal": "x", "rationale": "y"})  # missing steps
        with patch(
            _PATCH_TARGET,
            return_value=_fake_llm_returning(bad, bad, bad),
        ):
            with self.assertRaises(PlannerError) as ctx:
                asyncio.run(run_planner(
                    "Outlook for Indian IT?",
                    intent_flags=_flags(),
                    retries=1,
                ))
        # Exception carries the validation errors for diagnostics
        self.assertGreater(len(ctx.exception.validation_errors), 0)


# ---------------------------------------------------------------------------
# Policy-gate enforcement at the planner layer
# ---------------------------------------------------------------------------
class PolicyGateTests(unittest.TestCase):
    def test_claim_agent_plan_rejected_when_flag_false(self):
        # Plan uses claim_agent but intent_flags doesn't set
        # wants_claim_tracking. registry.validate_plan should reject.
        plan_json = _good_plan_json(with_claim=True)
        with patch(
            _PATCH_TARGET,
            # First reply uses claim_agent (rejected); second reply
            # is the topic-research plan (accepted).
            return_value=_fake_llm_returning(plan_json, _good_plan_json()),
        ):
            plan = asyncio.run(run_planner(
                "Did Tesla deliver on FSD?",
                intent_flags=_flags(),  # no flags set
            ))
        # Should have fallen back to the second response
        self.assertEqual(len(plan.steps), 2)
        self.assertNotIn("claim_agent", [s.agent for s in plan.steps])


if __name__ == "__main__":
    unittest.main(verbosity=2)
