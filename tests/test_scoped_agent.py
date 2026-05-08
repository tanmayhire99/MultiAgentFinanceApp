"""Unit tests for src.core.agents._base — Day 3 of the migration.

ScopedAgent is the per-step runtime wrapper. These tests exercise:

1. Construction validates the step against the registry (gates work)
2. Tool filtering is a strict allow-list
3. Synthetic tools (``get_prior_result`` / ``request_assistance``)
   behave correctly without invoking any LLM
4. The system prompt embeds the catalog + step-specific scoping rules
5. End-to-end ``run()`` with a fake chat model returns a StepResult

We deliberately use ``FakeMessagesListChatModel`` for the run-level
tests so we don't depend on a live LLM / API key. The fake model
returns pre-built ``AIMessage`` objects in sequence; we feed it one
final answer and let the ReAct loop finish on the first turn.

Run via::

    docker exec finai-api python -m unittest tests.test_scoped_agent -v
"""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any, List

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool


class _BindableFakeModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel that supports ``bind_tools`` as a no-op.

    ``langgraph.prebuilt.create_react_agent`` always calls
    ``model.bind_tools(...)`` during compilation, but the upstream fake
    models raise ``NotImplementedError``. Since we hand-craft the
    response stream (including any tool_call AIMessages) the binding
    step is irrelevant to the test - we just need the call to succeed.
    """

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self

from src.core.agents._base import (
    DEFAULT_RECURSION_LIMIT,
    ScopedAgent,
    ScopedAgentError,
)
from src.core.agents.registry import REGISTRY
from src.core.types import (
    KNOWN_INTENT_FLAGS,
    PlanStep,
    Scratchpad,
    StepResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flags(**overrides: bool) -> dict[str, bool]:
    base = {f: False for f in KNOWN_INTENT_FLAGS}
    base.update(overrides)
    return base


def _step(
    step_id: int = 1,
    *,
    agent: str = "research_agent",
    tool_subset: List[str] | None = None,
    deps: List[int] | None = None,
    desc: str = "Look up recent news on Apple.",
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=desc,
        agent=agent,
        tool_subset=tool_subset or [],
        depends_on=deps or [],
    )


def _fake_tool(name: str, *, description: str = "stub tool") -> StructuredTool:
    """Build a no-op StructuredTool that pretends to be a real MCP tool.

    We only need its ``name`` for filter/lookup tests; the actual
    callable is never invoked in the unit-test path.
    """

    def _call(**_kwargs: Any) -> str:
        return f"{name} called"

    return StructuredTool.from_function(
        func=_call,
        name=name,
        description=description,
    )


def _make_research_pool() -> List[StructuredTool]:
    """Six fake research-namespaced tools to mirror the real MCP pool."""
    return [
        _fake_tool("research__search_news"),
        _fake_tool("research__search_web"),
        _fake_tool("research__get_company_brief"),
        _fake_tool("research__list_supported_research_topics"),
        _fake_tool("us_stock__get_quote"),
        _fake_tool("us_stock__get_fundamentals"),
    ]


def _fake_model(*responses: AIMessage) -> _BindableFakeModel:
    """A bind_tools-friendly fake model pre-loaded with N responses."""
    return _BindableFakeModel(responses=list(responses))


# ---------------------------------------------------------------------------
# Construction & registry validation
# ---------------------------------------------------------------------------
class ConstructionTests(unittest.TestCase):
    def test_valid_step_constructs(self):
        sa = ScopedAgent(
            step=_step(
                tool_subset=["research__search_news"],
            ),
            scratchpad=Scratchpad(query="apple news"),
            all_mcp_tools=_make_research_pool(),
            model=_fake_model(AIMessage(content="ok")),
            intent_flags=_flags(),
        )
        self.assertIsNotNone(sa.agent_def)
        self.assertEqual(sa.agent_def.name, "research_agent")
        self.assertEqual(len(sa.mcp_tools), 1)
        self.assertEqual(sa.mcp_tools[0].name, "research__search_news")

    def test_unknown_agent_rejected(self):
        with self.assertRaises(ScopedAgentError) as ctx:
            ScopedAgent(
                step=_step(agent="not_a_real_agent"),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_make_research_pool(),
                model=_fake_model(AIMessage(content="ok")),
                intent_flags=_flags(),
            )
        self.assertIn("unknown agent", str(ctx.exception))

    def test_tool_not_owned_by_agent_rejected(self):
        with self.assertRaises(ScopedAgentError) as ctx:
            ScopedAgent(
                # research_agent does NOT own us_stock__get_quote
                step=_step(
                    agent="research_agent",
                    tool_subset=["us_stock__get_quote"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_make_research_pool(),
                model=_fake_model(AIMessage(content="ok")),
                intent_flags=_flags(),
            )
        self.assertIn("not owned by agent", str(ctx.exception))

    def test_gated_agent_blocked_without_intent_flag(self):
        # claim_agent requires wants_claim_tracking=True
        with self.assertRaises(ScopedAgentError) as ctx:
            ScopedAgent(
                step=_step(
                    agent="claim_agent",
                    tool_subset=["research__extract_forward_claims"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=[
                    _fake_tool("research__extract_forward_claims"),
                ],
                model=_fake_model(AIMessage(content="ok")),
                intent_flags=_flags(),  # all flags False
            )
        self.assertIn("policy-gated", str(ctx.exception))

    def test_gated_agent_allowed_with_intent_flag(self):
        # Same as above but with the right flag set
        sa = ScopedAgent(
            step=_step(
                agent="claim_agent",
                tool_subset=["research__extract_forward_claims"],
            ),
            scratchpad=Scratchpad(query="x"),
            all_mcp_tools=[
                _fake_tool("research__extract_forward_claims"),
                _fake_tool("research__compare_claim_to_reality"),
            ],
            model=_fake_model(AIMessage(content="ok")),
            intent_flags=_flags(wants_claim_tracking=True),
        )
        self.assertEqual(sa.agent_def.name, "claim_agent")


# ---------------------------------------------------------------------------
# Strict allow-list tool filtering
# ---------------------------------------------------------------------------
class ToolFilteringTests(unittest.TestCase):
    def test_empty_subset_yields_empty_mcp_list(self):
        sa = ScopedAgent(
            step=_step(tool_subset=[]),  # no MCP tools
            scratchpad=Scratchpad(query="x"),
            all_mcp_tools=_make_research_pool(),
            model=_fake_model(AIMessage(content="ok")),
            intent_flags=_flags(),
        )
        self.assertEqual(sa.mcp_tools, [])
        # Synthetic tools should still exist
        self.assertEqual(len(sa.synthetic_tools), 3)

    def test_subset_preserves_order(self):
        sa = ScopedAgent(
            step=_step(
                tool_subset=[
                    "research__get_company_brief",
                    "research__search_news",
                ],
            ),
            scratchpad=Scratchpad(query="x"),
            all_mcp_tools=_make_research_pool(),
            model=_fake_model(AIMessage(content="ok")),
            intent_flags=_flags(),
        )
        names = [t.name for t in sa.mcp_tools]
        self.assertEqual(names, [
            "research__get_company_brief",
            "research__search_news",
        ])

    def test_missing_tool_raises_at_construction(self):
        with self.assertRaises(ScopedAgentError) as ctx:
            ScopedAgent(
                step=_step(
                    tool_subset=["research__search_news"],  # in pool
                ),
                scratchpad=Scratchpad(query="x"),
                # Pool deliberately missing the requested tool
                all_mcp_tools=[_fake_tool("research__search_web")],
                model=_fake_model(AIMessage(content="ok")),
                intent_flags=_flags(),
            )
        # The structural validation in registry runs first (catches "not
        # owned by registry" against the canonical 34-tool catalog), so
        # we either see the registry's "not owned by agent" message OR
        # the MCP-pool "not present" message. Either is acceptable -
        # both block the bug. Check for either.
        msg = str(ctx.exception)
        self.assertTrue(
            "not present in the MCP pool" in msg
            or "not owned by agent" in msg,
            f"Unexpected error message: {msg}",
        )


# ---------------------------------------------------------------------------
# Synthetic tools (no LLM required)
# ---------------------------------------------------------------------------
class GetPriorResultTests(unittest.TestCase):
    """Exercise the get_prior_result synthetic tool in isolation.

    StructuredTool exposes ``.invoke({...})`` for sync use.
    """

    def _make_agent(self, *, deps: List[int] = None) -> ScopedAgent:
        return ScopedAgent(
            step=_step(deps=deps or [], tool_subset=[]),
            scratchpad=self._sp,
            all_mcp_tools=_make_research_pool(),
            model=_fake_model(AIMessage(content="ok")),
            intent_flags=_flags(),
        )

    def setUp(self):
        self._sp = Scratchpad(query="x")
        # Step 5 has completed
        self._sp.add(StepResult(
            step_id=5,
            status="complete",
            output={"text": "Apple's Q4 revenue was $89B."},
            tools_used=["research__search_news"],
        ))
        # Step 7 failed
        self._sp.add(StepResult(
            step_id=7,
            status="failed",
            output=None,
            error="API timeout",
            error_type="TimeoutError",
        ))

    def _get_tool(self, agent: ScopedAgent) -> StructuredTool:
        return next(
            t for t in agent.synthetic_tools if t.name == "get_prior_result"
        )

    def test_returns_output_for_declared_dep(self):
        agent = self._make_agent(deps=[5])
        tool = self._get_tool(agent)
        result = tool.invoke({"step_id": 5})
        parsed = json.loads(result)
        self.assertEqual(parsed["status"], "complete")
        self.assertEqual(parsed["output"]["text"], "Apple's Q4 revenue was $89B.")
        self.assertEqual(parsed["tools_used"], ["research__search_news"])

    def test_blocks_undeclared_step_id(self):
        # Step 5 exists in scratchpad but is NOT in this step's depends_on
        agent = self._make_agent(deps=[7])  # only depends on step 7
        tool = self._get_tool(agent)
        result = tool.invoke({"step_id": 5})
        parsed = json.loads(result)
        self.assertEqual(parsed["error"], "out_of_scope")
        self.assertIn("not in this step's declared dependencies", parsed["message"])

    def test_handles_missing_step_result(self):
        # Step 99 isn't in the scratchpad at all
        agent = self._make_agent(deps=[99])
        tool = self._get_tool(agent)
        result = tool.invoke({"step_id": 99})
        parsed = json.loads(result)
        self.assertEqual(parsed["error"], "not_yet_executed")

    def test_handles_failed_dep(self):
        agent = self._make_agent(deps=[7])
        tool = self._get_tool(agent)
        result = tool.invoke({"step_id": 7})
        parsed = json.loads(result)
        self.assertEqual(parsed["error"], "step_not_complete")
        self.assertEqual(parsed["step_status"], "failed")
        self.assertEqual(parsed["step_error"], "API timeout")
        self.assertEqual(parsed["step_error_type"], "TimeoutError")

    def test_no_deps_explicit_message_in_description(self):
        agent = self._make_agent(deps=[])
        tool = self._get_tool(agent)
        # The tool's description should mention that this step has NO deps
        # so the LLM doesn't waste a turn calling get_prior_result.
        self.assertIn("NO declared dependencies", tool.description)


class RequestAssistanceTests(unittest.TestCase):
    def setUp(self):
        self._sp = Scratchpad(query="x")
        self._agent = ScopedAgent(
            step=_step(step_id=3, tool_subset=[]),
            scratchpad=self._sp,
            all_mcp_tools=_make_research_pool(),
            model=_fake_model(AIMessage(content="ok")),
            intent_flags=_flags(),
        )
        self._tool = next(
            t for t in self._agent.synthetic_tools if t.name == "request_assistance"
        )

    def test_writes_to_scratchpad(self):
        self.assertEqual(self._sp.unmet_dependencies, [])
        result = self._tool.invoke({
            "target_agent": "us_stock_agent",
            "reason": "need a current AAPL quote to size the recommendation",
        })
        # Scratchpad has a new entry
        self.assertEqual(len(self._sp.unmet_dependencies), 1)
        ud = self._sp.unmet_dependencies[0]
        self.assertEqual(ud.requested_by_step_id, 3)
        self.assertEqual(ud.target_agent, "us_stock_agent")
        self.assertIn("AAPL quote", ud.reason)
        # Tool returns a confirmation message
        parsed = json.loads(result)
        self.assertTrue(parsed["recorded"])
        self.assertEqual(parsed["target_agent"], "us_stock_agent")

    def test_validates_reason_too_short(self):
        # Pydantic args_schema enforces min_length=10 on `reason`
        with self.assertRaises(Exception):
            self._tool.invoke({
                "target_agent": "us_stock_agent",
                "reason": "short",  # 5 chars
            })

    def test_multiple_calls_each_recorded(self):
        for i in range(3):
            self._tool.invoke({
                "target_agent": "us_stock_agent",
                "reason": f"call number {i+1} needs more context here",
            })
        self.assertEqual(len(self._sp.unmet_dependencies), 3)


# ---------------------------------------------------------------------------
# run_python synthetic tool
# ---------------------------------------------------------------------------
class RunPythonTests(unittest.TestCase):
    def _make_agent(self) -> ScopedAgent:
        return ScopedAgent(
            step=_step(tool_subset=[]),
            scratchpad=Scratchpad(query="x"),
            all_mcp_tools=_make_research_pool(),
            model=_fake_model(AIMessage(content="ok")),
            intent_flags=_flags(),
        )

    def _get_tool(self, agent: ScopedAgent) -> StructuredTool:
        return next(
            t for t in agent.synthetic_tools if t.name == "run_python"
        )

    def test_basic_arithmetic(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({"code": "result = 2 + 3 * 4"}))
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], 14)

    def test_print_output(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({"code": "print(42)"}))
        self.assertTrue(out["success"])
        self.assertEqual(out["stdout"], "42")

    def test_math_module(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({
            "code": "from math import comb\nresult = comb(52, 5)"
        }))
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], 2598960)

    def test_decimal_precision(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({
            "code": (
                "from decimal import Decimal, ROUND_HALF_UP\n"
                "d = Decimal('1.005').quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\n"
                "result = str(d)"
            )
        }))
        self.assertTrue(out["success"])
        self.assertEqual(out["result"], "1.01")

    def test_error_returns_traceback(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({"code": "1 / 0"}))
        self.assertFalse(out["success"])
        self.assertIn("ZeroDivisionError", out["error"])

    def test_syntax_error_returns_traceback(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({"code": "if True"}))
        self.assertFalse(out["success"])
        self.assertIn("SyntaxError", out["error"])

    def test_no_file_access(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({"code": "open('/etc/passwd')"}))
        self.assertFalse(out["success"])

    def test_no_import_os(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({"code": "import os"}))
        self.assertFalse(out["success"])

    def test_financial_ratio_computation(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({
            "code": (
                "net_income = 96995  # AAPL 2023 net income (millions)\n"
                "revenue = 383285    # AAPL 2023 revenue (millions)\n"
                "profit_margin = round(net_income / revenue * 100, 2)\n"
                "print(f'Net profit margin: {profit_margin}%')\n"
                "result = profit_margin"
            )
        }))
        self.assertTrue(out["success"])
        self.assertAlmostEqual(out["result"], 25.31, places=2)

    def test_compound_growth(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({
            "code": (
                "from decimal import Decimal\n"
                "initial = Decimal('100')\n"
                "rate = Decimal('0.08')\n"
                "years = 10\n"
                "final = initial * (1 + rate) ** years\n"
                "result = round(float(final), 2)"
            )
        }))
        self.assertTrue(out["success"])
        self.assertAlmostEqual(out["result"], 215.89, places=2)

    def test_converts_result_to_json(self):
        tool = self._get_tool(self._make_agent())
        out = json.loads(tool.invoke({
            "code": "result = {'pe': 28.5, 'eps': 6.42}"
        }))
        self.assertTrue(out["success"])
        self.assertEqual(out["result"]["pe"], 28.5)


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------
class SystemPromptTests(unittest.TestCase):
    def _build(
        self,
        *,
        deps: List[int] = None,
        tool_subset: List[str] = None,
        agent: str = "research_agent",
        intent_flags: dict = None,
        desc: str = "Look up recent news on Apple.",
    ) -> str:
        return ScopedAgent(
            step=_step(
                agent=agent,
                tool_subset=tool_subset or [],
                deps=deps or [],
                desc=desc,
            ),
            scratchpad=Scratchpad(query="x"),
            all_mcp_tools=_make_research_pool() + [
                _fake_tool("research__extract_forward_claims"),
                _fake_tool("research__compare_claim_to_reality"),
            ],
            model=_fake_model(AIMessage(content="ok")),
            intent_flags=intent_flags or _flags(),
        ).system_prompt

    def test_contains_step_description(self):
        prompt = self._build(desc="Investigate NVDA's H100 supply chain.")
        self.assertIn("Investigate NVDA's H100 supply chain", prompt)

    def test_contains_agent_name_and_title(self):
        prompt = self._build()
        # research_agent maps to "Research Agent"
        agent_def = REGISTRY.get("research_agent")
        self.assertIn(agent_def.name, prompt)
        self.assertIn(agent_def.title, prompt)

    def test_contains_tool_subset_list(self):
        prompt = self._build(tool_subset=["research__search_news"])
        self.assertIn("research__search_news", prompt)

    def test_contains_depends_on_list(self):
        prompt = self._build(deps=[2, 5])
        # Sorted form
        self.assertIn("[2, 5]", prompt)

    def test_empty_deps_renders_empty_list(self):
        prompt = self._build(deps=[])
        # Should render `[]` cleanly somewhere
        self.assertIn("[]", prompt)

    def test_contains_full_agent_catalog(self):
        prompt = self._build()
        # All 8 agents should be mentioned
        for agent_def in REGISTRY:
            self.assertIn(f"`{agent_def.name}`", prompt,
                          f"agent {agent_def.name!r} missing from prompt")

    def test_warns_against_other_agents(self):
        # The "stay in your lane" guidance is critical for the architecture
        prompt = self._build()
        # Multiple distinct phrasings - check at least one
        self.assertTrue(
            "Stay strictly in your lane" in prompt
            or "Do NOT try to do another agent's" in prompt,
            "Prompt should warn the agent not to do other agents' jobs",
        )

    def test_explains_request_assistance_mechanism(self):
        prompt = self._build()
        self.assertIn("request_assistance", prompt)
        self.assertIn("get_prior_result", prompt)

    def test_mentions_run_python(self):
        prompt = self._build()
        self.assertIn("run_python", prompt)


# ---------------------------------------------------------------------------
# Trajectory parsing
# ---------------------------------------------------------------------------
class TrajectoryHelperTests(unittest.TestCase):
    def test_extract_final_text_str_content(self):
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="here's my answer"),
        ]
        self.assertEqual(
            ScopedAgent._extract_final_text(msgs),
            "here's my answer",
        )

    def test_extract_final_text_list_content(self):
        msgs = [
            AIMessage(content=[
                {"type": "text", "text": "first part"},
                {"type": "text", "text": "second part"},
            ]),
        ]
        self.assertEqual(
            ScopedAgent._extract_final_text(msgs),
            "first part\nsecond part",
        )

    def test_extract_final_text_skips_empty(self):
        msgs = [
            AIMessage(content="real answer"),
            AIMessage(content=""),  # later but empty
        ]
        # Should fall through to the "real answer" message
        self.assertEqual(
            ScopedAgent._extract_final_text(msgs),
            "real answer",
        )

    def test_extract_final_text_no_ai_message(self):
        msgs = [HumanMessage(content="hi")]
        self.assertEqual(ScopedAgent._extract_final_text(msgs), "")

    def test_collect_tools_used_distinct_in_order(self):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tool_a", "args": {}, "id": "1"},
                    {"name": "tool_b", "args": {}, "id": "2"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "tool_a", "args": {}, "id": "3"},  # duplicate
                    {"name": "tool_c", "args": {}, "id": "4"},
                ],
            ),
        ]
        self.assertEqual(
            ScopedAgent._collect_tools_used(msgs),
            ["tool_a", "tool_b", "tool_c"],
        )


# ---------------------------------------------------------------------------
# End-to-end run() with a fake model
# ---------------------------------------------------------------------------
class RunIntegrationTests(unittest.TestCase):
    """The fake model returns a single AIMessage with no tool calls, which
    ends the ReAct loop on turn 1. This exercises the wiring end-to-end
    without depending on a real LLM.
    """

    def _run(self, agent: ScopedAgent) -> StepResult:
        # Consume the async generator to get the final result
        async def _collect_result():
            async for event in agent.run():
                if event.get("type") == "_step_result":
                    return event.get("result")
            return None
            
        return asyncio.run(_collect_result())

    def test_run_returns_complete_step_result(self):
        sa = ScopedAgent(
            step=_step(
                step_id=4,
                tool_subset=["research__search_news"],
                desc="Find recent Apple headlines.",
            ),
            scratchpad=Scratchpad(query="apple news"),
            all_mcp_tools=_make_research_pool(),
            model=_fake_model(
                AIMessage(content="Apple announced a new MacBook Pro in October."),
            ),
            intent_flags=_flags(),
        )
        result = self._run(sa)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.step_id, 4)
        self.assertIn("MacBook Pro", result.output["text"])
        self.assertIsNotNone(result.completed_at)
        self.assertIsNotNone(result.duration_s)
        # No tools called by the fake model on turn 1
        self.assertEqual(result.tools_used, [])

    def test_run_captures_crash_as_failed_status(self):
        # FakeMessagesListChatModel raises if responses run out and the
        # graph asks for another. We give it zero responses so the very
        # first call raises, and verify that becomes status='failed'.
        sa = ScopedAgent(
            step=_step(step_id=9, tool_subset=[]),
            scratchpad=Scratchpad(query="x"),
            all_mcp_tools=_make_research_pool(),
            model=_fake_model(),  # empty responses -> will raise
            intent_flags=_flags(),
        )
        result = self._run(sa)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.step_id, 9)
        self.assertIsNone(result.output)
        self.assertIsNotNone(result.error)
        self.assertIsNotNone(result.error_type)


if __name__ == "__main__":
    unittest.main(verbosity=2)
