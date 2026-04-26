"""Unit tests for src.core.agents._factories — Stage 1 of the slice.

Covers:

1. Each per-agent factory (research / filings / claim / synthesizer)
   produces a working ScopedAgent
2. Agent-appropriate model parameters are picked (temp / max_tokens)
3. The synthesizer factory uses ``system_prompt_override`` and the
   resulting prompt has user-facing framing (NOT "you are running ONE
   step of a larger plan")
4. Policy gates still apply through the factory layer (claim_agent
   blocked unless wants_claim_tracking=True)
5. ``build_scoped_agent_for_step`` dispatches correctly and raises a
   clean error for agents we haven't yet specialised (us_stock,
   indian_stock, portfolio, panel)

The factories internally call ``build_chat_model`` from
``src.agents.personas.base`` — that needs an NVIDIA API key. We
mock it out so tests run in any environment.

Run via::

    docker exec finai-api python -m unittest tests.test_factories -v
"""
from __future__ import annotations

import unittest
from typing import List
from unittest.mock import MagicMock, patch

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool


# Test-helper: bind_tools-compatible fake model. ``langgraph.create_react_agent``
# always calls ``model.bind_tools(...)`` at compile time and the upstream
# fake models raise NotImplementedError. We override to a no-op.
class _BindableFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


from src.core.agents._factories import (
    build_claim_agent,
    build_filings_agent,
    build_research_agent,
    build_scoped_agent_for_step,
    build_synthesizer,
)
from src.core.agents._base import ScopedAgent, ScopedAgentError
from src.core.agents.registry import REGISTRY
from src.core.types import KNOWN_INTENT_FLAGS, PlanStep, Scratchpad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flags(**overrides: bool) -> dict[str, bool]:
    base = {f: False for f in KNOWN_INTENT_FLAGS}
    base.update(overrides)
    return base


def _step(
    *,
    step_id: int = 1,
    agent: str,
    tool_subset: List[str],
    deps: List[int] | None = None,
    desc: str = "test step description",
) -> PlanStep:
    return PlanStep(
        id=step_id,
        description=desc,
        agent=agent,
        tool_subset=tool_subset,
        depends_on=deps or [],
    )


def _fake_tool(name: str) -> StructuredTool:
    """Stub MCP tool whose name matches one declared in the registry."""

    def _call(**_kwargs: any) -> str:
        return f"{name} called"

    return StructuredTool.from_function(
        func=_call,
        name=name,
        description=f"Stub for {name}",
    )


def _all_mcp_tools() -> List[StructuredTool]:
    """Stub of the entire 34-tool MCP pool — only names matter for tests."""
    names = []
    for agent_def in REGISTRY:
        names.extend(agent_def.tools)
    return [_fake_tool(n) for n in names]


def _fake_model() -> _BindableFakeModel:
    return _BindableFakeModel(
        responses=[AIMessage(content="placeholder")],
    )


# Convenience: every factory test patches build_chat_model to bypass NIM.
_PATCH_TARGET = "src.core.agents._factories.build_chat_model"


# ---------------------------------------------------------------------------
# build_research_agent
# ---------------------------------------------------------------------------
class BuildResearchAgentTests(unittest.TestCase):
    def test_returns_a_scoped_agent_for_research_agent(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()) as mock_build:
            agent = build_research_agent(
                step=_step(
                    agent="research_agent",
                    tool_subset=["research__search_news"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        self.assertIsInstance(agent, ScopedAgent)
        self.assertEqual(agent.agent_def.name, "research_agent")  # type: ignore[union-attr]
        self.assertEqual(len(agent.mcp_tools), 1)
        self.assertEqual(agent.mcp_tools[0].name, "research__search_news")
        # Default ScopedAgent system prompt (NOT a synthesizer override)
        self.assertIn("running step", agent.system_prompt)
        self.assertIn("ONE step of a larger plan", agent.system_prompt)
        # Built with research-appropriate params
        mock_build.assert_called_once()
        kwargs = mock_build.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.3)
        self.assertEqual(kwargs["max_tokens"], 1500)
        self.assertTrue(kwargs["streaming"])

    def test_default_intent_flags_works_for_ungated_agent(self):
        # research_agent has no policy gate; an empty intent_flags dict
        # should still produce a working ScopedAgent.
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_research_agent(
                step=_step(
                    agent="research_agent",
                    tool_subset=["research__search_web"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=None,  # default to empty
            )
        self.assertEqual(agent.agent_def.name, "research_agent")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# build_filings_agent
# ---------------------------------------------------------------------------
class BuildFilingsAgentTests(unittest.TestCase):
    def test_picks_filings_tools_only(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_filings_agent(
                step=_step(
                    agent="filings_agent",
                    tool_subset=[
                        "research__get_sec_filings",
                        "research__fetch_sec_document",
                    ],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        self.assertEqual(agent.agent_def.name, "filings_agent")  # type: ignore[union-attr]
        self.assertEqual(len(agent.mcp_tools), 2)

    def test_uses_filings_appropriate_model_params(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()) as mock_build:
            build_filings_agent(
                step=_step(
                    agent="filings_agent",
                    tool_subset=["research__get_sec_filings"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        kwargs = mock_build.call_args.kwargs
        # Lower temp + larger max_tokens than research (filings are long)
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_tokens"], 3000)


# ---------------------------------------------------------------------------
# build_claim_agent (the gated one)
# ---------------------------------------------------------------------------
class BuildClaimAgentTests(unittest.TestCase):
    def test_blocked_when_intent_flag_false(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            with self.assertRaises(ScopedAgentError) as ctx:
                build_claim_agent(
                    step=_step(
                        agent="claim_agent",
                        tool_subset=["research__extract_forward_claims"],
                    ),
                    scratchpad=Scratchpad(query="x"),
                    all_mcp_tools=_all_mcp_tools(),
                    intent_flags=_flags(wants_claim_tracking=False),
                )
        self.assertIn("policy-gated", str(ctx.exception))

    def test_blocked_when_intent_flag_missing(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            with self.assertRaises(ScopedAgentError):
                build_claim_agent(
                    step=_step(
                        agent="claim_agent",
                        tool_subset=["research__extract_forward_claims"],
                    ),
                    scratchpad=Scratchpad(query="x"),
                    all_mcp_tools=_all_mcp_tools(),
                    intent_flags=None,
                )

    def test_allowed_when_intent_flag_true(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_claim_agent(
                step=_step(
                    agent="claim_agent",
                    tool_subset=[
                        "research__extract_forward_claims",
                        "research__compare_claim_to_reality",
                    ],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(wants_claim_tracking=True),
            )
        self.assertEqual(agent.agent_def.name, "claim_agent")  # type: ignore[union-attr]
        self.assertEqual(len(agent.mcp_tools), 2)

    def test_uses_claim_appropriate_model_params(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()) as mock_build:
            build_claim_agent(
                step=_step(
                    agent="claim_agent",
                    tool_subset=["research__extract_forward_claims"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(wants_claim_tracking=True),
            )
        kwargs = mock_build.call_args.kwargs
        # Lowest temp (claim verdicts are structured), modest tokens
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_tokens"], 2000)


# ---------------------------------------------------------------------------
# build_synthesizer (the special one)
# ---------------------------------------------------------------------------
class BuildSynthesizerTests(unittest.TestCase):
    def test_synthesizer_uses_custom_system_prompt(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_synthesizer(
                step=_step(
                    agent="synthesizer",
                    tool_subset=[],
                    deps=[1, 2],
                    desc="Produce the final report combining the verdicts.",
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        # The synthesizer prompt should NOT use the default "running ONE
        # step of a larger plan" framing.
        prompt = agent.system_prompt
        self.assertNotIn("ONE step of a larger plan", prompt)
        # It SHOULD frame as the final agent.
        self.assertIn("FINAL", prompt)
        self.assertIn("FinAI Synthesizer", prompt)
        # The step description threads through
        self.assertIn("Produce the final report combining the verdicts.", prompt)
        # Declared deps render in the prompt so the LLM knows which step
        # IDs it can fetch
        self.assertIn("[1, 2]", prompt)

    def test_synthesizer_has_no_mcp_tools(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_synthesizer(
                step=_step(
                    agent="synthesizer",
                    tool_subset=[],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        # Synthesizer is LLM-only - no MCP tools.
        self.assertEqual(agent.mcp_tools, [])
        # But it DOES still get the synthetic tools (get_prior_result,
        # request_assistance) from the base class.
        self.assertEqual(len(agent.synthetic_tools), 2)

    def test_synthesizer_uses_largest_token_budget(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()) as mock_build:
            build_synthesizer(
                step=_step(
                    agent="synthesizer",
                    tool_subset=[],
                    deps=[1],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        kwargs = mock_build.call_args.kwargs
        # Synthesizer produces the user-visible report - largest budget.
        self.assertEqual(kwargs["max_tokens"], 4000)

    def test_synthesizer_prompt_forbids_recommendations_and_disclaimers(self):
        # Hard rules in the prompt that protect the demo's compliance posture.
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_synthesizer(
                step=_step(agent="synthesizer", tool_subset=[]),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        prompt = agent.system_prompt
        self.assertIn("DO NOT recommend buying or selling", prompt)
        self.assertIn("DO NOT include a regulatory disclaimer", prompt)
        self.assertIn("DO NOT fabricate numbers", prompt)


# ---------------------------------------------------------------------------
# Dispatcher: build_scoped_agent_for_step
# ---------------------------------------------------------------------------
class DispatcherTests(unittest.TestCase):
    def test_dispatches_to_research_agent(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_scoped_agent_for_step(
                step=_step(
                    agent="research_agent",
                    tool_subset=["research__search_news"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        self.assertEqual(agent.agent_def.name, "research_agent")  # type: ignore[union-attr]
        # Default prompt (not synthesizer override)
        self.assertIn("ONE step of a larger plan", agent.system_prompt)

    def test_dispatches_to_synthesizer_with_override(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_scoped_agent_for_step(
                step=_step(
                    agent="synthesizer",
                    tool_subset=[],
                    deps=[1, 2, 3],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        self.assertEqual(agent.agent_def.name, "synthesizer")  # type: ignore[union-attr]
        # Synthesizer override applies
        self.assertNotIn("ONE step of a larger plan", agent.system_prompt)
        self.assertIn("FinAI Synthesizer", agent.system_prompt)

    def test_dispatches_to_filings_and_claim(self):
        # Quick sanity that all 4 stage-1 factories are wired.
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            for agent_name, tool, flags in [
                ("filings_agent", "research__get_sec_filings", _flags()),
                (
                    "claim_agent",
                    "research__extract_forward_claims",
                    _flags(wants_claim_tracking=True),
                ),
            ]:
                with self.subTest(agent=agent_name):
                    agent = build_scoped_agent_for_step(
                        step=_step(agent=agent_name, tool_subset=[tool]),
                        scratchpad=Scratchpad(query="x"),
                        all_mcp_tools=_all_mcp_tools(),
                        intent_flags=flags,
                    )
                    self.assertEqual(agent.agent_def.name, agent_name)  # type: ignore[union-attr]

    def test_unknown_agent_raises_clean_error(self):
        # us_stock_agent / indian_stock / portfolio / panel are NOT in
        # stage 1's _FACTORY_MAP - they land in stage 4 of the slice.
        with self.assertRaises(ScopedAgentError) as ctx:
            build_scoped_agent_for_step(
                step=_step(
                    agent="us_stock_agent",
                    tool_subset=["us_stock__get_quote"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        msg = str(ctx.exception)
        self.assertIn("No factory registered", msg)
        self.assertIn("us_stock_agent", msg)
        # The error message lists the agents we DO have, so the planner
        # can self-correct.
        self.assertIn("synthesizer", msg)

    def test_dispatcher_propagates_gate_failure(self):
        # claim_agent without the flag should propagate ScopedAgentError
        # from the underlying factory.
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            with self.assertRaises(ScopedAgentError) as ctx:
                build_scoped_agent_for_step(
                    step=_step(
                        agent="claim_agent",
                        tool_subset=["research__extract_forward_claims"],
                    ),
                    scratchpad=Scratchpad(query="x"),
                    all_mcp_tools=_all_mcp_tools(),
                    intent_flags=_flags(wants_claim_tracking=False),
                )
        self.assertIn("policy-gated", str(ctx.exception))


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------
class CrossCuttingTests(unittest.TestCase):
    def test_all_stage1_agents_in_factory_map(self):
        # The dispatcher's map should know about exactly the 4 agent names
        # we need for the claim-tracker slice. If a 5th lands without an
        # entry, this surfaces immediately.
        from src.core.agents._factories import _FACTORY_MAP
        self.assertEqual(
            set(_FACTORY_MAP.keys()),
            {"research_agent", "filings_agent", "claim_agent", "synthesizer"},
        )

    def test_factory_propagates_recursion_limit(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_research_agent(
                step=_step(
                    agent="research_agent",
                    tool_subset=["research__search_news"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
                recursion_limit=42,
            )
        self.assertEqual(agent.recursion_limit, 42)


if __name__ == "__main__":
    unittest.main(verbosity=2)
