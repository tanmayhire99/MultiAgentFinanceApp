"""Unit tests for the per-agent factories + factory_dispatch — Stages 1 + 4.

Stage 1 (Day 4) covers the four agents the **claim-tracker** slice
needed: research / filings / claim / synthesizer.

Stage 4 (Day 4b) adds the four agents the **panel** slice needs:
us_stock / indian_stock / portfolio / panel. After this stage every
agent in :data:`REGISTRY` has a factory, so the planner can emit
plans naming any of the 8 agents and the executor will construct
them cleanly. Tests in this file cover both.

Covers:

1. Each per-agent factory produces a working ScopedAgent (or, for
   ``panel_agent``, a :class:`PanelScopedAgent` subclass).
2. Agent-appropriate model parameters are picked (temp / max_tokens).
3. The synthesizer factory uses ``system_prompt_override`` and the
   resulting prompt has user-facing framing (NOT "you are running ONE
   step of a larger plan").
4. Policy gates still apply through the factory layer:
    * claim_agent blocked unless ``wants_claim_tracking=True``
    * panel_agent blocked unless ``wants_panel_debate=True``
5. ``build_scoped_agent_for_step`` dispatches correctly to all 8
   agents and raises a clean error for agents not in the registry.
6. :class:`PanelScopedAgent.run` delegates to the debate machinery
   and returns a complete :class:`StepResult`.

The factories internally call ``build_chat_model`` from
``src.personas.base`` — that needs an NVIDIA API key. We
mock it out so tests run in any environment. The panel agent's
``run`` likewise has every dependency mocked (``run_debate_loop``,
the moderator-synthesis chat model, etc.) so no live LLM is
required.

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


from src.core.agents.claim_agent import build_claim_agent
from src.core.agents.filings_agent import build_filings_agent
from src.core.agents.indian_stock_agent import build_indian_stock_agent
from src.core.agents.panel_agent import PanelScopedAgent, build_panel_agent
from src.core.agents.portfolio_agent import build_portfolio_agent
from src.core.agents.research_agent import build_research_agent
from src.core.agents.synthesizer import build_synthesizer
from src.core.agents.us_stock_agent import build_us_stock_agent
from src.core.agents.factory_dispatch import build_scoped_agent_for_step
from src.core.agents._base import ScopedAgent, ScopedAgentError
from src.core.agents.registry import REGISTRY
from src.core.types import KNOWN_INTENT_FLAGS, PlanStep, Scratchpad, StepResult


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
_PATCH_TARGET = "src.core.agents._model.build_chat_model"


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
        self.assertEqual(len(agent.synthetic_tools), 3)

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
        # As of Stage 4, every registry agent has a factory. To still
        # exercise the "missing factory" guard, we hand the dispatcher
        # a step whose ``agent`` name is neither in the registry nor in
        # the factory map. ``PlanStep`` itself is permissive about the
        # name (it's just a non-empty string at the schema level), so a
        # planner producing a hallucinated agent name lands here at
        # construction time with a clean diagnostic.
        with self.assertRaises(ScopedAgentError) as ctx:
            build_scoped_agent_for_step(
                step=_step(
                    agent="ghost_agent_404",
                    tool_subset=[],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        msg = str(ctx.exception)
        self.assertIn("No factory registered", msg)
        self.assertIn("ghost_agent_404", msg)
        # The error message lists the agents we DO have so the planner
        # can self-correct. After Stage 4 every registry agent shows up.
        for canonical in (
            "research_agent",
            "filings_agent",
            "us_stock_agent",
            "indian_stock_agent",
            "portfolio_agent",
            "claim_agent",
            "synthesizer",
            "panel_agent",
        ):
            self.assertIn(canonical, msg)

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
# build_us_stock_agent / build_indian_stock_agent (Stage 4 — panel slice)
# ---------------------------------------------------------------------------
class BuildUsStockAgentTests(unittest.TestCase):
    def test_picks_us_stock_tools_only(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_us_stock_agent(
                step=_step(
                    agent="us_stock_agent",
                    tool_subset=[
                        "us_stock__get_quote",
                        "us_stock__get_fundamentals",
                    ],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        self.assertIsInstance(agent, ScopedAgent)
        self.assertEqual(agent.agent_def.name, "us_stock_agent")  # type: ignore[union-attr]
        self.assertEqual(len(agent.mcp_tools), 2)
        # Default ScopedAgent system prompt (NOT a synthesizer override)
        self.assertIn("ONE step of a larger plan", agent.system_prompt)

    def test_uses_us_stock_appropriate_model_params(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()) as mock_build:
            build_us_stock_agent(
                step=_step(
                    agent="us_stock_agent",
                    tool_subset=["us_stock__get_quote"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        kwargs = mock_build.call_args.kwargs
        # Lowest temperature (numeric extraction; no rephrasing wanted)
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_tokens"], 1500)
        self.assertTrue(kwargs["streaming"])

    def test_us_stock_no_policy_gate(self):
        # us_stock_agent has no policy gate, so it constructs even with
        # an empty / None intent_flags dict.
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_us_stock_agent(
                step=_step(
                    agent="us_stock_agent",
                    tool_subset=["us_stock__get_quote"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=None,
            )
        self.assertEqual(agent.agent_def.name, "us_stock_agent")  # type: ignore[union-attr]


class BuildIndianStockAgentTests(unittest.TestCase):
    def test_picks_indian_stock_tools_only(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_indian_stock_agent(
                step=_step(
                    agent="indian_stock_agent",
                    tool_subset=[
                        "indian_stock__get_quote",
                        "indian_stock__get_fundamentals",
                    ],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        self.assertIsInstance(agent, ScopedAgent)
        self.assertEqual(agent.agent_def.name, "indian_stock_agent")  # type: ignore[union-attr]
        self.assertEqual(len(agent.mcp_tools), 2)

    def test_indian_stock_uses_same_model_params_as_us(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()) as mock_build:
            build_indian_stock_agent(
                step=_step(
                    agent="indian_stock_agent",
                    tool_subset=["indian_stock__get_quote"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        kwargs = mock_build.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_tokens"], 1500)


# ---------------------------------------------------------------------------
# build_portfolio_agent (Stage 4 — panel slice)
# ---------------------------------------------------------------------------
class BuildPortfolioAgentTests(unittest.TestCase):
    def test_picks_portfolio_tools_only(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_portfolio_agent(
                step=_step(
                    agent="portfolio_agent",
                    tool_subset=[
                        "portfolio__get_holdings",
                        "portfolio__get_portfolio_summary",
                    ],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        self.assertIsInstance(agent, ScopedAgent)
        self.assertEqual(agent.agent_def.name, "portfolio_agent")  # type: ignore[union-attr]
        self.assertEqual(len(agent.mcp_tools), 2)

    def test_uses_portfolio_appropriate_model_params(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()) as mock_build:
            build_portfolio_agent(
                step=_step(
                    agent="portfolio_agent",
                    tool_subset=["portfolio__get_holdings"],
                ),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(),
            )
        kwargs = mock_build.call_args.kwargs
        # Deterministic Python is the source of truth; LLM should NOT
        # paraphrase or extrapolate, so we pin temperature to 0.1.
        self.assertEqual(kwargs["temperature"], 0.1)
        self.assertEqual(kwargs["max_tokens"], 1500)


# ---------------------------------------------------------------------------
# build_panel_agent (Stage 4 — special-cased; gated)
# ---------------------------------------------------------------------------
class BuildPanelAgentTests(unittest.TestCase):
    def test_blocked_when_intent_flag_false(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            with self.assertRaises(ScopedAgentError) as ctx:
                build_panel_agent(
                    step=_step(
                        agent="panel_agent",
                        tool_subset=[],
                    ),
                    scratchpad=Scratchpad(query="x"),
                    all_mcp_tools=_all_mcp_tools(),
                    intent_flags=_flags(wants_panel_debate=False),
                )
        self.assertIn("policy-gated", str(ctx.exception))

    def test_blocked_when_intent_flag_missing(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            with self.assertRaises(ScopedAgentError):
                build_panel_agent(
                    step=_step(agent="panel_agent", tool_subset=[]),
                    scratchpad=Scratchpad(query="x"),
                    all_mcp_tools=_all_mcp_tools(),
                    intent_flags=None,
                )

    def test_returns_panel_scoped_agent_when_flag_true(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_panel_agent(
                step=_step(agent="panel_agent", tool_subset=[]),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(wants_panel_debate=True),
            )
        # PanelScopedAgent is a subclass of ScopedAgent, so the executor
        # can treat it identically (both expose .run() returning a
        # StepResult).
        self.assertIsInstance(agent, ScopedAgent)
        self.assertIsInstance(agent, PanelScopedAgent)
        self.assertEqual(agent.agent_def.name, "panel_agent")  # type: ignore[union-attr]
        # panel_agent owns no MCP tools at the planner level.
        self.assertEqual(agent.mcp_tools, [])
        # Synthetic tools (get_prior_result + request_assistance) still
        # come through the base class.
        self.assertEqual(len(agent.synthetic_tools), 3)

    def test_panel_uses_moderator_appropriate_params(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()) as mock_build:
            build_panel_agent(
                step=_step(agent="panel_agent", tool_subset=[]),
                scratchpad=Scratchpad(query="x"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(wants_panel_debate=True),
            )
        kwargs = mock_build.call_args.kwargs
        # Matches the moderator-synthesis params used by the static
        # portfolio_analysis flow's closing-brief LLM call.
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertEqual(kwargs["max_tokens"], 1100)


# ---------------------------------------------------------------------------
# PanelScopedAgent.run delegation (Stage 4 — special-cased)
#
# The override is the architecturally novel piece of Day 4b: instead of
# running the standard ReAct loop, the panel agent drives
# ``run_debate_loop`` and the moderator-synthesis chat call. We mock
# both so the test runs in <50ms with no network and assert that the
# resulting StepResult is well-shaped.
# ---------------------------------------------------------------------------
def _fake_async_iter(events: list):
    """Async generator yielding the supplied events in order."""

    async def _gen(*args, **kwargs):
        for ev in events:
            yield ev

    return _gen


class _FakeScratchpadEntry:
    """Stand-in for src.core.debate.ScratchpadEntry — duck-typed."""

    def __init__(self, *, persona, persona_title, round, stance,
                 one_liner="", confidence="medium", content=""):
        self.persona = persona
        self.persona_title = persona_title
        self.round = round
        self.stance = stance
        self.one_liner = one_liner
        self.confidence = confidence
        self.content = content


class _FakePanelScratchpad:
    """Stand-in for src.core.debate.PanelScratchpad — only the bits
    PanelScopedAgent.run reads."""

    def __init__(self, entries):
        self.entries = entries

    def stance_evolution_md(self):
        return "_(stance evolution rendered)_"


class PanelScopedAgentRunTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for ``PanelScopedAgent.run`` with mocked deps."""

    async def test_run_delegates_to_debate_loop_and_synthesizer(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_panel_agent(
                step=_step(
                    step_id=3,
                    agent="panel_agent",
                    tool_subset=[],
                    desc="Debate the user's portfolio",
                ),
                scratchpad=Scratchpad(query="What do you think of my portfolio?"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(wants_panel_debate=True),
            )

        fake_pad = _FakePanelScratchpad(entries=[
            _FakeScratchpadEntry(
                persona="buffett", persona_title="Warren Buffett",
                round=1, stance="cautious", one_liner="Solid, but rich.",
            ),
            _FakeScratchpadEntry(
                persona="wood", persona_title="Cathie Wood",
                round=1, stance="bullish", one_liner="Innovation cycle ahead.",
            ),
        ])

        fake_loop = _fake_async_iter([
            {"type": "header", "text": "### Round 1\n\n"},
            {"type": "text", "text": "Buffett spoke first.\n"},
            {"type": "text", "text": "Wood replied.\n"},
            {"type": "_debate_done", "scratchpad": fake_pad},
        ])

        result = None
        with patch("src.core.debate.run_debate_loop", new=fake_loop), \
             patch("src.personas.base.build_chat_model", return_value=_fake_model()):
            async for event in agent.run():
                if event.get("type") == "_step_result":
                    result = event.get("result")
                    break

        self.assertIsInstance(result, StepResult)
        self.assertEqual(result.status, "complete")
        self.assertIsInstance(result.output, dict)

        text = result.output["text"]
        self.assertIn("Panel Conversation Summary", text)

        self.assertIn("panel_debate_loop", result.tools_used)
        self.assertIn("moderator_synthesis", result.tools_used)

    async def test_run_handles_debate_loop_crash_gracefully(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_panel_agent(
                step=_step(agent="panel_agent", tool_subset=[]),
                scratchpad=Scratchpad(query="anything"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(wants_panel_debate=True),
            )

        async def _crashing_iter(*args, **kwargs):
            if False:
                yield
            raise RuntimeError("debate boom")

        result = None
        with patch("src.core.debate.run_debate_loop", new=_crashing_iter), \
             patch("src.personas.base.build_chat_model", return_value=_fake_model()):
            async for event in agent.run():
                if event.get("type") == "_step_result":
                    result = event.get("result")
                    break

        self.assertEqual(result.status, "failed")
        self.assertIn("debate boom", str(result.error))

    async def test_run_with_no_scratchpad_emits_apologetic_brief(self):
        with patch(_PATCH_TARGET, return_value=_fake_model()):
            agent = build_panel_agent(
                step=_step(agent="panel_agent", tool_subset=[]),
                scratchpad=Scratchpad(query="anything"),
                all_mcp_tools=_all_mcp_tools(),
                intent_flags=_flags(wants_panel_debate=True),
            )

        fake_loop = _fake_async_iter([
            {"type": "text", "text": "An empty debate."},
        ])

        result = None
        with patch("src.core.debate.run_debate_loop", new=fake_loop), \
             patch("src.personas.base.build_chat_model", return_value=_fake_model()):
            async for event in agent.run():
                if event.get("type") == "_step_result":
                    result = event.get("result")
                    break

        self.assertEqual(result.status, "complete")
        self.assertIn("Panel Conversation Summary", result.output["text"])
        self.assertIn(
            "did not produce a scratchpad", result.output["text"],
        )


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------
class CrossCuttingTests(unittest.TestCase):
    def test_factory_map_covers_every_registered_agent(self):
        # Day 4b invariant: every agent in REGISTRY has a factory.
        # If a 9th agent is added to the registry without a matching
        # factory, this surfaces immediately.
        from src.core.agents.factory_dispatch import _FACTORY_MAP
        self.assertEqual(
            set(_FACTORY_MAP.keys()),
            {a.name for a in REGISTRY},
        )

    def test_factory_map_has_exactly_eight_factories(self):
        from src.core.agents.factory_dispatch import _FACTORY_MAP
        self.assertEqual(
            set(_FACTORY_MAP.keys()),
            {
                "research_agent",
                "filings_agent",
                "us_stock_agent",
                "indian_stock_agent",
                "portfolio_agent",
                "claim_agent",
                "synthesizer",
                "panel_agent",
            },
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
