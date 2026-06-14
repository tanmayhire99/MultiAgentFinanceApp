"""Portfolio Agent — holdings + analytics for the (mocked) user portfolio.

Owns the ``portfolio__*`` namespaced MCP tools (list_supported_users,
get_holdings, get_portfolio_summary, get_sector_allocation,
get_concentration_risks, get_diversification_score). These tools
return **deterministic Python summaries** computed by
:mod:`src.agents.workers.portfolio_mcp` over a fixture broker
account — there is no LLM-derived number anywhere in this layer.

The agent's job is to call the right tools and present the results;
it should NOT recompute or extrapolate, as the deterministic Python
is the source of truth.

Architecture
------------
* **Registered** in :data:`src.core.agents.registry.PORTFOLIO_AGENT`.
* **Constructed** via :func:`build_portfolio_agent` below.
* **Runs** the standard :class:`~src.core.agents._base.ScopedAgent`
  ReAct loop with the planner-chosen ``tool_subset``.

Model parameters
----------------
* **Temperature 0.1** — deterministic-Python is the source of truth;
  the LLM should NOT paraphrase or extrapolate.
* **max_tokens 1500** — the tools' outputs are already well-shaped
  JSON; the agent just needs a paragraph or two of narration.

Tests: ``tests/test_factories.py::BuildPortfolioAgentTests``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.personas.base import build_chat_model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_portfolio_agent(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None,
    registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ScopedAgent:
    """ScopedAgent specialised for portfolio holdings + analytics."""
    model = build_chat_model(
        temperature=0.1,
        max_tokens=1500,
        streaming=True,
        api_key_slot=api_key_slot,
    )
    return ScopedAgent(
        step=step,
        scratchpad=scratchpad,
        all_mcp_tools=all_mcp_tools,
        model=model,
        registry=registry,
        intent_flags=intent_flags,
        recursion_limit=recursion_limit,
    )


__all__ = ["build_portfolio_agent"]
