"""US Stock Agent — live US-equity quotes, fundamentals, growth + defensive metrics.

Owns the ``us_stock__*`` namespaced MCP tool family (get_quote,
get_fundamentals, get_growth_metrics, get_defensive_metrics,
get_moat_signals). The agent's job is to call those tools and
surface the **structured numeric data** they return, with a short
prose summary — NOT to invent metrics or paraphrase to vagueness.

Architecture
------------
* **Registered** in :data:`src.core.agents.registry.US_STOCK_AGENT`.
* **Constructed** via :func:`build_us_stock_agent` below.
* **Runs** the standard :class:`~src.core.agents._base.ScopedAgent`
  ReAct loop with the planner-chosen ``tool_subset``.

Mirror of :mod:`src.core.agents.indian_stock_agent`. The two agents
are kept distinct so the registry's per-agent tool ownership stays a
single source of truth (``us_stock_agent.tools`` vs
``indian_stock_agent.tools`` partition the namespaced MCP tool
space) and so a future divergence (currency conversion narration,
NSE-specific quirks, etc.) has a clean home.

Model parameters
----------------
* **Temperature 0.1** — lowest setting; we want the LLM to repeat
  the tool's numbers verbatim, not round or rephrase.
* **max_tokens 1500** — fundamentals tables are short (~10 metrics)
  and the agent's prose summary should fit in a paragraph or two.

Tests: ``tests/test_factories.py::BuildUsStockAgentTests``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.personas.base import build_chat_model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_us_stock_agent(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None,
    registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ScopedAgent:
    """ScopedAgent specialised for US-equity quote / fundamentals fetching."""
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


__all__ = ["build_us_stock_agent"]
