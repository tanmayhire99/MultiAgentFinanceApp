"""Indian Stock Agent — live NSE / BSE quotes, fundamentals, growth + defensive metrics.

Owns the ``indian_stock__*`` namespaced MCP tool family. Same shape
as :mod:`src.core.agents.us_stock_agent` but for tickers listed on
NSE / BSE (TCS, INFY, RELIANCE, HDFCBANK, ITC, etc.). All
currency-denominated fields are converted to USD by the underlying
worker so panel synthesis can compare apples-to-apples across US
and Indian holdings; the native currency is reported in the result
for transparency.

Architecture
------------
* **Registered** in :data:`src.core.agents.registry.INDIAN_STOCK_AGENT`.
* **Constructed** via :func:`build_indian_stock_agent` below.
* **Runs** the standard :class:`~src.core.agents._base.ScopedAgent`
  ReAct loop with the planner-chosen ``tool_subset``.

Mirrors :mod:`us_stock_agent` rather than sharing a parametrised
factory because the registry's tool ownership map partitions the
two namespaces (``us_stock__*`` vs ``indian_stock__*``) — keeping
two factory files lets a future divergence (currency conversion
narration, NSE-specific quirks) land in the right place without
disturbing the US flow.

Model parameters
----------------
Identical to the US stock agent: ``temperature=0.1``,
``max_tokens=1500``, ``streaming=True``.

Tests: ``tests/test_factories.py::BuildIndianStockAgentTests``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.personas.base import build_chat_model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_indian_stock_agent(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None,
    registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ScopedAgent:
    """ScopedAgent specialised for NSE / BSE quote / fundamentals fetching."""
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


__all__ = ["build_indian_stock_agent"]
