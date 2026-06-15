"""Indian Stock Agent — NSE / BSE quote / fundamentals fetching.

Registered as :data:`src.core.agents.registry.INDIAN_STOCK_AGENT`; reached at
runtime via :func:`src.core.agents.factory_dispatch.build_scoped_agent_for_step`.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents import _model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_indian_stock_agent(
    *,
    step: PlanStep, scratchpad: Scratchpad, all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None, registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary", recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    user_id: str = "demo",
) -> ScopedAgent:
    """ScopedAgent specialised for NSE / BSE quote / fundamentals fetching.

    Mirror of :func:`build_us_stock_agent`. The two factories are kept
    distinct so the registry's per-agent tool ownership stays a single
    source of truth (``us_stock_agent.tools`` vs
    ``indian_stock_agent.tools`` partition the namespaced MCP tool
    space) and so a future divergence (currency conversion narration,
    NSE-specific quirks, etc.) has a clean home.
    """
    model = _model.build_chat_model(
        temperature=0.1, max_tokens=1500, streaming=True,
        api_key_slot=api_key_slot, cycle_keys=True,
    )
    return ScopedAgent(
        step=step, scratchpad=scratchpad, all_mcp_tools=all_mcp_tools,
        model=model, registry=registry, intent_flags=intent_flags,
        recursion_limit=recursion_limit, user_id=user_id,
    )


__all__ = ["build_indian_stock_agent"]
