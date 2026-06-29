"""Quant Strategy Agent — read-only systematic F&O backtesting.

Reached via :func:`src.core.agents.factory_dispatch.build_scoped_agent_for_step`
only when the opt-in quant MCP server (sibling automated-trading project) is
enabled. It is a standard ReAct :class:`ScopedAgent` over the read-only quant
tools (``quant__list_strategies`` / ``quant__backtest_strategy``) — there is no
execution surface to reach.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents import _model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_quant_agent(
    *,
    step: PlanStep, scratchpad: Scratchpad, all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None, registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary", recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    user_id: str = "demo",
) -> ScopedAgent:
    """ScopedAgent for read-only NIFTY F&O strategy backtesting.

    Low temperature (0.1): its job is to call the backtest tool and report the
    returned metrics faithfully, not to invent numbers.
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


__all__ = ["build_quant_agent"]
