"""US Stock Agent — US-equity quote / fundamentals fetching.

Registered as :data:`src.core.agents.registry.US_STOCK_AGENT`; reached at
runtime via :func:`src.core.agents.factory_dispatch.build_scoped_agent_for_step`.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents import _model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_us_stock_agent(
    *,
    step: PlanStep, scratchpad: Scratchpad, all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None, registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary", recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    user_id: str = "demo",
) -> ScopedAgent:
    """ScopedAgent specialised for US-equity quote / fundamentals fetching.

    The stock agents (US + Indian) wrap MCP tools that return
    structured numeric data. The agent's job is to call the right
    tools, surface the results, and add a short factual summary -
    NOT to invent numbers or paraphrase to vagueness. Hence:

    * **Lowest temperature (0.1)** — we want the LLM to repeat the
    tool's numbers verbatim, not to round or rephrase.
    * **Modest max_tokens (1500)** — fundamentals tables are short
    (~10 metrics) and the agent's prose summary should fit in a
    paragraph or two.

    Same shape as :func:`build_indian_stock_agent` so a market-agnostic
    refactor can fold them together later if we ever decide to.
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


__all__ = ["build_us_stock_agent"]
