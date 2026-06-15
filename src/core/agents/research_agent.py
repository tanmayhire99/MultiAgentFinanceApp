"""Research Agent — general web / news / company-brief gathering.

Registered as :data:`src.core.agents.registry.RESEARCH_AGENT`; reached at
runtime via :func:`src.core.agents.factory_dispatch.build_scoped_agent_for_step`.
This file owns only the runtime specialisation (model params); the
registry owns the agent's tools / description / role.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents import _model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_research_agent(
    *,
    step: PlanStep, scratchpad: Scratchpad, all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None, registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary", recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    user_id: str = "demo",
) -> ScopedAgent:
    """ScopedAgent specialised for general web/news research steps.

    Higher temperature than analytical agents because we want some
    fluency in news/source summarisation, but bounded so the agent
    doesn't drift off the user's question.
    """
    model = _model.build_chat_model(
        temperature=0.3, max_tokens=1500, streaming=True,
        api_key_slot=api_key_slot, cycle_keys=True,
    )
    return ScopedAgent(
        step=step, scratchpad=scratchpad, all_mcp_tools=all_mcp_tools,
        model=model, registry=registry, intent_flags=intent_flags,
        recursion_limit=recursion_limit, user_id=user_id,
    )


__all__ = ["build_research_agent"]
