"""Filings Agent — SEC / Indian filings extraction.

Registered as :data:`src.core.agents.registry.FILINGS_AGENT`; reached at
runtime via :func:`src.core.agents.factory_dispatch.build_scoped_agent_for_step`.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents import _model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_filings_agent(
    *,
    step: PlanStep, scratchpad: Scratchpad, all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None, registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary", recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    user_id: str = "demo",
) -> ScopedAgent:
    """ScopedAgent specialised for SEC / Indian filings extraction.

    Filings work needs longer max_tokens (10-Ks and concall transcripts
    are dense) and a much lower temperature (we want analytical
    extraction, not paraphrase).
    """
    model = _model.build_chat_model(
        temperature=0.1, max_tokens=3000, streaming=True,
        api_key_slot=api_key_slot, cycle_keys=True,
    )
    return ScopedAgent(
        step=step, scratchpad=scratchpad, all_mcp_tools=all_mcp_tools,
        model=model, registry=registry, intent_flags=intent_flags,
        recursion_limit=recursion_limit, user_id=user_id,
    )


__all__ = ["build_filings_agent"]
