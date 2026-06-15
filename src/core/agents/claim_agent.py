"""Claim Agent — claim extraction + comparison (policy-gated).

Registered as :data:`src.core.agents.registry.CLAIM_AGENT`; reached at
runtime via :func:`src.core.agents.factory_dispatch.build_scoped_agent_for_step`.

This agent is **policy-gated**: ``ScopedAgent.__init__`` raises
:class:`ScopedAgentError` unless ``intent_flags["wants_claim_tracking"]``
is True. The gate is enforced uniformly in the registry, not here.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents import _model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_claim_agent(
    *,
    step: PlanStep, scratchpad: Scratchpad, all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None, registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary", recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    user_id: str = "demo",
) -> ScopedAgent:
    """ScopedAgent for claim extraction + comparison.

    This agent is **policy-gated**: ``ScopedAgent.__init__`` will raise
    :class:`ScopedAgentError` unless ``intent_flags["wants_claim_tracking"]``
    is True. The factory itself does no extra gate work - the registry
    enforcement runs uniformly on every ScopedAgent construction.

    Tight temperature (0.1) because claim verdicts are
    structured outputs (claim text + verdict label + evidence cite),  not free-form prose.
    """
    model = _model.build_chat_model(
        temperature=0.1, max_tokens=2000, streaming=True,
        api_key_slot=api_key_slot, cycle_keys=True,
    )
    return ScopedAgent(
        step=step, scratchpad=scratchpad, all_mcp_tools=all_mcp_tools,
        model=model, registry=registry, intent_flags=intent_flags,
        recursion_limit=recursion_limit, user_id=user_id,
    )


__all__ = ["build_claim_agent"]
