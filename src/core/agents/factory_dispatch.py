"""Dispatch from ``step.agent`` (string) → the right per-agent factory.

This is the **single entry point** the executor uses to construct a
ScopedAgent for a given :class:`~src.core.types.PlanStep`. The executor
asks ``build_scoped_agent_for_step(step=..., ...)`` and gets back a
configured :class:`~src.core.agents._base.ScopedAgent` (or, for the
panel agent, a :class:`~src.core.agents.panel_agent.PanelScopedAgent`).

Adding a new agent
------------------
1. Register the agent in :mod:`src.core.agents.registry`.
2. Create ``src/core/agents/<name>_agent.py`` with a ``build_<name>_agent``
   factory (the synthesizer is named after its registry entry).
3. Import the factory here and add it to :data:`_FACTORY_MAP`.
4. Add a focused unit test in ``tests/test_factories.py``.

The cross-cutting test asserts that :data:`_FACTORY_MAP` keys exactly
equal ``{a.name for a in REGISTRY}`` — so a registry agent without a
factory (or vice versa) surfaces immediately.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents._base import (
    DEFAULT_RECURSION_LIMIT,
    ScopedAgent,
    ScopedAgentError,
)
from src.core.agents.claim_agent import build_claim_agent
from src.core.agents.filings_agent import build_filings_agent
from src.core.agents.indian_stock_agent import build_indian_stock_agent
from src.core.agents.panel_agent import build_panel_agent
from src.core.agents.portfolio_agent import build_portfolio_agent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.agents.research_agent import build_research_agent
from src.core.agents.synthesizer import build_synthesizer
from src.core.agents.us_stock_agent import build_us_stock_agent
from src.core.types import PlanStep, Scratchpad


# ---------------------------------------------------------------------------
# Factory map
# ---------------------------------------------------------------------------
# Maps an agent's snake_case ``name`` (matching the registry) to its
# factory function. Keys MUST equal ``{a.name for a in REGISTRY}`` — the
# cross-cutting test enforces this so a new registry agent without a
# factory (or vice versa) surfaces immediately in CI.
_FactoryFn = Callable[..., ScopedAgent]
_FACTORY_MAP: Dict[str, _FactoryFn] = {
    "research_agent": build_research_agent,
    "filings_agent": build_filings_agent,
    "us_stock_agent": build_us_stock_agent,
    "indian_stock_agent": build_indian_stock_agent,
    "portfolio_agent": build_portfolio_agent,
    "claim_agent": build_claim_agent,
    "synthesizer": build_synthesizer,
    "panel_agent": build_panel_agent,
}


def build_scoped_agent_for_step(
    *, step: PlanStep, scratchpad: Scratchpad, all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None, registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary", recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    user_id: str = "demo",
) -> ScopedAgent:
    """Dispatch on ``step.agent`` and call the right factory.

    Raises :class:`ScopedAgentError` if no factory is registered for
    ``step.agent``. This will happen if a planner produces a step
    naming an agent we haven't yet specialised - the error message
    lists every known factory so the planner can self-correct on the
    next retry.
    """
    factory = _FACTORY_MAP.get(step.agent)
    if factory is None:
        registered = sorted(_FACTORY_MAP.keys())
        raise ScopedAgentError(
            f"No factory registered for agent {step.agent!r}. "
            f"Available factories: {registered}. "
            "Either register the agent or fix the planner's output."
        )
    return factory(
        step=step,
        scratchpad=scratchpad,
        all_mcp_tools=all_mcp_tools,
        intent_flags=intent_flags,
        registry=registry,
        api_key_slot=api_key_slot,
        recursion_limit=recursion_limit,
        user_id=user_id,
    )


__all__ = [
    "build_scoped_agent_for_step",
    "_FACTORY_MAP",
]
