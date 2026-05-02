"""Agent layer of the planner-first architecture.

Public surface:

* :class:`PolicyGate` — intent-flag precondition for gated agents
* :class:`AgentDefinition` — one row of the catalog
* :class:`AgentRegistry` — query / validate against the catalog
* :data:`REGISTRY` — the canonical instance built at import time
* :class:`ScopedAgent` — single-step runtime wrapper (Day 3+)
* :class:`PanelScopedAgent` — debate-orchestrating subclass (Day 4b+)
* :func:`build_scoped_agent_for_step` — factory dispatch
  (8 agents fully covered as of Day 4b — every registry agent has
  a factory, including the panel slice's ``us_stock_agent``,
  ``indian_stock_agent``, ``portfolio_agent``, ``panel_agent``).

See ``docs/MULTI_AGENT_ARCHITECTURE.md`` for the architectural rationale.
"""
from ._base import (
    DEFAULT_RECURSION_LIMIT,
    ScopedAgent,
    ScopedAgentError,
)
from ._panel_agent import PanelScopedAgent
from ._factories import (
    build_claim_agent,
    build_filings_agent,
    build_indian_stock_agent,
    build_panel_agent,
    build_portfolio_agent,
    build_research_agent,
    build_scoped_agent_for_step,
    build_synthesizer,
    build_us_stock_agent,
)
from .registry import (
    REGISTRY,
    AgentDefinition,
    AgentRegistry,
    CLAIM_AGENT,
    FILINGS_AGENT,
    INDIAN_STOCK_AGENT,
    PANEL_AGENT,
    PORTFOLIO_AGENT,
    PolicyGate,
    RESEARCH_AGENT,
    SYNTHESIZER,
    US_STOCK_AGENT,
)

__all__ = [
    # registry
    "REGISTRY",
    "AgentDefinition",
    "AgentRegistry",
    "PolicyGate",
    "CLAIM_AGENT",
    "FILINGS_AGENT",
    "INDIAN_STOCK_AGENT",
    "PANEL_AGENT",
    "PORTFOLIO_AGENT",
    "RESEARCH_AGENT",
    "SYNTHESIZER",
    "US_STOCK_AGENT",
    # scoped agent runtime
    "DEFAULT_RECURSION_LIMIT",
    "ScopedAgent",
    "ScopedAgentError",
    "PanelScopedAgent",
    # factories
    "build_research_agent",
    "build_filings_agent",
    "build_us_stock_agent",
    "build_indian_stock_agent",
    "build_portfolio_agent",
    "build_claim_agent",
    "build_synthesizer",
    "build_panel_agent",
    "build_scoped_agent_for_step",
]
