"""Agent layer of the planner-first architecture.

Public surface:

* :class:`PolicyGate` — intent-flag precondition for gated agents
* :class:`AgentDefinition` — one row of the catalog
* :class:`AgentRegistry` — query / validate against the catalog
* :data:`REGISTRY` — the canonical instance built at import time
* :class:`ScopedAgent` — single-step runtime wrapper (Day 3+)
* :func:`build_scoped_agent_for_step` — factory dispatch (Day 4+)

See ``docs/MULTI_AGENT_ARCHITECTURE.md`` for the architectural rationale.
"""
from ._base import (
    DEFAULT_RECURSION_LIMIT,
    ScopedAgent,
    ScopedAgentError,
)
from ._factories import (
    build_claim_agent,
    build_filings_agent,
    build_research_agent,
    build_scoped_agent_for_step,
    build_synthesizer,
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
    # factories
    "build_research_agent",
    "build_filings_agent",
    "build_claim_agent",
    "build_synthesizer",
    "build_scoped_agent_for_step",
]
