"""Agent layer of the planner-first architecture.

Public surface:

* :class:`PolicyGate` — intent-flag precondition for gated agents
* :class:`AgentDefinition` — one row of the catalog
* :class:`AgentRegistry` — query / validate against the catalog
* :data:`REGISTRY` — the canonical instance built at import time

See ``docs/MULTI_AGENT_ARCHITECTURE.md`` for the architectural rationale.
"""
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
]
