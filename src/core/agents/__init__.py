"""Agent layer of the planner-first architecture.

Layout
------
After the Day 4c per-agent-files refactor, the agent layer is
**one file per concrete agent** plus a small set of shared modules:

::

    src/core/agents/
    ├── __init__.py            # public re-exports (this file)
    ├── _base.py               # ScopedAgent runtime + ScopedAgentError
    ├── registry.py            # PolicyGate / AgentDefinition / AgentRegistry / REGISTRY
    ├── factory_dispatch.py    # _FACTORY_MAP + build_scoped_agent_for_step
    ├── research_agent.py      # build_research_agent
    ├── us_stock_agent.py      # build_us_stock_agent
    ├── indian_stock_agent.py  # build_indian_stock_agent
    ├── filings_agent.py       # build_filings_agent
    ├── portfolio_agent.py     # build_portfolio_agent
    ├── claim_agent.py         # build_claim_agent (policy-gated)
    ├── synthesizer.py         # build_synthesizer + _SYNTHESIZER_PROMPT_TEMPLATE
    └── panel_agent.py         # PanelScopedAgent + build_panel_agent
                               # + _DEBATE_SYNTH_SYSTEM
                               # + _format_scratchpad_for_moderator

Public surface (re-exported below)
----------------------------------
* :class:`PolicyGate` — intent-flag precondition for gated agents
* :class:`AgentDefinition` — one row of the catalog
* :class:`AgentRegistry` — query / validate against the catalog
* :data:`REGISTRY` — the canonical instance built at import time
* :class:`ScopedAgent` — single-step runtime wrapper
* :class:`PanelScopedAgent` — debate-orchestrating subclass
* :func:`build_scoped_agent_for_step` — factory dispatch
* Per-agent factories: ``build_<name>_agent`` for each of the 8
  registered agents

See ``docs/MULTI_AGENT_ARCHITECTURE.md`` for the architectural
rationale and ``docs/migration/`` for the migration log.
"""
from src.core.agents._base import (
    DEFAULT_RECURSION_LIMIT,
    ScopedAgent,
    ScopedAgentError,
)
from src.core.agents.claim_agent import build_claim_agent
from src.core.agents.factory_dispatch import build_scoped_agent_for_step
from src.core.agents.filings_agent import build_filings_agent
from src.core.agents.indian_stock_agent import build_indian_stock_agent
from src.core.agents.panel_agent import PanelScopedAgent, build_panel_agent
from src.core.agents.portfolio_agent import build_portfolio_agent
from src.core.agents.registry import (
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
from src.core.agents.research_agent import build_research_agent
from src.core.agents.synthesizer import build_synthesizer
from src.core.agents.us_stock_agent import build_us_stock_agent

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
    # per-agent factories
    "build_research_agent",
    "build_filings_agent",
    "build_us_stock_agent",
    "build_indian_stock_agent",
    "build_portfolio_agent",
    "build_claim_agent",
    "build_synthesizer",
    "build_panel_agent",
    # dispatcher
    "build_scoped_agent_for_step",
]
