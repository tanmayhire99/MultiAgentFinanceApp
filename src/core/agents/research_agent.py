"""Research Agent — web / news / company-brief gathering.

The research agent's job is to **collect external textual context**
about a ticker, sector, or topic via the namespaced ``research__*``
MCP tools (search_news, search_web, get_company_brief, etc.). It
does NOT compute valuations and it does NOT extract or verify
forward-looking claims — those are owned by the stock and claim
agents respectively.

Architecture
------------
* **Registered** in :data:`src.core.agents.registry.RESEARCH_AGENT`.
  The registry is the source of truth for the agent's description,
  owned tools, and role hint. This file owns the *runtime
  specialisation* (model parameters, prompt overrides) only.
* **Constructed** via :func:`build_research_agent` below. The
  executor reaches this through
  :func:`src.core.agents.factory_dispatch.build_scoped_agent_for_step`,
  which looks the factory up by ``step.agent`` from
  :data:`~src.core.agents.factory_dispatch._FACTORY_MAP`.
* **Runs** the standard ReAct loop implemented in
  :class:`src.core.agents._base.ScopedAgent` with the planner-chosen
  ``tool_subset``. No prompt override — the default per-step prompt
  framing ("you are running ONE step of a larger plan") is
  appropriate for an intermediate research step.

Model parameters
----------------
* **Temperature 0.3** — slight creative latitude for fluent news /
  source summarisation, bounded so the agent doesn't drift off the
  user's question.
* **max_tokens 1500** — modest budget; the agent's job is to surface
  primary-source links + a short summary, not to write long-form
  prose. Long-form writing is the synthesizer's job.
* ``streaming=True`` so chunks reach the SSE stream as they arrive.

Tests live in ``tests/test_factories.py::BuildResearchAgentTests``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.personas.base import build_chat_model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_research_agent(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None,
    registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ScopedAgent:
    """ScopedAgent specialised for general web/news research steps.

    Higher temperature than analytical agents because we want some
    fluency in news/source summarisation, but bounded so the agent
    doesn't drift off the user's question.
    """
    model = build_chat_model(
        temperature=0.3,
        max_tokens=1500,
        streaming=True,
        api_key_slot=api_key_slot,
    )
    return ScopedAgent(
        step=step,
        scratchpad=scratchpad,
        all_mcp_tools=all_mcp_tools,
        model=model,
        registry=registry,
        intent_flags=intent_flags,
        recursion_limit=recursion_limit,
    )


__all__ = ["build_research_agent"]
