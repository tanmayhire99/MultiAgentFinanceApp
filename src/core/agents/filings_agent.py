"""Filings Agent — SEC EDGAR + Indian regulatory filings extraction.

Owns the document-oriented ``research__get_sec_filings``,
``research__fetch_sec_document``, ``research__get_indian_filings``,
``research__fetch_indian_document``, ``research__get_screener_snapshot``,
``research__get_indian_concall_urls``, and
``research__get_indian_annual_reports`` MCP tools. The agent's job
is to **pull primary-source filings**, extract the relevant text or
metadata, and surface it for downstream steps to reason over.
Layout-aware PDF extraction is handled inside the workers; this
agent doesn't need to know.

Architecture
------------
* **Registered** in :data:`src.core.agents.registry.FILINGS_AGENT`.
* **Constructed** via :func:`build_filings_agent` below.
* **Runs** the standard :class:`~src.core.agents._base.ScopedAgent`
  ReAct loop with the planner-chosen ``tool_subset``.

Model parameters
----------------
Filings work needs **longer max_tokens (3000)** because 10-Ks and
concall transcripts are dense, and a **much lower temperature
(0.1)** because we want analytical extraction, not paraphrase.

Tests: ``tests/test_factories.py::BuildFilingsAgentTests``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.personas.base import build_chat_model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_filings_agent(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None,
    registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ScopedAgent:
    """ScopedAgent specialised for SEC / Indian filings extraction.

    Filings work needs longer max_tokens (10-Ks and concall transcripts
    are dense) and a much lower temperature (we want analytical
    extraction, not paraphrase).
    """
    model = build_chat_model(
        temperature=0.1,
        max_tokens=3000,
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


__all__ = ["build_filings_agent"]
