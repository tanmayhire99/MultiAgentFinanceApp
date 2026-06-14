"""Claim Agent — forward-claim extraction + verdict-vs-reality (gated).

Owns the ``research__extract_forward_claims`` and
``research__compare_claim_to_reality`` MCP tools. The agent
extracts forward-looking claims from corporate documents
(transcripts, 10-Ks, press releases) and produces verdicts comparing
each claim against the latest evidence.

Policy gate
-----------
This agent is **policy-gated** by the registry: the planner is only
permitted to include ``claim_agent`` in a plan when the classifier
LLM (Phase 1) has set ``intent_flags["wants_claim_tracking"] = True``
based on its semantic reading of the user's query. The gate is
enforced at three layers:

1. **Catalog** — :data:`src.core.agents.registry.CLAIM_AGENT.policy_gate`
   is shown verbatim in the planner's system prompt so the LLM
   learns the intuition.
2. **Plan validation** — :meth:`AgentRegistry.validate_plan` rejects
   any plan containing a ``claim_agent`` step when the flag is False.
3. **Construction** — :class:`ScopedAgent.__init__` re-runs
   :meth:`AgentRegistry.validate_step` and raises
   :class:`ScopedAgentError` if the gate is unsatisfied. The factory
   below does no extra gate work — the registry enforcement runs
   uniformly on every ScopedAgent construction.

Architecture
------------
* **Registered** in :data:`src.core.agents.registry.CLAIM_AGENT`.
* **Constructed** via :func:`build_claim_agent` below.
* **Runs** the standard :class:`~src.core.agents._base.ScopedAgent`
  ReAct loop with the planner-chosen ``tool_subset``.

Model parameters
----------------
* **Temperature 0.1** — claim verdicts are **structured outputs**
  (claim text + verdict label + evidence cite), not free-form prose.
* **max_tokens 2000** — modest budget; verdict tables fit
  comfortably under this.

Tests: ``tests/test_factories.py::BuildClaimAgentTests``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.personas.base import build_chat_model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


def build_claim_agent(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None,
    registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ScopedAgent:
    """ScopedAgent for claim extraction + comparison.

    This agent is **policy-gated**: ``ScopedAgent.__init__`` will raise
    :class:`ScopedAgentError` unless ``intent_flags["wants_claim_tracking"]``
    is True. The factory itself does no extra gate work - the registry
    enforcement runs uniformly on every ScopedAgent construction.

    Tight temperature (0.1) because claim verdicts are
    structured outputs (claim text + verdict label + evidence cite),
    not free-form prose.
    """
    model = build_chat_model(
        temperature=0.1,
        max_tokens=2000,
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


__all__ = ["build_claim_agent"]
