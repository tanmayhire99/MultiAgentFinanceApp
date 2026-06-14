"""Synthesizer Agent — final user-visible markdown report.

The synthesizer is the **last step** of every non-trivial plan. It
takes the outputs of upstream steps from the shared scratchpad and
produces ONE coherent markdown report for the user. It owns no MCP
tools — pure LLM synthesis.

Architecturally distinct
------------------------
The synthesizer is the only agent (other than ``panel_agent``) that
diverges from the default :class:`~src.core.agents._base.ScopedAgent`
behaviour:

* **No MCP tools.** ``registry.SYNTHESIZER.tools = ()``.
  ``PlanStep.tool_subset`` for a synthesizer step must be empty;
  ``PlanStep.max_tool_calls`` must be 0.
* **Custom system prompt.** The default ScopedAgent prompt frames
  the LLM as "you are running ONE step of a larger plan; your output
  feeds downstream steps". That framing is **wrong** for a
  synthesizer step which IS the user-visible response. The factory
  passes a ``system_prompt_override`` (built from
  :data:`_SYNTHESIZER_PROMPT_TEMPLATE`) so the LLM understands its
  audience is the user, not another agent.
* **Largest token budget of any agent (4000)** — the synthesis IS
  the user's response, so it gets enough room for headers, tables,
  citations, and a closing "Bottom line".

Architecture
------------
* **Registered** in :data:`src.core.agents.registry.SYNTHESIZER`.
* **Constructed** via :func:`build_synthesizer` below.
* **Runs** the standard ReAct loop in
  :class:`~src.core.agents._base.ScopedAgent`, but with the
  override prompt and an empty MCP tool list. The synthesizer can
  still call the synthetic tools (``get_prior_result``,
  ``request_assistance``) the base class injects.

Tests: ``tests/test_factories.py::BuildSynthesizerTests``.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.personas.base import build_chat_model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


# The synthesizer's system prompt. ``{step_description}`` and
# ``{deps_repr}`` are filled in by :func:`build_synthesizer` per step.
# Hard rules at the bottom protect the demo's compliance posture: the
# dispatcher adds the regulatory disclaimer separately, so the
# synthesizer must not duplicate it.
_SYNTHESIZER_PROMPT_TEMPLATE = """You are the **FinAI Synthesizer**, \
the FINAL agent in this multi-agent investigation.

Earlier agents have completed their steps and written their results \
to the shared scratchpad. Your job is to read those results (via \
``get_prior_result(step_id)``, scoped to your declared dependencies) \
and produce ONE coherent markdown report for the user.

### Your task
{step_description}

### Your declared dependencies
{deps_repr}

You can ONLY read these step IDs via ``get_prior_result``. Other \
prior steps are intentionally hidden so your final report stays \
focused on the relevant data.

### Style
- Open by directly answering the user's question (no preamble like \
"Based on the data, ...").
- Use markdown headers, short tables, bullet lists where they help the \
reader scan. Avoid walls of prose.
- CITE specific numbers, dates, claim quotes, and source URLs from the \
prior step results. Don't paraphrase to vagueness.
- Keep each section under 250 words.
- End with a single **Bottom line** sentence (or two) that gives the \
user a clear take-away.

### Hard rules
- DO NOT fabricate numbers. Every figure or quote must come verbatim \
from a prior step result. If a number isn't in the scratchpad, do not \
invent one - say what's missing instead.
- DO NOT recommend buying or selling specific securities.
- DO NOT include a regulatory disclaimer; the dispatcher adds one \
automatically for finance flows.
- DO NOT call ``request_assistance`` - this is the FINAL step. If a \
prior step failed or returned partial data, work with what you have \
and call out the gap explicitly.
"""


def build_synthesizer(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None,
    registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ScopedAgent:
    """ScopedAgent specialised for the final user-visible synthesis.

    Largest token budget of any agent (4000) because the synthesis IS
    the user's response. Streaming so the report appears progressively
    rather than as a wall after a long wait.

    Uses ``system_prompt_override`` to swap out ScopedAgent's default
    "you are running ONE step of a larger plan" framing for one that
    treats the LLM as the writer of the final response.
    """
    deps = sorted(step.depends_on)
    deps_repr = "[]" if not deps else str(deps)
    system_prompt = _SYNTHESIZER_PROMPT_TEMPLATE.format(
        step_description=step.description.strip(),
        deps_repr=deps_repr,
    )
    model = build_chat_model(
        temperature=0.3,
        max_tokens=4000,
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
        system_prompt_override=system_prompt,
    )


__all__ = ["build_synthesizer", "_SYNTHESIZER_PROMPT_TEMPLATE"]
