"""Per-agent ``ScopedAgent`` factory functions.

The :class:`ScopedAgent` class in :mod:`src.core.agents._base` is
agent-agnostic: it takes a step, a model, and a tool pool, and wires
them into a ReAct loop. Each *kind* of agent (research, filings, claim,
synthesizer, …) actually wants slightly different model parameters
(temperature, max_tokens) and, in the synthesizer's case, an entirely
different system prompt.

This module centralises that per-agent specialisation so the executor
doesn't have to know:

::

    from src.core.agents._factories import build_scoped_agent_for_step

    agent = build_scoped_agent_for_step(
        step=plan.steps[i],
        scratchpad=run_scratchpad,
        all_mcp_tools=mcp_pool,
        intent_flags=intent_flags,
    )
    result = await agent.run()

The dispatch from ``step.agent`` (a string like ``"research_agent"``)
to the right factory function happens here.

What a factory does
-------------------
1. Picks a :class:`ChatOpenAI` model with agent-appropriate
   ``temperature`` / ``max_tokens`` (research agents allow some
   creativity for summarising; filings + claim agents are tighter to
   stay analytical; the synthesizer gets the largest token budget
   because it produces the user-visible report).
2. Optionally supplies a :class:`ScopedAgent` ``system_prompt_override``.
   Today only the synthesizer uses this - other factories let
   ``ScopedAgent._build_system_prompt`` produce its standard
   "you are running ONE step of a larger plan" framing.
3. Constructs and returns a ready-to-run :class:`ScopedAgent`.

The factories deliberately stay TINY (handful of lines each). New
agent types should be added by:
1. Registering the agent in :mod:`src.core.agents.registry`.
2. Adding a ``build_<agent>_agent`` factory here.
3. Adding the new factory to :data:`_FACTORY_MAP`.
4. Adding a focused unit test in ``tests/test_factories.py``.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.agents.personas.base import build_chat_model
from src.core.agents._base import (
    DEFAULT_RECURSION_LIMIT,
    ScopedAgent,
    ScopedAgentError,
)
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


# ---------------------------------------------------------------------------
# Per-agent factory functions
#
# Each factory accepts the same arguments (so :func:`build_scoped_agent_for_step`
# can dispatch uniformly) and differs only in (a) model params and (b) prompt.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Synthesizer
#
# The synthesizer is special: it produces the user-visible final report,
# not an intermediate result. Its system prompt MUST override the default
# "you are running ONE step of a larger plan" framing because that
# framing tells the LLM to write for downstream steps - exactly the wrong
# audience here.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
# Maps an agent's snake_case ``name`` (matching the registry) to the
# factory function. The executor calls :func:`build_scoped_agent_for_step`
# without caring which agent it's about to run.
_FactoryFn = Callable[..., ScopedAgent]
_FACTORY_MAP: Dict[str, _FactoryFn] = {
    "research_agent": build_research_agent,
    "filings_agent": build_filings_agent,
    "claim_agent": build_claim_agent,
    "synthesizer": build_synthesizer,
    # NOTE: us_stock_agent / indian_stock_agent / portfolio_agent /
    # panel_agent factories are added in Stage 4 (panel slice). Stage 1
    # only covers the four agents the claim-tracker slice needs.
}


def build_scoped_agent_for_step(
    *,
    step: PlanStep,
    scratchpad: Scratchpad,
    all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None,
    registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary",
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> ScopedAgent:
    """Dispatch on ``step.agent`` and call the right factory.

    Raises :class:`ScopedAgentError` if no factory is registered for
    ``step.agent``. This will happen if a planner produces a step
    naming an agent we haven't yet specialised - the registry's
    canonical 8 agents minus the 4 covered here = 4 missing factories
    until Stage 4 of the migration slice lands.
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
    )


__all__ = [
    "build_research_agent",
    "build_filings_agent",
    "build_claim_agent",
    "build_synthesizer",
    "build_scoped_agent_for_step",
]
