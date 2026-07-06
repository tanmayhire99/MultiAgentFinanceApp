"""Synthesizer — the FINAL, user-visible report writer.

Registered as :data:`src.core.agents.registry.SYNTHESIZER`; reached at
runtime via :func:`src.core.agents.factory_dispatch.build_scoped_agent_for_step`.

The synthesizer is special: it produces the user-visible final report,
not an intermediate result. Its system prompt MUST override the default
"you are running ONE step of a larger plan" framing because that framing
tells the LLM to write for downstream steps - exactly the wrong audience
here.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from langchain_core.tools import BaseTool

from src.core.agents import _model
from src.core.agents._base import DEFAULT_RECURSION_LIMIT, ScopedAgent
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad


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

### Markdown rigor (the renderer is strict GFM)
Your report is rendered by a strict GitHub-Flavored Markdown renderer. \
Malformed syntax shows up as literal `#` or `|` characters in the user's \
chat, so follow these rules exactly:
- **Blank line before every table.** A `|` row must start on its own \
line, preceded by a blank line. Never run a label or sentence directly \
into a table — `**Label:**| a | b |` will render the pipes literally.
- **Space after `#` in headings.** Write `### Section`, never `###Section`. \
CommonMark requires the space; without it the hashes render literally.
- **Blank line before and after every heading** so the renderer can detect \
the block boundary.
- **Tables need a separator row.** The line immediately after the header \
row must be `|---|---|` (one dash cell per column). GFM will not infer it.
- **No `_italic_` for lines containing `|`, `{{`, `}}`, `[`, or `:`** — \
those characters break the underscore delimiter. Use `*italic*` or \
backticks instead.

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
    step: PlanStep, scratchpad: Scratchpad, all_mcp_tools: Sequence[BaseTool],
    intent_flags: Optional[Dict[str, bool]] = None, registry: AgentRegistry = REGISTRY,
    api_key_slot: str = "primary", recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    user_id: str = "demo",
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
    model = _model.build_chat_model(
        temperature=0.3, max_tokens=4000, streaming=True,
        api_key_slot=api_key_slot, cycle_keys=True,
    )
    return ScopedAgent(
        step=step, scratchpad=scratchpad, all_mcp_tools=all_mcp_tools,
        model=model, registry=registry, intent_flags=intent_flags,
        recursion_limit=recursion_limit, system_prompt_override=system_prompt,
        user_id=user_id,
    )


__all__ = ["build_synthesizer", "_SYNTHESIZER_PROMPT_TEMPLATE"]
