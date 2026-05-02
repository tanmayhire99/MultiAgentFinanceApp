"""Planner — turns a user query + intent flags into a validated :class:`Plan`.

This is **Phase 2** of the planner-first pipeline (the classifier in
:mod:`src.core.router` is Phase 1). The planner is a single LLM call
with strict JSON output, validated against the
:class:`~src.core.types.Plan` schema and the registry's policy gates.

Why a single LLM call (not an agent loop)
-----------------------------------------
Plan-and-Execute / LLMCompiler papers — the planner sees the whole
query, picks a DAG of agent invocations up front, and lets the
executor run them. Iterative re-planning happens in the joiner phase
(future work, currently just runs the plan once).

What the planner LLM is given
-----------------------------
* The user's query (post-trace / post-artifact-prefix stripping).
* The classifier's ``intent_flags`` dict — boolean flags like
  ``wants_claim_tracking`` that policy gates check at validation time.
* :func:`AgentRegistry.planner_catalog_text` — markdown listing every
  agent's name, description, role hint, owned tools, and policy gate
  (if any).
* The :class:`Plan` JSON schema (auto-generated via Pydantic v2's
  ``model_json_schema``) so the LLM knows exactly what fields to emit.
* A handful of worked examples that show good DAG shapes for typical
  intents (so the LLM doesn't have to invent the conventions).

What the planner LLM must produce
---------------------------------
A single JSON object that parses as :class:`Plan`. Validation runs
on the parsed object via :meth:`AgentRegistry.validate_plan`,
checking:

* Every step's ``agent`` is registered.
* Every tool in each ``tool_subset`` is owned by that agent.
* Every gated agent (claim_agent, panel_agent) has its required
  intent flag set in ``intent_flags`` — otherwise the plan is
  rejected.

The :class:`Plan` model itself enforces the rest: unique step IDs,
dependencies must exist, no cycles, no self-loops, deterministic
topological order.

Failure handling
----------------
* **JSON parse failure** — retry once with a clearer prompt
  appended ("your previous output was not valid JSON, here is the
  schema again, try again"). After two failures, raise
  :class:`PlannerError`.
* **Schema / registry validation failure** — retry once with the
  validation errors echoed back to the LLM. After two failures,
  raise :class:`PlannerError`.
* **LLM connection failure** — retry once with a 500ms backoff
  (matches :func:`src.core.router.classify_query`'s retry policy).
* All failures end up in a structured :class:`PlannerError` so the
  pipeline can decide whether to fall back to a deterministic flow
  or surface the error to the user.

This module is import-safe with no MCP dependency. The executor
uses the produced Plan to drive ScopedAgents (which DO need MCP
tools); the planner itself just decides shape.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from src.agents.personas.base import build_chat_model
from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import KNOWN_INTENT_FLAGS, Plan


log = logging.getLogger("finai.planner")


# Two retries total: the first call PLUS one retry on transient
# failure. JSON-parse / validation failures get one repair-retry.
DEFAULT_RETRIES = 1
DEFAULT_TIMEOUT_SECONDS = 45.0


class PlannerError(Exception):
    """Unrecoverable planner failure.

    Carries the LLM output (if any) and the underlying validation /
    parse errors so the caller can produce a useful error message
    or log entry.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_output: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.validation_errors = validation_errors or []


# ---------------------------------------------------------------------------
# System prompt
#
# The system prompt is large but structured: role, hard rules, agent
# catalog, schema, examples, output contract. Each section is a clear
# block so the LLM can re-read what it needs.
# ---------------------------------------------------------------------------
_PLANNER_SYSTEM_TEMPLATE = """You are the **FinAI Planner**. \
Your single job is to produce a JSON ``Plan`` object that solves \
the user's query by orchestrating the available agents.

## Hard rules (the executor will reject your output if any are violated)

1. **Output strict JSON only.** No markdown fences, no prose, no \
trailing commentary. Just the JSON object.
2. **Every step's ``agent`` MUST appear in the agent catalog below.**
3. **Every tool in a step's ``tool_subset`` MUST be owned by that \
step's agent.** Cross-agent tool calls are not allowed; if step N \
needs another agent's data, add a separate step for that agent and \
make step N depend on it via ``depends_on``.
4. **Policy-gated agents may only be included when the corresponding \
intent flag is True.** The intent flags for THIS query are listed \
below; if a flag you'd need is False, do not include the gated agent.
5. **The LAST step MUST be ``synthesizer``** with empty \
``tool_subset`` and ``depends_on`` referencing every step whose \
output the final report depends on. The synthesizer writes the \
user-visible report.
6. **Steps MUST form a DAG.** Use ``depends_on: []`` for steps with \
no prerequisites. No self-loops. No cycles. Step IDs are positive \
integers, unique within the plan.
7. **Keep plans tight.** Aim for 2-5 steps. More than 8 is rejected.

## Agent catalog

{agent_catalog}

## Intent flags for THIS query

{intent_flags_block}

## Plan JSON schema (pydantic v2)

The plan you produce MUST conform to this schema:

```json
{plan_schema}
```

## Worked examples

### Example A — simple stock research, no panel, no claim tracking
User query: "Tell me about NVDA"
Intent flags: all False (no panel, no claims, no filings, no portfolio).

```json
{example_simple_stock}
```

### Example B — claim tracking
User query: "Did Tesla deliver on FSD?"
Intent flags: wants_claim_tracking=True, wants_filings=True, \
wants_historical_news=True, wants_deep_research=True.

```json
{example_claim_tracking}
```

### Example C — panel debate on a portfolio
User query: "Run a panel debate on my portfolio"
Intent flags: wants_portfolio_data=True, wants_panel_debate=True.

```json
{example_panel_portfolio}
```

### Example D — panel debate on a single stock
User query: "Run a panel debate on NVDA"
Intent flags: wants_panel_debate=True.

```json
{example_panel_stock}
```

## Output contract

Return EXACTLY one JSON object. No code fence, no commentary.
"""


# ---------------------------------------------------------------------------
# Worked-example plans
#
# Hard-coded here so the planner LLM has a concrete shape to follow.
# These are also useful as test fixtures for the executor.
# ---------------------------------------------------------------------------
_EXAMPLE_SIMPLE_STOCK = {
    "schema_version": "1.0",
    "goal": "Brief NVDA: live fundamentals + recent catalysts + a synthesised analyst note.",
    "rationale": "stock research with no panel and no claim tracking - one stock_agent step, one research_agent step, one synthesizer step.",
    "estimated_complexity": "moderate",
    "steps": [
        {
            "id": 1,
            "description": "Fetch live fundamentals, growth metrics, and defensive metrics for NVDA via the US Stock Agent.",
            "agent": "us_stock_agent",
            "tool_subset": [
                "us_stock__get_fundamentals",
                "us_stock__get_growth_metrics",
                "us_stock__get_defensive_metrics",
            ],
            "depends_on": [],
            "max_tool_calls": 5,
        },
        {
            "id": 2,
            "description": "Pull recent news and a one-paragraph company brief for NVDA via the Research Agent.",
            "agent": "research_agent",
            "tool_subset": [
                "research__search_news",
                "research__get_company_brief",
            ],
            "depends_on": [],
            "max_tool_calls": 4,
        },
        {
            "id": 3,
            "description": "Write the structured analyst note for the user, citing the fundamentals from step 1 and catalysts from step 2.",
            "agent": "synthesizer",
            "tool_subset": [],
            "depends_on": [1, 2],
            "max_tool_calls": 0,
        },
    ],
}


_EXAMPLE_PANEL_PORTFOLIO = {
    "schema_version": "1.0",
    "goal": "Run a Buffett / Wood / Graham multi-round debate over the user's portfolio holdings, then synthesise a moderator closing brief.",
    "rationale": "Portfolio-level panel: portfolio_agent surfaces holdings + risks, panel_agent runs the multi-persona debate loop, synthesizer writes the final brief.",
    "estimated_complexity": "heavy",
    "steps": [
        {
            "id": 1,
            "description": "Pull the user's holdings, sector allocation, concentration risks, and diversification score via the Portfolio Agent.",
            "agent": "portfolio_agent",
            "tool_subset": [
                "portfolio__get_holdings",
                "portfolio__get_portfolio_summary",
                "portfolio__get_sector_allocation",
                "portfolio__get_concentration_risks",
                "portfolio__get_diversification_score",
            ],
            "depends_on": [],
            "max_tool_calls": 5,
        },
        {
            "id": 2,
            "description": "Run the full multi-round Buffett / Wood / Graham investor-panel debate on the portfolio surfaced in step 1.",
            "agent": "panel_agent",
            "tool_subset": [],
            "depends_on": [1],
            "max_tool_calls": 0,
        },
        {
            "id": 3,
            "description": "Write the moderator closing brief synthesising the panel debate transcript and verdicts from step 2.",
            "agent": "synthesizer",
            "tool_subset": [],
            "depends_on": [2],
            "max_tool_calls": 0,
        },
    ],
}

_EXAMPLE_PANEL_STOCK = {
    "schema_version": "1.0",
    "goal": "Run a Buffett / Wood / Graham multi-round debate over NVDA fundamentals + recent catalysts, then synthesise a moderator closing brief.",
    "rationale": "Single-ticker panel: us_stock_agent pulls live fundamental data, research_agent gets catalysts, panel_agent runs the multi-persona debate loop, synthesizer writes the final brief.",
    "estimated_complexity": "heavy",
    "steps": [
        {
            "id": 1,
            "description": "Fetch NVDA live fundamentals, growth metrics, and defensive metrics via the US Stock Agent.",
            "agent": "us_stock_agent",
            "tool_subset": [
                "us_stock__get_fundamentals",
                "us_stock__get_growth_metrics",
                "us_stock__get_defensive_metrics",
            ],
            "depends_on": [],
            "max_tool_calls": 5,
        },
        {
            "id": 2,
            "description": "Pull recent catalysts and a company brief for NVDA via the Research Agent.",
            "agent": "research_agent",
            "tool_subset": [
                "research__get_key_catalysts",
                "research__get_company_brief",
            ],
            "depends_on": [],
            "max_tool_calls": 4,
        },
        {
            "id": 3,
            "description": "Run the full multi-round Buffett / Wood / Graham debate on NVDA using fundamentals from step 1 and catalysts from step 2.",
            "agent": "panel_agent",
            "tool_subset": [],
            "depends_on": [1, 2],
            "max_tool_calls": 0,
        },
        {
            "id": 4,
            "description": "Write the moderator closing brief synthesising the NVDA panel debate transcript and verdicts from step 3.",
            "agent": "synthesizer",
            "tool_subset": [],
            "depends_on": [3],
            "max_tool_calls": 0,
        },
    ],
}

_EXAMPLE_CLAIM_TRACKING = {
    "schema_version": "1.0",
    "goal": "Verify Tesla's FSD timeline claims against reality.",
    "rationale": "Claim tracking on Tesla - need SEC filings (forward claims) + historical news (reality check) + claim_agent for extraction and comparison.",
    "estimated_complexity": "heavy",
    "steps": [
        {
            "id": 1,
            "description": "List + fetch Tesla's recent 10-K, 10-Q, and 8-K filings from SEC EDGAR.",
            "agent": "filings_agent",
            "tool_subset": [
                "research__get_sec_filings",
                "research__fetch_sec_document",
            ],
            "depends_on": [],
            "max_tool_calls": 6,
        },
        {
            "id": 2,
            "description": "Pull historical news on Tesla FSD timeline and rollouts (2020-present).",
            "agent": "research_agent",
            "tool_subset": ["research__search_historical_news"],
            "depends_on": [],
            "max_tool_calls": 4,
        },
        {
            "id": 3,
            "description": "Extract forward-looking FSD claims from the filings retrieved in step 1.",
            "agent": "claim_agent",
            "tool_subset": ["research__extract_forward_claims"],
            "depends_on": [1],
            "max_tool_calls": 3,
        },
        {
            "id": 4,
            "description": "Compare each extracted claim against the historical news from step 2.",
            "agent": "claim_agent",
            "tool_subset": ["research__compare_claim_to_reality"],
            "depends_on": [3, 2],
            "max_tool_calls": 5,
        },
        {
            "id": 5,
            "description": "Write the verdict report grouping promised vs delivered, citing specific claims and news sources.",
            "agent": "synthesizer",
            "tool_subset": [],
            "depends_on": [3, 4],
            "max_tool_calls": 0,
        },
    ],
}


# ---------------------------------------------------------------------------
# Prompt assembly helpers
# ---------------------------------------------------------------------------
def _intent_flags_block(intent_flags: Dict[str, bool]) -> str:
    """Render the intent_flags dict as a checklist block for the prompt."""
    lines = []
    for flag in sorted(KNOWN_INTENT_FLAGS):
        v = bool(intent_flags.get(flag, False))
        marker = "✓" if v else "✗"
        lines.append(f"- {marker} ``{flag}``: {v}")
    return "\n".join(lines)


def _build_system_prompt(
    intent_flags: Dict[str, bool],
    registry: AgentRegistry,
) -> str:
    """Build the planner's system prompt for a single query."""
    return _PLANNER_SYSTEM_TEMPLATE.format(
        agent_catalog=registry.planner_catalog_text(),
        intent_flags_block=_intent_flags_block(intent_flags),
        plan_schema=json.dumps(Plan.model_json_schema(), indent=2),
        example_simple_stock=json.dumps(_EXAMPLE_SIMPLE_STOCK, indent=2),
        example_claim_tracking=json.dumps(_EXAMPLE_CLAIM_TRACKING, indent=2),
        example_panel_portfolio=json.dumps(_EXAMPLE_PANEL_PORTFOLIO, indent=2),
        example_panel_stock=json.dumps(_EXAMPLE_PANEL_STOCK, indent=2),
    )


def _build_user_message(query: str, history_summary: Optional[str] = None) -> str:
    parts = [f"User query: {query.strip()}"]
    if history_summary:
        parts.append("")
        parts.append(f"Prior turn context: {history_summary.strip()}")
    parts.append("")
    parts.append("Produce the Plan JSON now.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON parsing + repair
# ---------------------------------------------------------------------------
def _try_repair_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort: carve the first balanced JSON object out of raw output.

    Some models occasionally prepend "Here is the plan:" or wrap the
    output in ```json fences despite the system prompt. Strip those
    and try again before giving up.
    """
    if not raw:
        return None
    s = raw.strip()
    # Strip markdown code fence
    if s.startswith("```"):
        s = s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    # Carve the first balanced { ... }
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def plan(
    query: str,
    *,
    intent_flags: Dict[str, bool],
    registry: AgentRegistry = REGISTRY,
    history_summary: Optional[str] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> Plan:
    """Produce and validate a :class:`Plan` for ``query``.

    Returns a :class:`Plan` object that:
      * Parsed from valid JSON
      * Conforms to the Pydantic schema (cycles, missing deps, etc.
        already rejected at parse time)
      * Passes :meth:`AgentRegistry.validate_plan` against
        ``intent_flags`` (no policy gate violations, no orphan
        agents, no orphan tools)

    Raises :class:`PlannerError` on any unrecoverable failure.
    """
    system_prompt = _build_system_prompt(intent_flags, registry)
    user_message = _build_user_message(query, history_summary)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    llm = build_chat_model(
        temperature=0.1,
        max_tokens=2500,
        streaming=False,
        response_format={"type": "json_object"},
    )

    last_raw: Optional[str] = None
    last_validation_errors: List[str] = []

    for attempt in range(retries + 1):
        # 1) LLM call (with one transient-retry inside this attempt)
        try:
            resp = await asyncio.wait_for(
                llm.ainvoke(messages), timeout=timeout_seconds
            )
        except Exception as e:
            log.warning("Planner LLM call failed (attempt %d): %s", attempt + 1, e)
            if attempt < retries:
                await asyncio.sleep(0.5)
                continue
            raise PlannerError(
                f"Planner LLM call failed after {retries + 1} attempts: {e}"
            ) from e

        last_raw = getattr(resp, "content", "") or ""

        # 2) Parse JSON (with a repair attempt for fenced / wrapped output)
        try:
            data = json.loads(last_raw)
        except (json.JSONDecodeError, TypeError):
            data = _try_repair_json(last_raw)
            if data is None:
                log.warning(
                    "Planner JSON parse failed (attempt %d). Raw output (first 300): %r",
                    attempt + 1, last_raw[:300],
                )
                if attempt < retries:
                    # Echo the parse failure back so the LLM can self-correct
                    messages.append(
                        HumanMessage(
                            content=(
                                "Your previous output was not valid JSON. "
                                "Re-emit the Plan as a single JSON object. "
                                "No code fence, no prose, no trailing text."
                            )
                        )
                    )
                    continue
                raise PlannerError(
                    "Planner did not return valid JSON.",
                    raw_output=last_raw,
                )

        # 3) Schema-validate via Pydantic
        try:
            plan_obj = Plan.model_validate(data)
        except ValidationError as ve:
            errors = [str(e) for e in ve.errors()]
            log.warning(
                "Planner schema validation failed (attempt %d): %s",
                attempt + 1, errors[:3],
            )
            last_validation_errors = errors
            if attempt < retries:
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous Plan failed Pydantic validation:\n"
                            + "\n".join(f"  - {e}" for e in errors[:5])
                            + "\n\nFix these and emit the Plan again."
                        )
                    )
                )
                continue
            raise PlannerError(
                "Planner output did not match the Plan schema.",
                raw_output=last_raw,
                validation_errors=errors,
            )

        # 4) Registry / policy-gate validation
        registry_errors = registry.validate_plan(plan_obj, intent_flags)
        if registry_errors:
            log.warning(
                "Planner registry validation failed (attempt %d): %s",
                attempt + 1, registry_errors[:3],
            )
            last_validation_errors = registry_errors
            if attempt < retries:
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous Plan failed registry validation:\n"
                            + "\n".join(f"  - {e}" for e in registry_errors)
                            + "\n\nFix these (e.g. only use tools owned by "
                            "each agent; don't include policy-gated agents "
                            "without their flag set in intent_flags) and "
                            "emit the Plan again."
                        )
                    )
                )
                continue
            raise PlannerError(
                "Planner produced a plan that violates the agent registry.",
                raw_output=last_raw,
                validation_errors=registry_errors,
            )

        log.info(
            "Planner produced a valid plan: %d steps, agents=%s",
            len(plan_obj.steps), sorted(plan_obj.all_agents()),
        )
        return plan_obj

    # Unreachable - all paths above either return or raise.
    raise PlannerError(
        "Planner exhausted all retries without producing a valid Plan.",
        raw_output=last_raw,
        validation_errors=last_validation_errors,
    )


__all__ = [
    "PlannerError",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "plan",
]
