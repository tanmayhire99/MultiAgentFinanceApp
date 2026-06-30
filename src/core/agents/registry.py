"""Agent registry — the canonical catalog the planner sees.

This is the **agent layer** of the planner-first architecture (Day 2 of
the migration in ``docs/MULTI_AGENT_ARCHITECTURE.md``). Every agent
listed here is a **logical role** the planner can assign work to. Each
one declares:

* a unique ``name`` (the value that goes into ``PlanStep.agent``)
* a human-readable ``description`` (used in the planner LLM's prompt)
* an exhaustive ``tools`` list (the value of ``PlanStep.tool_subset`` for
  this agent must be a subset of this list)
* an optional ``policy_gate`` — an **intent-flag** check that determines
  whether the planner is allowed to include this agent in a plan,
  given what the classifier LLM said the user wants

The registry is intentionally **pure data**: no MCP imports, no LangChain
imports, no networking. The executor will resolve tool names to actual
LangChain ``BaseTool`` objects from the MCP adapter at execution time;
the registry just declares what *should* be available.

The single source of truth for tool names is the namespaced output of
``src.config.mcp_servers.get_tools()``. If the MCP namespacing scheme
changes, update :data:`REGISTRY` and bump the corresponding
``known_counts`` entry in :mod:`src.config.mcp_servers`.

Why a registry (instead of importing the agents directly)?
---------------------------------------------------------
The planner LLM picks agents by name. Decoupling the *catalog* (data)
from the *agent implementations* (code, ports, LLM calls) lets us:

* serialise the catalog into the planner's system prompt cheaply
* validate plans before any expensive code is invoked
* unit-test agent selection without spinning up MCP subprocesses
* swap individual agent implementations (LoRA-fine-tuned vs base) at
  runtime without touching the planner

Policy gates — LLM-driven, not rule-based
-----------------------------------------
Two agents currently have policy gates: ``claim_agent`` (extract /
compare forward-looking claims) and ``panel_agent`` (Buffett / Wood /
Graham debate). Without a gate, every "research this stock" query
ends up running claim analysis or the panel because the planner LLM
finds them useful.

The gate is **strictly structural**: it checks whether one of the
**boolean intent flags** that the classifier LLM produced in Phase 1 is
True. Concretely, ``claim_agent`` requires
``intent_flags["wants_claim_tracking"]`` to be True; ``panel_agent``
requires ``intent_flags["wants_panel_debate"]``.

The classifier — a real LLM, not a regex — is the sole place where
natural language understanding happens. If the classifier sets the
flag based on the user saying "did Tesla follow through on FSD" or
"has Microsoft fulfilled its AI promises" or any other paraphrase,
the gate accepts it. If the classifier reads "deep research on TCS"
and decides those flags should be False, the gate rejects any plan
that tried to include the gated agents anyway.

This means policy gates do **defense in depth**, not primary
decision-making: if the planner LLM hallucinates and includes
``claim_agent`` for a research-only query, the gate catches it
because the classifier's flag is False. The decision is still LLM-
driven; the gate just enforces consistency between the classifier's
output (Phase 1) and the planner's output (Phase 2).
"""
from __future__ import annotations

import os
import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.types import KNOWN_INTENT_FLAGS, Plan, PlanStep

# ``re`` is still imported because the snake_case agent-name validator below
# uses ``re.fullmatch``. Nothing else in this module pattern-matches text -
# semantic understanding is the classifier LLM's job.


# ---------------------------------------------------------------------------
# PolicyGate
# ---------------------------------------------------------------------------
class PolicyGate(BaseModel):
    """An intent-flag precondition for including an agent in a plan.

    The gate is purely **structural**: it checks whether the boolean
    intent flags that the classifier LLM produced in Phase 1 include
    any of the flags this gate requires. There is **no text matching
    or regex** anywhere in the gate — natural language understanding
    is delegated entirely to the classifier.

    Example: ``claim_agent`` declares
    ``required_intent_flags=["wants_claim_tracking"]``. When the user
    asks "did Tesla deliver on FSD?", the classifier LLM is responsible
    for setting ``intent_flags["wants_claim_tracking"]=True``, and the
    gate then permits the plan to include ``claim_agent``. When the
    user asks "deep research on TCS", the classifier sets the flag to
    False, the gate rejects any plan that tried to include
    ``claim_agent``.

    When :attr:`hard_block_unless_match` is True (the default), the
    :class:`AgentRegistry` rejects any :class:`PlanStep` using this
    agent unless the flag is set. When False, the gate is advisory:
    surfaced in the planner prompt but not enforced.

    Why this design
    ---------------
    Earlier iterations used regex / phrase whitelists. That was a
    brittle pretend-classifier living next to a real one (the Phase 1
    LLM). It missed natural paraphrases ("fulfilled its promises",
    "lived up to", "kept its word") and bloated with new patterns
    every time we found a miss. By delegating to the classifier flag,
    paraphrases become free — the LLM does what it's good at — and
    the registry stays a clean structural check.
    """

    model_config = ConfigDict(extra="forbid")

    required_intent_flags: List[str] = Field(
        default_factory=list,
        description=(
            "Names of intent flags from "
            ":data:`src.core.types.KNOWN_INTENT_FLAGS`. The gate is "
            "satisfied when ANY of these flags is True in the "
            "classifier's output. Empty list = no constraint (an "
            "empty gate is effectively a no-op)."
        ),
    )
    hard_block_unless_match: bool = Field(
        default=True,
        description=(
            "When True, the registry rejects any PlanStep using this "
            "agent if the gate is unsatisfied. When False, the gate "
            "is advisory only - shown to the planner LLM but not "
            "enforced at validation time."
        ),
    )
    description: str = Field(
        default="",
        description=(
            "One-sentence explanation surfaced to the planner LLM and "
            "in error messages. Tell the planner WHY this agent is "
            "gated so it learns the right intuition."
        ),
    )

    @field_validator("required_intent_flags")
    @classmethod
    def _flags_must_be_known(cls, v: List[str]) -> List[str]:
        # Catch typos at import time. A flag the classifier never sets
        # would silently keep the gate closed forever, which is a safe
        # but confusing failure mode.
        unknown = [f for f in v if f not in KNOWN_INTENT_FLAGS]
        if unknown:
            raise ValueError(
                f"Unknown intent flag(s): {unknown}. "
                f"Valid flags: {sorted(KNOWN_INTENT_FLAGS)}. "
                f"Add new flags to "
                f"src.core.types.KNOWN_INTENT_FLAGS first."
            )
        return v

    def matches(self, intent_flags: Dict[str, bool]) -> bool:
        """True iff any required flag is True in ``intent_flags``.

        ``intent_flags`` is the dict produced by the Phase 1
        classifier (see ``src/core/router.py``). Missing flags are
        treated as False - the classifier is responsible for setting
        every relevant flag explicitly.
        """
        if not self.required_intent_flags:
            return True  # gate with no required flags is vacuously open
        return any(
            bool(intent_flags.get(flag, False))
            for flag in self.required_intent_flags
        )

    def explain(self) -> str:
        """Human-readable version of the gate (for planner prompt + UI)."""
        block = "HARD-BLOCK" if self.hard_block_unless_match else "advisory"
        flags = ", ".join(self.required_intent_flags) or "(none)"
        prefix = self.description.strip()
        body = f"requires intent flag: {flags} [{block}]"
        return f"{prefix} {body}" if prefix else body


# ---------------------------------------------------------------------------
# AgentDefinition
# ---------------------------------------------------------------------------
class AgentDefinition(BaseModel):
    """One row of the agent catalog. Pure data, no behaviour."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=2, max_length=64)
    title: str = Field(..., description="Display label for the UI / logs.")
    description: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description=(
            "What this agent is for, in one paragraph the planner LLM "
            "can reason over. Tell the planner the AGENT'S JOB, not the "
            "list of tools - the tools are listed separately."
        ),
    )
    tools: Tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Exhaustive tuple of namespaced MCP tool names this agent "
            "owns. ``PlanStep.tool_subset`` must be a subset of this. "
            "Frozen tuple so the registry is hashable / cacheable."
        ),
    )
    policy_gate: Optional[PolicyGate] = Field(
        default=None,
        description=(
            "Optional gate that must be satisfied by the user query "
            "for this agent to be usable in a plan. None = freely "
            "usable."
        ),
    )
    max_tool_calls_default: int = Field(
        default=20,
        ge=1,
        le=200,
        description=(
            "Default ``PlanStep.max_tool_calls`` budget when the "
            "planner doesn't specify one for this agent."
        ),
    )
    role_hint: str = Field(
        default="",
        max_length=240,
        description=(
            "One-sentence usage hint for the planner ('Use for: ...'). "
            "Surfaced verbatim in the planner prompt's catalog block."
        ),
    )

    @field_validator("name")
    @classmethod
    def _name_lowercase_snake(cls, v: str) -> str:
        # Keep names predictable for the planner LLM; only [a-z0-9_].
        if not re.fullmatch(r"[a-z][a-z0-9_]+", v):
            raise ValueError(
                f"Agent name {v!r} must be lowercase snake_case "
                f"(matched against [a-z][a-z0-9_]+)"
            )
        return v


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------
class AgentRegistry:
    """A collection of :class:`AgentDefinition` objects with query helpers.

    Constructed once at import time as :data:`REGISTRY`. The dispatcher,
    planner, executor and policy gate all consume it.
    """

    def __init__(self, agents: Sequence[AgentDefinition]) -> None:
        self._by_name = {a.name: a for a in agents}
        if len(self._by_name) != len(agents):
            seen: set[str] = set()
            dupes: list[str] = []
            for a in agents:
                if a.name in seen:
                    dupes.append(a.name)
                seen.add(a.name)
            raise ValueError(
                f"Duplicate agent name(s) in registry: {dupes}"
            )

        # Detect tool ownership clashes (a tool name in two agents). Different
        # MCP namespaces are unique by design, but we double-check here so
        # an editing mistake surfaces immediately at import time.
        owners: dict[str, str] = {}
        for agent in self._by_name.values():
            for tool in agent.tools:
                if tool in owners:
                    raise ValueError(
                        f"Tool {tool!r} is claimed by both "
                        f"{owners[tool]!r} and {agent.name!r}; "
                        f"each tool must have exactly one owner."
                    )
                owners[tool] = agent.name
        self._tool_owner = owners

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[AgentDefinition]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def get(self, name: str) -> Optional[AgentDefinition]:
        return self._by_name.get(name)

    def names(self) -> List[str]:
        return list(self._by_name.keys())

    def all(self) -> List[AgentDefinition]:
        return list(self._by_name.values())

    def tool_owner(self, tool_name: str) -> Optional[str]:
        """Return the agent name that owns ``tool_name`` (None if unregistered)."""
        return self._tool_owner.get(tool_name)

    def gated_agents(self) -> List[AgentDefinition]:
        """Subset with non-None policy gates - useful for observability."""
        return [a for a in self._by_name.values() if a.policy_gate is not None]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_step(
        self,
        step: PlanStep,
        intent_flags: Dict[str, bool],
    ) -> List[str]:
        """Return a list of error messages for ``step`` against this registry.

        Empty list = step is valid. Errors include:

        * unknown agent name
        * tool in ``step.tool_subset`` not owned by ``step.agent``
        * policy gate unsatisfied (when ``hard_block_unless_match=True``)

        ``intent_flags`` is the dict the Phase 1 classifier produces
        (see ``src.core.router.classify_query``). Used by the
        planner-output validator (Day 4 work) and by the executor at
        scheduling time as a defence-in-depth check that the
        planner's choices are consistent with the classifier's intent.
        """
        errors: List[str] = []
        agent = self.get(step.agent)
        if agent is None:
            errors.append(
                f"step {step.id}: unknown agent {step.agent!r}; "
                f"valid agents: {self.names()}"
            )
            return errors  # further checks meaningless without an agent

        for tool in step.tool_subset:
            if tool not in agent.tools:
                errors.append(
                    f"step {step.id}: tool {tool!r} is not owned by "
                    f"agent {step.agent!r}. "
                    f"This agent's tools: {sorted(agent.tools)}"
                )

        if (
            agent.policy_gate is not None
            and agent.policy_gate.hard_block_unless_match
            and not agent.policy_gate.matches(intent_flags)
        ):
            errors.append(
                f"step {step.id}: agent {step.agent!r} is policy-gated; "
                f"the classifier did not set any required intent flag "
                f"({agent.policy_gate.explain()}). "
                f"Classifier intent_flags: {dict(intent_flags)}"
            )

        return errors

    def validate_plan(
        self,
        plan: Plan,
        intent_flags: Dict[str, bool],
    ) -> List[str]:
        """Aggregate :meth:`validate_step` across every step in ``plan``."""
        errors: List[str] = []
        for step in plan.steps:
            errors.extend(self.validate_step(step, intent_flags))
        return errors

    # ------------------------------------------------------------------
    # Planner-prompt material
    # ------------------------------------------------------------------
    def planner_catalog_text(self) -> str:
        """Render the catalog as markdown for the planner LLM's system prompt.

        Order matters: ungated agents first (most likely to be useful),
        gated agents last with their gates loud. The planner reads this
        once per request so it should stay readable and bounded - keep
        descriptions short.
        """
        ungated = [a for a in self if a.policy_gate is None]
        gated = [a for a in self if a.policy_gate is not None]

        lines: List[str] = ["## Available Agents", ""]
        for a in ungated + gated:
            lines.append(f"### `{a.name}` — {a.title}")
            lines.append(a.description.strip())
            if a.role_hint:
                lines.append(f"_Use for_: {a.role_hint.strip()}")
            if a.tools:
                tool_block = ", ".join(f"`{t}`" for t in a.tools)
                lines.append(f"_Tools_: {tool_block}")
            else:
                lines.append("_Tools_: none (LLM-only agent)")
            if a.policy_gate is not None:
                lines.append(
                    f"⚠️ **POLICY GATE** — {a.policy_gate.explain()}"
                )
            lines.append("")
        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Canonical agent catalog
# ---------------------------------------------------------------------------
# Tool names match the namespaced output of ``mcp_servers.get_tools()`` as of
# 2026-06-29. Total tool count = 37 across 8 agents (verified by tests). When
# a tool is added or removed in src/mcp/*, update its entry here
# AND bump src/config/mcp_servers.py's ``known_counts``.
# ---------------------------------------------------------------------------
RESEARCH_AGENT = AgentDefinition(
    name="research_agent",
    title="Research Agent",
    description=(
        "Performs web search, retrieves company briefs, and pulls news "
        "articles (recent and historical). The agent's job is to "
        "collect external textual context about a ticker, sector, or "
        "topic — NOT to compute valuations or extract claims."
    ),
    role_hint=(
        "any query needing news, sector context, macro themes, or "
        "open-ended web search"
    ),
    tools=(
        "research__search_news",
        "research__search_historical_news",
        "research__search_web",
        "research__get_company_brief",
        "research__get_key_catalysts",
        "research__get_analyst_takes",
        "research__list_supported_tickers",
    ),
)

US_STOCK_AGENT = AgentDefinition(
    name="us_stock_agent",
    title="US Stock Agent",
    description=(
        "Fetches live US-equity quotes, fundamentals, growth, defensive, "
        "and moat-signal metrics for tickers listed on US exchanges "
        "(AAPL, NVDA, TSLA, MSFT, GOOGL, WDC, etc.). Returns structured "
        "numbers, not narrative."
    ),
    role_hint="any US ticker for which we need fundamentals or quote data",
    tools=(
        "us_stock__get_quote",
        "us_stock__get_fundamentals",
        "us_stock__get_growth_metrics",
        "us_stock__get_defensive_metrics",
        "us_stock__get_moat_signals",
        "us_stock__list_supported_tickers",
    ),
)

INDIAN_STOCK_AGENT = AgentDefinition(
    name="indian_stock_agent",
    title="Indian Stock Agent",
    description=(
        "NSE / BSE tickers (TCS, INFY, RELIANCE, HDFCBANK, etc.): quote, "
        "fundamentals, growth, defensive, moat. Also warehouse-backed "
        "NIFTY-50 market data — exchange-sourced price history, 30-day top "
        "movers, and weekly sector performance (equity-pipeline). Currency "
        "fields converted to USD; native currency reported."
    ),
    role_hint=(
        "any NSE / BSE ticker for fundamentals/quote, or NSE market-wide "
        "questions like price history, top movers, or sector performance"
    ),
    tools=(
        "indian_stock__get_quote",
        "indian_stock__get_fundamentals",
        "indian_stock__get_growth_metrics",
        "indian_stock__get_defensive_metrics",
        "indian_stock__get_moat_signals",
        "indian_stock__list_supported_tickers",
        "indian_stock__get_price_history",
        "indian_stock__get_technicals",
        "indian_stock__get_top_movers",
        "indian_stock__get_sector_performance",
    ),
)

FILINGS_AGENT = AgentDefinition(
    name="filings_agent",
    title="Filings Agent",
    description=(
        "Pulls regulatory filings and IR-page documents: SEC EDGAR "
        "(10-K / 10-Q / 8-K) for US issuers, BSE / NSE Reg 30 + Reg 33 "
        "filings for Indian issuers, plus Annual Reports and concall "
        "transcripts via the Screener.in aggregator. Layout-aware PDF "
        "extraction is built in. Returns plain text and metadata, no "
        "interpretation."
    ),
    role_hint=(
        "queries that need primary-source filings, transcripts, or "
        "Annual Reports"
    ),
    tools=(
        "research__get_sec_filings",
        "research__fetch_sec_document",
        "research__get_indian_filings",
        "research__fetch_indian_document",
        "research__get_screener_snapshot",
        "research__get_indian_concall_urls",
        "research__get_indian_annual_reports",
    ),
)

PORTFOLIO_AGENT = AgentDefinition(
    name="portfolio_agent",
    title="Portfolio Agent",
    description=(
        "Reads the user's holdings from the (mocked) Upstox-style "
        "broker fixture and computes deterministic Python summaries: "
        "sector allocation, concentration risks, diversification "
        "score. No web access, no LLM-derived numbers."
    ),
    role_hint=(
        "queries that explicitly reference 'my portfolio', 'my "
        "holdings', or the user's positions"
    ),
    tools=(
        "portfolio__list_supported_users",
        "portfolio__get_holdings",
        "portfolio__get_portfolio_summary",
        "portfolio__get_sector_allocation",
        "portfolio__get_concentration_risks",
        "portfolio__get_diversification_score",
    ),
)

SYNTHESIZER = AgentDefinition(
    name="synthesizer",
    title="Synthesizer",
    description=(
        "Final-report writer. Takes the outputs of all upstream steps "
        "from the scratchpad and produces a single coherent markdown "
        "report grounded in those outputs. No tools — pure LLM "
        "synthesis. Always the last step in any non-trivial plan."
    ),
    role_hint="every multi-step plan should end with this agent",
    tools=(),  # no tools by design
)

# --- POLICY-GATED AGENTS -----------------------------------------------------

CLAIM_AGENT = AgentDefinition(
    name="claim_agent",
    title="Claim Tracking Agent",
    description=(
        "Extracts forward-looking claims from corporate documents "
        "(transcripts, 10-Ks, press releases) and produces verdicts "
        "comparing each claim against the latest evidence. "
        "DELIBERATELY EXPENSIVE — runs 1 LLM call per claim and 1 per "
        "verdict. Only include in a plan when the classifier has "
        "decided the user wants claim tracking."
    ),
    role_hint=(
        "queries that explicitly ask whether management delivered on past "
        "guidance or commitments — including any natural paraphrase like "
        "'did X follow through', 'has Y kept its promises', "
        "'fulfilled their pledges'"
    ),
    tools=(
        "research__extract_forward_claims",
        "research__compare_claim_to_reality",
    ),
    policy_gate=PolicyGate(
        description=(
            "Claim tracking is opt-in. The classifier must set "
            "wants_claim_tracking=True based on its semantic reading "
            "of the query — no rule-based phrase matching."
        ),
        required_intent_flags=["wants_claim_tracking"],
    ),
)

PANEL_AGENT = AgentDefinition(
    name="panel_agent",
    title="Investor Panel",
    description=(
        "Runs the multi-round Buffett / Wood / Graham debate over a "
        "ticker or portfolio. Internally orchestrates three persona "
        "sub-agents on a shared scratchpad with sequential rounds and "
        "convergence detection. Heavy: ~60-180 seconds per run. Only "
        "include when the classifier has decided the user wants a "
        "panel view."
    ),
    role_hint=(
        "queries that explicitly request the investor panel, the "
        "named personas, or any natural paraphrase like 'what would "
        "the experts say' or 'a debate among investors'"
    ),
    # The panel agent doesn't expose tools to the planner directly -
    # internally it spawns persona ReAct agents which do have tools.
    # From the planner's perspective, this is a zero-tool agent.
    tools=(),
    policy_gate=PolicyGate(
        description=(
            "The investor panel is opt-in due to its compute cost. "
            "The classifier must set wants_panel_debate=True based on "
            "its semantic reading of the query."
        ),
        required_intent_flags=["wants_panel_debate"],
    ),
)


# --- OPT-IN CROSS-PROJECT AGENT ----------------------------------------------
# Surfaces the sibling automated-trading project's READ-ONLY quant backtester
# (via its MCP server). Added to the live registry ONLY when that server is
# enabled (QUANT_MCP_PYTHON + QUANT_MCP_CWD set), so the planner never
# advertises a capability that isn't connected. Backtesting only — the MCP
# server exposes no execution surface (see automated-trading/quant_mcp.py).
QUANT_AGENT = AgentDefinition(
    name="quant_agent",
    title="Quant Strategy Agent",
    description=(
        "Read-only systematic F&O research (sibling automated-trading project): "
        "lists disclosed NIFTY options strategy templates (short straddle, iron "
        "condor) and backtests them on warehoused NSE EOD data, returning "
        "risk-adjusted metrics (Sharpe, expectancy, drawdown, alpha) plus a 2x "
        "cost-stress check. Backtesting only — places no trades."
    ),
    role_hint=(
        "backtesting or evaluating systematic NIFTY options strategies "
        "(short straddle / iron condor), or their historical performance"
    ),
    tools=(
        "quant__list_strategies",
        "quant__backtest_strategy",
    ),
)


def _quant_enabled() -> bool:
    """The quant agent/server is opt-in (sibling project, separate runtime)."""
    return bool(os.getenv("QUANT_MCP_PYTHON") and os.getenv("QUANT_MCP_CWD"))


_REGISTRY_AGENTS = [
    RESEARCH_AGENT,
    US_STOCK_AGENT,
    INDIAN_STOCK_AGENT,
    FILINGS_AGENT,
    PORTFOLIO_AGENT,
    SYNTHESIZER,
    CLAIM_AGENT,
    PANEL_AGENT,
]
if _quant_enabled():
    # Sit with the ungated data/analysis agents, before the synthesizer.
    _REGISTRY_AGENTS.insert(5, QUANT_AGENT)

REGISTRY = AgentRegistry(_REGISTRY_AGENTS)


__all__ = [
    "PolicyGate",
    "AgentDefinition",
    "AgentRegistry",
    "REGISTRY",
    "RESEARCH_AGENT",
    "US_STOCK_AGENT",
    "INDIAN_STOCK_AGENT",
    "FILINGS_AGENT",
    "PORTFOLIO_AGENT",
    "SYNTHESIZER",
    "CLAIM_AGENT",
    "PANEL_AGENT",
    "QUANT_AGENT",
]
