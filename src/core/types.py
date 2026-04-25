"""Typed core for the planner-first architecture (Plan, PlanStep, Scratchpad).

This module defines the data types every other module in the new
planner-executor pipeline depends on:

    Phase 1  Classify     -> RouteDecision (existing, in src/core/router.py)
    Phase 2  Plan         -> Plan { goal, rationale, steps[] }
    Phase 3  Execute      -> Scratchpad { results[] }, ExecutionState
    Phase 4  Join         -> JoinDecision { action, additional_steps?, ... }
    Phase 5  Synthesize   -> str (markdown report)

Design notes
------------
* **Pydantic v2** for runtime validation, JSON-schema generation (used by
  the planner's ``response_format``), and clear error messages.
* **DAG-based plans.** Each :class:`PlanStep` declares ``depends_on`` (list
  of prior step IDs) so the executor can topologically sort and run
  independent steps in parallel.
* **Tool-subset isolation.** Each step declares ``tool_subset`` - the
  exact tool names it is allowed to call. The executor will build a
  scoped agent containing ONLY those tools, never the full system tool
  pool. This is the architectural fix for the "deep research also runs
  claim analysis" problem.
* **Variable references** in inputs use the ``#<step_id>`` or
  ``#<step_id>.<key>`` syntax (LLMCompiler / ReWOO style). The executor
  resolves these from the scratchpad at runtime.
* **Single shared scratchpad** holds every step's output, but each
  step's prompt only includes its declared dependencies. This is the
  hybrid context model from the architecture doc - Cognition's
  Principle 1 satisfied (full traces shareable), Anthropic's
  scoped-context principle also satisfied.
* **Status tracking** on every step lets the executor handle partial
  failures and the joiner decide whether to re-plan.

Stability contract
------------------
The shapes here are the public API for the planner LLM (the JSON schema
goes into its ``response_format`` field). Adding optional fields is
safe; renaming or removing fields is a breaking change for any
historical Plans we want to replay from logs. Bump ``PLAN_SCHEMA_VERSION``
when making incompatible changes.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PLAN_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Intent-flag vocabulary
# ---------------------------------------------------------------------------
# The classifier in Phase 1 (``src/core/router.py``) emits a set of boolean
# intent flags describing what the user wants. The registry's policy gates
# (``src/core/agents/registry.py``) consume these flags - no text matching,
# no regex. The classifier is the SOLE place where natural-language
# understanding happens; the registry just enforces structural consistency.
#
# This frozenset is the shared vocabulary. Adding a new flag means:
#   1. Add the name here
#   2. Make the classifier produce it (with semantic LLM judgement)
#   3. Reference it in any gated agent's ``required_intent_flags``
#
# Validators on ``PolicyGate.required_intent_flags`` reject typos against
# this set, so an invalid spelling fails at import time.
KNOWN_INTENT_FLAGS = frozenset({
    # Output-shape flags
    "wants_claim_tracking",     # user explicitly asked for claim verification
    "wants_panel_debate",       # user explicitly asked for the persona panel
    "wants_filings",            # user wants raw 10-K / 10-Q / Annual Report content
    "wants_portfolio_data",     # user is asking about THEIR portfolio, not just market
    "wants_historical_news",    # user wants old news (claim or other historical context)
    "wants_deep_research",      # user wants a long-form, multi-step research run
})


# ---------------------------------------------------------------------------
# Status enums (kept as Literal types for cheap JSON schema friendliness)
# ---------------------------------------------------------------------------
StepStatus = Literal[
    "pending",   # not yet started
    "running",   # being executed right now
    "complete",  # finished, output available
    "failed",    # raised an exception or returned a structured error
    "skipped",   # the joiner / a dependency failure caused us to skip it
]

JoinAction = Literal[
    "finish",   # the gathered results are sufficient; synthesise the report
    "replan",   # a gap exists; the joiner has produced ``additional_steps``
    "abort",    # unrecoverable; bail out with whatever partial output we have
]


# ---------------------------------------------------------------------------
# Variable-reference syntax in step inputs
# ---------------------------------------------------------------------------
# A step input can be a literal value OR a reference to another step's output:
#
#     "#3"           -> the entire output of step 3
#     "#3.url"       -> the ``url`` field of step 3's output (if it's a dict)
#     "#3.items[0]"  -> the first element of step 3's ``items`` list
#
# The executor resolves these against the scratchpad just before invoking the
# step's agent. ``_VAR_REF_RE`` is the canonical parser; do NOT duplicate it
# elsewhere in the codebase.
_VAR_REF_RE = re.compile(r"^#(\d+)(?:\.([\w\[\]\.]+))?$")


def parse_var_ref(value: str) -> Optional[Tuple[int, Optional[str]]]:
    """If ``value`` is a variable reference like ``"#3.url"``, return ``(3, "url")``.

    Returns ``(step_id, None)`` for a bare ``"#3"``. Returns ``None`` if
    ``value`` isn't a reference. Centralised so the executor and the
    joiner agree on the syntax.
    """
    if not isinstance(value, str):
        return None
    m = _VAR_REF_RE.match(value.strip())
    if m is None:
        return None
    return int(m.group(1)), m.group(2)


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------
class PlanStep(BaseModel):
    """One node in the execution DAG.

    The planner produces these; the executor consumes them. Each step is
    one invocation of one agent with a strictly scoped tool subset.

    Field semantics
    ---------------
    * ``id`` MUST be unique within the plan and is the value other
      steps reference via ``depends_on`` and ``"#<id>"`` variable refs.
    * ``description`` is human-readable; surfaced in the UI and used as
      part of the agent's user prompt.
    * ``agent`` is a key into the agent registry (e.g. ``"research_agent"``).
      Validated at execution time; not constrained at the type level
      so adding new agents doesn't require a schema bump.
    * ``tool_subset`` is the exhaustive list of tool names this step
      is allowed to call. The executor enforces this; a tool the
      planner forgot to declare is unreachable.
    * ``inputs`` is an arbitrary JSON-shaped payload passed to the
      agent. String values may contain ``"#<id>"`` references (see
      :func:`parse_var_ref`) which the executor resolves before invocation.
    * ``depends_on`` is a list of step IDs whose output this step
      needs to read. The executor uses it for topological ordering
      AND for context scoping (the step's prompt only includes
      these dependencies' outputs, not the entire scratchpad).
    * ``max_tool_calls`` caps the ReAct loop budget for this specific
      step. Prevents runaway agents on otherwise-bounded subtasks.
    * ``rationale`` explains why this step is in the plan; useful for
      debugging the planner LLM and for audit trails.
    """

    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        ...,
        ge=1,
        description=(
            "Step number, unique within the plan. Use sequential ints "
            "starting at 1; the executor tolerates gaps but it's harder "
            "to read."
        ),
    )
    description: str = Field(
        ...,
        min_length=4,
        max_length=400,
        description=(
            "One-sentence description of what this step accomplishes. "
            "Surfaced verbatim in the UI and concatenated into the "
            "agent's user prompt."
        ),
    )
    agent: str = Field(
        ...,
        min_length=2,
        description=(
            "Name of the agent to handle this step. Must match a key "
            "registered in the agent registry (e.g. 'research_agent', "
            "'us_stock_agent', 'claim_agent', 'synthesizer'). Unknown "
            "agent names cause the executor to fail this step with a "
            "structured error."
        ),
    )
    tool_subset: List[str] = Field(
        default_factory=list,
        description=(
            "Exhaustive list of tool names this step is allowed to "
            "invoke. The executor builds a scoped agent containing "
            "ONLY these tools - tools not in this list are unreachable "
            "to the LLM during this step. Empty list is valid for "
            "synthesizer / pure-LLM steps with no tools."
        ),
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Inputs handed to the agent. Values may be variable "
            "references like '#3' or '#3.url' which the executor "
            "resolves from the scratchpad just before invocation."
        ),
    )
    depends_on: List[int] = Field(
        default_factory=list,
        description=(
            "Step IDs this step depends on. The executor will wait "
            "for all of these to complete before scheduling this "
            "step, AND will include their outputs in this step's "
            "context (but no others, to keep the prompt focused)."
        ),
    )
    max_tool_calls: int = Field(
        default=20,
        ge=1,
        le=200,
        description=(
            "Hard cap on the number of tool calls this step's agent "
            "may make. Enforced inside the ReAct loop. Most steps "
            "should be 5-15; long-running steps (e.g. paginating a "
            "300-page Annual Report) may justify higher."
        ),
    )
    rationale: str = Field(
        default="",
        max_length=400,
        description=(
            "Why the planner included this step. Optional but "
            "strongly recommended; powers debugging and the audit "
            "trail."
        ),
    )

    # Replan-tracking metadata (set by the joiner / dispatcher, not the planner)
    replan_round: int = Field(
        default=0,
        ge=0,
        description=(
            "0 = part of the original plan; 1+ = added by the joiner "
            "during a replan iteration. Internal field; planner LLM "
            "should leave at 0."
        ),
    )

    @field_validator("tool_subset")
    @classmethod
    def _no_empty_tool_names(cls, v: List[str]) -> List[str]:
        for t in v:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("tool_subset entries must be non-empty strings")
        return [t.strip() for t in v]

    @field_validator("depends_on")
    @classmethod
    def _depends_on_no_self(cls, v: List[int]) -> List[int]:
        # Self-dependencies are caught more cleanly at Plan-level (where we
        # know the step's own id) but we can at least reject duplicates here.
        if len(set(v)) != len(v):
            raise ValueError("depends_on must not contain duplicates")
        return v


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
class Plan(BaseModel):
    """An ordered DAG of :class:`PlanStep` s emitted by the planner.

    The executor receives a Plan and runs it. Cross-cutting validators
    enforce DAG well-formedness:

    * step IDs are unique within the plan
    * every ``depends_on`` references an existing step
    * no self-dependencies
    * no cycles (so we can topologically sort)
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=PLAN_SCHEMA_VERSION,
        description="Version of the Plan schema this object was emitted against.",
    )
    goal: str = Field(
        ...,
        min_length=4,
        max_length=400,
        description="The user's goal in one sentence; planner-restated.",
    )
    rationale: str = Field(
        ...,
        min_length=4,
        max_length=600,
        description=(
            "Why the planner chose this specific structure of steps. "
            "Powers explainability ('here's what I'm going to do') "
            "and is shown in the UI before execution begins."
        ),
    )
    steps: List[PlanStep] = Field(
        default_factory=list,
        description="The DAG of steps to execute, in any order.",
    )
    estimated_complexity: Literal["trivial", "low", "medium", "high"] = Field(
        default="medium",
        description=(
            "Planner's self-rated complexity for this query. Used to "
            "scale per-step budgets and decide whether to allow "
            "re-planning."
        ),
    )

    # ----- Validation -----------------------------------------------------
    @model_validator(mode="after")
    def _validate_dag(self) -> "Plan":
        ids = [s.id for s in self.steps]
        # Unique IDs
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"Plan step IDs must be unique within the plan; got {ids}"
            )
        valid_ids: Set[int] = set(ids)
        # All depends_on references exist + no self-loops
        for s in self.steps:
            for dep in s.depends_on:
                if dep == s.id:
                    raise ValueError(
                        f"Step {s.id} depends on itself; self-loops not allowed"
                    )
                if dep not in valid_ids:
                    raise ValueError(
                        f"Step {s.id}.depends_on={dep} references a "
                        f"non-existent step. Valid IDs: {sorted(valid_ids)}"
                    )
        # Cycle detection via a Kahn-style topological sort attempt
        if self.steps:
            self._topological_sort()  # raises if cycles exist
        return self

    # ----- Helpers used by the executor + joiner --------------------------
    def step_by_id(self, step_id: int) -> Optional[PlanStep]:
        """Return the step with ``step_id`` or ``None`` if not in the plan."""
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def _topological_sort(self) -> List[PlanStep]:
        """Return steps in topological order; raise on cycles.

        Used internally for validation. The executor uses
        :meth:`ready_steps` instead because it can react to partial
        completions.
        """
        in_degree: Dict[int, int] = {s.id: len(s.depends_on) for s in self.steps}
        children: Dict[int, List[int]] = {s.id: [] for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                children[dep].append(s.id)
        # Kahn's algorithm
        ready = [sid for sid, deg in in_degree.items() if deg == 0]
        ordered: List[int] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for child in children[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self.steps):
            cyc = [sid for sid, deg in in_degree.items() if deg > 0]
            raise ValueError(
                f"Plan has a dependency cycle involving steps: {cyc}"
            )
        # Map ordered IDs back to PlanStep objects
        by_id = {s.id: s for s in self.steps}
        return [by_id[i] for i in ordered]

    def topological_order(self) -> List[PlanStep]:
        """Public alias for :meth:`_topological_sort` (for diagnostics)."""
        return self._topological_sort()

    def ready_steps(self, completed_ids: Set[int]) -> List[PlanStep]:
        """Return steps whose dependencies are all in ``completed_ids``.

        Excludes steps already in ``completed_ids`` themselves. The
        executor calls this on every tick to decide what new work to
        schedule.
        """
        ready: List[PlanStep] = []
        for s in self.steps:
            if s.id in completed_ids:
                continue
            if all(dep in completed_ids for dep in s.depends_on):
                ready.append(s)
        return ready

    def all_tools(self) -> Set[str]:
        """Union of every tool any step in this plan can call.

        Cheap pre-flight check the dispatcher uses to ensure the MCP
        adapter has all required tools loaded before execution starts.
        """
        out: Set[str] = set()
        for s in self.steps:
            out.update(s.tool_subset)
        return out

    def all_agents(self) -> Set[str]:
        """Union of every agent name any step in this plan invokes.

        Used by :func:`PolicyGate` to enforce per-agent phrase
        whitelists (e.g. ``claim_agent`` requires explicit user intent).
        """
        return {s.agent for s in self.steps}


# ---------------------------------------------------------------------------
# Step result + Scratchpad
# ---------------------------------------------------------------------------
class StepResult(BaseModel):
    """The outcome of executing a single :class:`PlanStep`.

    The executor appends one of these to the :class:`Scratchpad` for
    every step that was attempted (including failures and skips). The
    joiner reads them to decide finish-or-replan.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: int = Field(..., ge=1)
    status: StepStatus
    output: Any = Field(
        default=None,
        description=(
            "Whatever the step's agent produced. Typically a dict "
            "(tool result), a string (LLM rationale), or a list. "
            "On ``failed`` status this is None and ``error`` is set."
        ),
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="Names of tools actually invoked during this step.",
    )
    started_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when execution began.",
    )
    completed_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when execution ended; None if still running.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if ``status == 'failed'``; None otherwise.",
    )
    error_type: Optional[str] = Field(
        default=None,
        description="Exception class name on failure (e.g. 'TimeoutError').",
    )

    @property
    def duration_s(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    @property
    def is_terminal(self) -> bool:
        return self.status in ("complete", "failed", "skipped")


class UnmetDependency(BaseModel):
    """A scoped agent's structured request for help from another agent.

    Emitted via the synthetic ``request_assistance(target_agent, reason)``
    tool that every :class:`ScopedAgent` exposes (see
    ``src/core/agents/_base.py``). The agent **never calls another agent
    directly** - this just records that its own step has hit a gap that
    the planner could fill via a replan.

    The executor surfaces these to the joiner. The joiner's policy
    (e.g. "any unmet_dependencies → return ``replan``") drives the
    decision; we do not auto-spawn new steps from inside ScopedAgent
    because the planner is the single source of truth for plan shape.
    """

    model_config = ConfigDict(extra="forbid")

    requested_by_step_id: int = Field(
        ..., gt=0,
        description="ID of the step whose agent issued the request.",
    )
    target_agent: str = Field(
        ..., min_length=1,
        description=(
            "Name of the agent the requester believes can answer the "
            "gap (e.g. 'us_stock_agent'). The validator below catches "
            "obvious garbage; the planner-side validator (in the "
            "registry) catches unknown agents."
        ),
    )
    reason: str = Field(
        ..., min_length=10, max_length=500,
        description=(
            "Why the gap matters - in the requester's own words. "
            "Surfaced verbatim to the planner so its replan can be "
            "informed by the actual ask."
        ),
    )
    raised_at: float = Field(
        default_factory=time.time,
        description="UNIX timestamp; useful for ordering and logs.",
    )


class Scratchpad(BaseModel):
    """Shared blackboard the executor and joiner read/write across a run.

    Hybrid context model (per architecture doc Section 4):

    * The scratchpad holds **every** step's full output, so the joiner
      and the synthesizer have a complete trace (Cognition's Principle 1).
    * But each step's *prompt* only includes the outputs of its
      declared ``depends_on`` (Anthropic's scoped context); see
      :meth:`relevant_results_for_step`.

    Mutation policy
    ---------------
    Only the executor appends to ``results``. The joiner reads but
    never mutates. Parallel steps write to disjoint ``step_id`` keys
    so there is no race condition - asyncio.gather aggregates results
    by id before the executor commits them.

    Scoped agents (see :mod:`src.core.agents._base`) may also append
    to ``unmet_dependencies`` via the synthetic ``request_assistance``
    tool. Unlike ``results`` this list can grow concurrently from
    multiple parallel steps; we therefore use ``add_unmet_dependency``
    instead of direct mutation so the contract is explicit at every
    write site.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="The original user query (for audit).")
    results: Dict[int, StepResult] = Field(default_factory=dict)
    unmet_dependencies: List[UnmetDependency] = Field(
        default_factory=list,
        description=(
            "Structured 'I need help from X' notes raised by scoped "
            "agents during execution. Read by the joiner; the joiner "
            "decides whether to replan based on these."
        ),
    )

    # ----- Read API -------------------------------------------------------
    def get(self, step_id: int) -> Optional[StepResult]:
        return self.results.get(step_id)

    def get_output(self, step_id: int) -> Any:
        """Convenience: return the ``output`` of a completed step or ``None``."""
        r = self.results.get(step_id)
        return r.output if (r and r.status == "complete") else None

    def completed_ids(self) -> Set[int]:
        """IDs of steps with status ``complete`` (excludes failed and skipped)."""
        return {sid for sid, r in self.results.items() if r.status == "complete"}

    def terminal_ids(self) -> Set[int]:
        """IDs of steps that have stopped (complete + failed + skipped).

        The executor uses this when deciding what's "done" for the
        purposes of scheduling new ready steps - a failed step
        unblocks no descendants, but its descendants still see it as
        done so they can be skipped/handled.
        """
        return {sid for sid, r in self.results.items() if r.is_terminal}

    def has_failures(self) -> bool:
        return any(r.status == "failed" for r in self.results.values())

    def relevant_results_for_step(self, step: PlanStep) -> Dict[int, StepResult]:
        """Subset of results matching the step's declared dependencies.

        This is the Anthropic-style scoped context: a step's prompt
        gets ONLY its dependencies' outputs, not the whole scratchpad.
        Dramatically reduces context bloat and prevents cross-talk.
        """
        return {
            dep: self.results[dep]
            for dep in step.depends_on
            if dep in self.results
        }

    # ----- Variable reference resolution ---------------------------------
    def resolve_value(self, value: Any) -> Any:
        """Recursively replace ``"#<id>"`` / ``"#<id>.field"`` refs in ``value``.

        Used by the executor on a step's ``inputs`` before invoking
        the agent. Non-string / non-ref values pass through unchanged.

        Handles dotted paths and list-indexing in the field part:
        ``"#3.items[0].url"`` resolves to the URL of the first item.
        """
        if isinstance(value, str):
            ref = parse_var_ref(value)
            if ref is None:
                return value
            step_id, path = ref
            base = self.get_output(step_id)
            if base is None:
                return None
            return _follow_path(base, path) if path else base

        if isinstance(value, dict):
            return {k: self.resolve_value(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self.resolve_value(v) for v in value]

        return value

    # ----- Write API (executor only) -------------------------------------
    def add(self, result: StepResult) -> None:
        """Insert / overwrite the result for ``result.step_id``.

        Overwriting is allowed because a re-plan iteration may rerun a
        previously-failed step with refined inputs.
        """
        self.results[result.step_id] = result

    def add_unmet_dependency(
        self,
        *,
        requested_by_step_id: int,
        target_agent: str,
        reason: str,
    ) -> UnmetDependency:
        """Record a request-for-help from a scoped agent.

        Returns the constructed :class:`UnmetDependency` so callers (the
        ``request_assistance`` synthetic tool) can echo a confirmation
        back to the LLM. We construct the model here rather than letting
        the caller pass one so the call site can't accidentally write to
        ``raised_at`` or skip validation.
        """
        dep = UnmetDependency(
            requested_by_step_id=requested_by_step_id,
            target_agent=target_agent,
            reason=reason,
        )
        self.unmet_dependencies.append(dep)
        return dep


# ---------------------------------------------------------------------------
# Variable path resolution helper (used by Scratchpad.resolve_value)
# ---------------------------------------------------------------------------
_PATH_TOKEN_RE = re.compile(r"([\w\-]+)|\[(\d+)\]")


def _follow_path(base: Any, path: str) -> Any:
    """Walk ``path`` (e.g. ``"items[0].url"``) into ``base`` and return the leaf.

    Returns ``None`` for any miss along the way; never raises. We deliberately
    keep this lenient so a planner-introduced typo in a variable reference
    yields a missing-input error at the next agent boundary, not a crash here.
    """
    cur = base
    for m in _PATH_TOKEN_RE.finditer(path):
        key, idx = m.group(1), m.group(2)
        try:
            if key is not None:
                cur = cur[key] if isinstance(cur, dict) else getattr(cur, key, None)
            else:
                cur = cur[int(idx)] if isinstance(cur, (list, tuple)) else None
        except (KeyError, IndexError, TypeError):
            return None
        if cur is None:
            return None
    return cur


# ---------------------------------------------------------------------------
# Execution state (top-level container the dispatcher hands around)
# ---------------------------------------------------------------------------
class ExecutionState(BaseModel):
    """Snapshot of an in-flight planner-executor run.

    Bundles the immutable inputs (query, plan) with the mutable state
    (scratchpad, replan counter, timing). The dispatcher creates one
    of these per request and passes it through every phase.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    query: str
    plan: Plan
    scratchpad: Scratchpad
    user_id: str = Field(default="demo")

    # Replan / iteration tracking
    replan_count: int = Field(
        default=0,
        ge=0,
        description="Number of times the joiner has expanded the plan.",
    )
    max_replans: int = Field(
        default=2,
        ge=0,
        description=(
            "Hard cap on replan iterations. Per the architecture doc, "
            "we synthesise with whatever we have once this is hit."
        ),
    )

    # Timing
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    @property
    def can_replan(self) -> bool:
        return self.replan_count < self.max_replans

    def all_steps_terminal(self) -> bool:
        """True iff every step in ``plan`` has a terminal result in ``scratchpad``."""
        return all(
            sid in self.scratchpad.results
            and self.scratchpad.results[sid].is_terminal
            for sid in (s.id for s in self.plan.steps)
        )


# ---------------------------------------------------------------------------
# Joiner output
# ---------------------------------------------------------------------------
class JoinDecision(BaseModel):
    """The joiner's verdict on whether the plan has produced enough.

    Three terminal actions (per architecture doc Section 4):

    * ``finish``  - results are sufficient; synthesise the final report.
    * ``replan``  - gaps remain; ``additional_steps`` extends the DAG.
    * ``abort``   - unrecoverable; emit a graceful failure to the user.
    """

    model_config = ConfigDict(extra="forbid")

    action: JoinAction = Field(
        ...,
        description=(
            "What the dispatcher should do next. One of "
            "'finish' / 'replan' / 'abort'."
        ),
    )
    reasoning: str = Field(
        ...,
        min_length=4,
        max_length=600,
        description="Joiner's explanation; surfaced in the UI.",
    )
    additional_steps: List[PlanStep] = Field(
        default_factory=list,
        description=(
            "Only populated when ``action == 'replan'``. The dispatcher "
            "merges these into ``state.plan.steps`` (with "
            "``replan_round`` bumped) and re-enters Phase 3."
        ),
    )
    final_summary: Optional[str] = Field(
        default=None,
        description=(
            "Only populated when ``action == 'finish'`` and the joiner "
            "wanted to seed the synthesizer with a one-paragraph "
            "summary. Optional; the synthesizer can ignore it."
        ),
    )

    @model_validator(mode="after")
    def _action_consistency(self) -> "JoinDecision":
        if self.action == "replan" and not self.additional_steps:
            raise ValueError(
                "JoinDecision.action='replan' requires at least one "
                "PlanStep in additional_steps"
            )
        if self.action != "replan" and self.additional_steps:
            raise ValueError(
                f"JoinDecision.additional_steps must be empty when "
                f"action='{self.action}'"
            )
        return self


# ---------------------------------------------------------------------------
# Convenience: a small public surface for dispatcher / planner / executor
# ---------------------------------------------------------------------------
__all__ = [
    "PLAN_SCHEMA_VERSION",
    "KNOWN_INTENT_FLAGS",
    "StepStatus",
    "JoinAction",
    "PlanStep",
    "Plan",
    "StepResult",
    "Scratchpad",
    "UnmetDependency",
    "ExecutionState",
    "JoinDecision",
    "parse_var_ref",
]
