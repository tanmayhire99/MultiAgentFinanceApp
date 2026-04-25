"""``ScopedAgent`` — a single-step agent with a constrained tool surface.

This is the unit of execution under the new planner-executor architecture.
The planner produces a :class:`Plan` whose steps each name an ``agent``
(from :class:`AgentRegistry`) and a ``tool_subset`` (a list of MCP tool
names). The DAG executor instantiates one ``ScopedAgent`` per step,
runs it, and stores the resulting :class:`StepResult` in the shared
:class:`Scratchpad`.

Design contract
---------------
A ``ScopedAgent`` runs **exactly one step**. It is given:

1. A ``PlanStep`` — the description of what to do, the planner-chosen
   ``tool_subset``, and the ``depends_on`` list.
2. A reference to the run's ``Scratchpad`` — for reading prior results
   and for writing ``UnmetDependency`` notes via ``request_assistance``.
3. The full pool of MCP tools (from
   :func:`src.config.mcp_servers.get_tools`). The ``ScopedAgent``
   filters this pool to the planner's strict allow-list.
4. A pre-built chat model.
5. The classifier's ``intent_flags`` so the registry can re-validate
   the policy gate (defensive double-check; the executor should
   already have called :meth:`AgentRegistry.validate_step`).

It exposes:

* ``mcp_tools`` — the filtered, planner-approved tool list
* ``synthetic_tools`` — ``get_prior_result`` and ``request_assistance``
* ``system_prompt`` — generated text including the agent catalog
* ``run() -> StepResult`` — async entry point

The agent is **NOT** a multi-step orchestrator. It does not delegate
to other agents. It does not see results from steps outside its
declared dependencies. If it needs cross-step coordination it must
call ``request_assistance``, which writes to the scratchpad's
``unmet_dependencies`` list - the **planner** then decides whether
to add follow-up steps in a replan.

Why scope helps
---------------
Anthropic's "Building Effective Agents" research multi-agent post
showed that giving every sub-agent a narrow, well-defined tool
surface produced 90.2% better results than letting the orchestrator
work alone. Cognition's "Don't Build Multi-Agents" rebuttal warned
against agents that talk to each other unmoderated. ``ScopedAgent``
is the synthesis: each step is its own little agent with a narrow
surface (Anthropic), but coordination happens only through the
central planner / scratchpad (Cognition). See
``docs/MULTI_AGENT_ARCHITECTURE.md`` Section 3 for citations.

Tests live in ``tests/test_scoped_agent.py`` and exercise:

* construction-time validation
* tool filtering (strict allow-list)
* synthetic tool behaviour in isolation (no LLM call)
* system-prompt assembly
* end-to-end run with a mock model
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.types import PlanStep, Scratchpad, StepResult


log = logging.getLogger("finai.scoped_agent")


# Default ReAct recursion depth. Each iteration is roughly one
# tool call, so this caps a single step at ~25 tool calls.
# Personas in panel.py use 50; we default lower because individual
# steps should be tightly scoped. The executor can override.
DEFAULT_RECURSION_LIMIT = 25


class ScopedAgentError(Exception):
    """Raised when a step fails validation or has an unrecoverable setup error.

    Construction errors are programming bugs (planner produced an
    invalid plan, or the executor tried to instantiate without first
    calling ``validate_step``) — distinct from runtime tool / LLM
    failures, which are captured in ``StepResult(status='failed')``.
    """


# ---------------------------------------------------------------------------
# Synthetic-tool argument schemas
#
# Defined as Pydantic models (not bare type hints) so the underlying
# ``StructuredTool`` can advertise rich JSON-schema docs to the LLM
# planner. The model docstrings become the "description" surface the
# tool-using LLM sees.
# ---------------------------------------------------------------------------
class _PriorResultArgs(BaseModel):
    """Arguments for the ``get_prior_result`` synthetic tool."""

    step_id: int = Field(
        ..., gt=0,
        description=(
            "The integer ID of the prior step whose output you want. "
            "MUST be in your step's depends_on list."
        ),
    )


class _RequestAssistanceArgs(BaseModel):
    """Arguments for the ``request_assistance`` synthetic tool."""

    target_agent: str = Field(
        ..., min_length=1,
        description=(
            "The name of the agent you believe can fill this gap "
            "(e.g. 'us_stock_agent'). See your system prompt's agent "
            "directory for the canonical list."
        ),
    )
    reason: str = Field(
        ..., min_length=10, max_length=500,
        description=(
            "1-2 sentences explaining what data you need and why your "
            "current step cannot proceed without it. Be specific - "
            "the planner sees this verbatim and uses it to decide "
            "whether to add a follow-up step."
        ),
    )


# ---------------------------------------------------------------------------
# ScopedAgent
# ---------------------------------------------------------------------------
class ScopedAgent:
    """Single-step agent: narrow tool surface, awareness of others.

    See module docstring for the full design contract.
    """

    def __init__(
        self,
        *,
        step: PlanStep,
        scratchpad: Scratchpad,
        all_mcp_tools: Sequence[BaseTool],
        model: BaseChatModel,
        registry: AgentRegistry = REGISTRY,
        intent_flags: Optional[Dict[str, bool]] = None,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
    ) -> None:
        self.step = step
        self.scratchpad = scratchpad
        self.registry = registry
        self.intent_flags: Dict[str, bool] = dict(intent_flags or {})
        self.recursion_limit = recursion_limit
        self.model = model

        # 1) Validate the step against the registry. The executor SHOULD
        #    have done this already, but we re-check here so a
        #    hand-constructed ScopedAgent can never escape the gates -
        #    the policy gates are the demo's only safety net against
        #    accidentally running expensive agents.
        errors = registry.validate_step(step, self.intent_flags)
        if errors:
            raise ScopedAgentError(
                f"Step {step.id} failed registry validation:\n  - "
                + "\n  - ".join(errors)
            )
        self.agent_def = registry.get(step.agent)
        # Guard for static analysis - validate_step ensures get() != None
        if self.agent_def is None:  # pragma: no cover - defensive
            raise ScopedAgentError(
                f"Step {step.id}: registry.get({step.agent!r}) returned None "
                "after validate_step passed - this is a registry bug."
            )

        # 2) Filter MCP tools to the strict allow-list
        self.mcp_tools = self._filter_tools(all_mcp_tools, step.tool_subset)

        # 3) Construct synthetic tools (closure over self for scratchpad/step)
        self.synthetic_tools = self._build_synthetic_tools()

        # 4) Build the system prompt
        self.system_prompt = self._build_system_prompt()

        # 5) Compile the ReAct graph
        self._compiled = create_react_agent(
            model=self.model,
            tools=list(self.mcp_tools) + list(self.synthetic_tools),
            prompt=self.system_prompt,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _filter_tools(
        self,
        all_mcp_tools: Sequence[BaseTool],
        tool_subset: Sequence[str],
    ) -> List[BaseTool]:
        """Strict allow-list filter: only tools named in ``tool_subset``.

        Raises ``ScopedAgentError`` if any name in ``tool_subset`` is not
        in ``all_mcp_tools``. This catches planner typos before the LLM
        wastes turns trying to call a non-existent tool.

        Order is preserved from ``tool_subset`` so the LLM sees tools
        in the planner's intended priority order.
        """
        by_name = {t.name: t for t in all_mcp_tools}
        missing = [n for n in tool_subset if n not in by_name]
        if missing:
            raise ScopedAgentError(
                f"Step {self.step.id} declared tool(s) not present in the "
                f"MCP pool: {missing}. "
                f"Available ({len(by_name)} total): "
                f"{sorted(by_name)[:8]}..."
            )
        return [by_name[n] for n in tool_subset]

    def _build_synthetic_tools(self) -> List[BaseTool]:
        """Construct ``get_prior_result`` and ``request_assistance``.

        Both close over ``self.step`` and ``self.scratchpad``. They are
        re-built per ScopedAgent instance because their behaviour
        depends on the step's ``depends_on`` set.
        """
        deps_set = set(self.step.depends_on)
        # Pre-format for stable, sorted display in tool descriptions
        deps_list = sorted(deps_set)
        return [
            self._build_get_prior_result_tool(deps_set, deps_list),
            self._build_request_assistance_tool(),
        ]

    def _build_get_prior_result_tool(
        self,
        deps_set: set[int],
        deps_list: List[int],
    ) -> BaseTool:
        step = self.step
        scratchpad = self.scratchpad

        def _get_prior_result(step_id: int) -> str:
            """Fetch a prior step's output (scoped to declared dependencies)."""
            if step_id not in deps_set:
                return json.dumps({
                    "error": "out_of_scope",
                    "message": (
                        f"Step {step_id} is not in this step's declared "
                        f"dependencies ({deps_list}). If you genuinely "
                        "need this data, call request_assistance to ask "
                        "the orchestrator for a replan instead."
                    ),
                })
            result = scratchpad.get(step_id)
            if result is None:
                return json.dumps({
                    "error": "not_yet_executed",
                    "message": (
                        f"Step {step_id} has no result in the scratchpad. "
                        "Either it has not run yet (executor bug) or "
                        "it failed silently."
                    ),
                })
            if result.status != "complete":
                return json.dumps({
                    "error": "step_not_complete",
                    "step_status": result.status,
                    "step_error": result.error,
                    "step_error_type": result.error_type,
                })
            return json.dumps(
                {
                    "step_id": step_id,
                    "status": result.status,
                    "output": result.output,
                    "tools_used": result.tools_used,
                },
                # ``default=str`` so any datetimes / other non-JSON
                # objects in step output don't crash serialization.
                default=str,
            )

        return StructuredTool.from_function(
            func=_get_prior_result,
            name="get_prior_result",
            description=(
                "Fetch the full output of a prior step from the shared "
                "scratchpad. Accepts ONLY step IDs in your "
                f"depends_on list ({deps_list}). Use this instead of "
                "trying to call other agents' tools yourself - those "
                "tools are not in your tool_subset and will not work."
                if deps_list
                else
                "Fetch the full output of a prior step. NOTE: your "
                "step has NO declared dependencies (depends_on=[]), so "
                "any call to this tool will return an out_of_scope "
                "error. If you discover you need a prior step's output, "
                "call request_assistance and the planner will replan."
            ),
            args_schema=_PriorResultArgs,
        )

    def _build_request_assistance_tool(self) -> BaseTool:
        step = self.step
        scratchpad = self.scratchpad

        def _request_assistance(target_agent: str, reason: str) -> str:
            """Record an unmet-dependency note for the orchestrator."""
            scratchpad.add_unmet_dependency(
                requested_by_step_id=step.id,
                target_agent=target_agent,
                reason=reason,
            )
            log.info(
                "step %d (%s) requested assistance from %s: %s",
                step.id, step.agent, target_agent, reason,
            )
            return json.dumps({
                "recorded": True,
                "target_agent": target_agent,
                "message": (
                    "Your request has been recorded for the orchestrator. "
                    "The orchestrator may add a follow-up step in a "
                    "replan after your step finishes. You should now "
                    "complete your step with whatever data you already "
                    "have, then end your turn."
                ),
            })

        return StructuredTool.from_function(
            func=_request_assistance,
            name="request_assistance",
            description=(
                "Flag a missing dependency. Use this when you discover "
                "you need data from another agent that wasn't included "
                "in your step's depends_on. Does NOT call the other "
                "agent directly - only records a note that the planner "
                "may act on in a replan. After calling this you should "
                "finish your step with whatever data you have."
            ),
            args_schema=_RequestAssistanceArgs,
        )

    def _build_system_prompt(self) -> str:
        """Render the system prompt: task + scope rules + agent directory.

        The directory is **for awareness, not delegation** — see module
        docstring. Including it lets the agent reason about whether
        to call ``request_assistance`` instead of going off-script.
        """
        catalog = self.registry.planner_catalog_text()
        deps = sorted(self.step.depends_on)
        # Use a short empty-list rendering for clarity
        deps_repr = "[]" if not deps else str(deps)
        tools_repr = (
            "[]" if not self.step.tool_subset else list(self.step.tool_subset)
        )
        agent = self.agent_def
        # NOTE: the doubled curly braces in ``f"{{...}}"`` would trip an f-string;
        # we deliberately use plain str.format here for clarity instead.
        return (
            f"You are the **{agent.title}** ({agent.name}) running step "
            f"{self.step.id} of a multi-agent investigation.\n\n"
            "### Your task\n"
            f"{self.step.description.strip()}\n\n"
            "### Your scope\n"
            "You are running ONE step of a larger plan. Stay strictly in "
            "your lane:\n"
            f"- You can ONLY call the MCP tools in your tool_subset: "
            f"{tools_repr}\n"
            "- You can read prior step outputs via "
            "`get_prior_result(step_id)`, but ONLY for step IDs in your "
            f"depends_on list: {deps_repr}\n"
            "- If you need help from another agent, call "
            "`request_assistance(target_agent, reason)` to record a note "
            "for the orchestrator. **Do NOT try to do another agent's "
            "job yourself.**\n\n"
            "### Agent directory (situational awareness only)\n"
            "These are the OTHER agents in the system. You CANNOT call "
            "them directly - knowing what they do helps you decide when "
            "to use `request_assistance`.\n\n"
            f"{catalog}\n\n"
            "### Output\n"
            "Produce a clear, focused answer that addresses your step's "
            "task. Cite specific numbers from your tool results where "
            "applicable. The orchestrator will feed your output to "
            "downstream steps that depend on yours, so be unambiguous "
            "and structured."
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def run(self) -> StepResult:
        """Invoke the ReAct loop and return a :class:`StepResult`.

        Caught exceptions become ``status='failed'`` results so the
        executor can continue running the rest of the DAG. Only
        construction-time bugs (which raise ``ScopedAgentError`` from
        ``__init__``) are propagated.
        """
        started_at = time.time()
        # The user message is short - the heavy lifting is in the system
        # prompt. We still send something so the LLM has a HumanMessage
        # to anchor its turn.
        user_message = (
            f"Please complete step {self.step.id}: "
            f"{self.step.description.strip()}"
        )
        try:
            graph_result = await self._compiled.ainvoke(
                {"messages": [HumanMessage(content=user_message)]},
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception as exc:
            log.exception("step %d crashed during ReAct loop", self.step.id)
            return StepResult(
                step_id=self.step.id,
                status="failed",
                output=None,
                error=str(exc),
                error_type=type(exc).__name__,
                started_at=started_at,
                completed_at=time.time(),
                tools_used=[],
            )

        messages = graph_result.get("messages", []) if isinstance(graph_result, dict) else []
        final_text = self._extract_final_text(messages)
        tools_used = self._collect_tools_used(messages)

        return StepResult(
            step_id=self.step.id,
            status="complete",
            output={"text": final_text},
            tools_used=tools_used,
            started_at=started_at,
            completed_at=time.time(),
        )

    # ------------------------------------------------------------------
    # Trajectory parsing helpers (extracted for testability)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_final_text(messages: Sequence[Any]) -> str:
        """Return the content of the last AIMessage with non-empty content."""
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            content = msg.content
            if not content:
                continue
            if isinstance(content, str):
                return content.strip()
            # OpenAI-style structured content (list of {type, text} blocks)
            if isinstance(content, list):
                parts: List[str] = []
                for blk in content:
                    if isinstance(blk, dict) and "text" in blk:
                        parts.append(str(blk["text"]))
                    elif isinstance(blk, str):
                        parts.append(blk)
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
        return ""

    @staticmethod
    def _collect_tools_used(messages: Sequence[Any]) -> List[str]:
        """Return distinct tool names invoked, in first-seen order.

        Used for both the ``StepResult.tools_used`` audit field and the
        joiner's "did this step actually call any tools?" heuristic.
        """
        seen: List[str] = []
        seen_set: set[str] = set()
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                name = (
                    tc.get("name")
                    if isinstance(tc, dict)
                    else getattr(tc, "name", None)
                )
                if name and name not in seen_set:
                    seen.append(name)
                    seen_set.add(name)
        return seen


__all__ = [
    "DEFAULT_RECURSION_LIMIT",
    "ScopedAgent",
    "ScopedAgentError",
]
