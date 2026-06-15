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
from typing import Any, Callable, Dict, List, Optional, Sequence, AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.core.agents.registry import REGISTRY, AgentRegistry
from src.core.panel import PanelEvent
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
        ...,
        min_length=1,
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


class _RunPythonArgs(BaseModel):
    """Arguments for the ``run_python`` synthetic tool."""

    code: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description=(
            "Python code to execute. Must be a self-contained block "
            "that computes a result. Use `print()` for text output; "
            "the last expression is also captured as `result`. "
            "Available: math, decimal, statistics, fractions, "
            "datetime, itertools, json, re. No file or network access."
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
        system_prompt_override: Optional[str] = None,
        user_id: str = "demo",
    ) -> None:
        """Build a ScopedAgent for a single ``step`` of a plan.

        ``system_prompt_override`` lets a factory (typically the
        synthesizer) replace the default per-step prompt with its own.
        The default prompt is shaped around "you are running ONE step
        of a larger plan; your output feeds downstream steps" — that
        framing is wrong for a synthesizer step which IS the user's
        final output. Other agent factories should leave this None
        and rely on ``step.description`` to specialise the prompt.
        """
        self.step = step
        self.scratchpad = scratchpad
        self.registry = registry
        self.intent_flags: Dict[str, bool] = dict(intent_flags or {})
        self.recursion_limit = recursion_limit
        self.model = model
        self.user_id = user_id

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

        # 4) Build the system prompt (or use the factory-supplied override)
        self.system_prompt = (
            system_prompt_override
            if system_prompt_override is not None
            else self._build_system_prompt()
        )

        # 5) Compile the ReAct graph
        self._compiled = create_react_agent(
            model=self.model,
            tools=list(self.mcp_tools) + list(self.synthetic_tools),
            prompt=self.system_prompt,
        )

        # 6) Streaming state — populated during run()
        self._streamed_messages: List[AIMessage] = []
        self._thinking_banner_emitted = False

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
        """Construct ``get_prior_result``, ``request_assistance``, and ``run_python``.

        All close over ``self.step`` and ``self.scratchpad``. They are
        re-built per ScopedAgent instance because their behaviour
        depends on the step's ``depends_on`` set.
        """
        deps_set = set(self.step.depends_on)
        deps_list = sorted(deps_set)
        return [
            self._build_get_prior_result_tool(deps_set, deps_list),
            self._build_request_assistance_tool(),
            self._build_run_python_tool(),
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

    # ------------------------------------------------------------------
    # run_python — sandboxed numerical computation
    # ------------------------------------------------------------------
    _SAFE_BUILTINS = {
        "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
        "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format, "hex": hex,
        "int": int, "isinstance": isinstance, "iter": iter,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "oct": oct, "ord": ord, "pow": pow, "print": print,
        "range": range, "repr": repr, "reversed": reversed,
        "round": round, "set": set, "sorted": sorted, "str": str,
        "sum": sum, "tuple": tuple, "zip": zip,
    }

    _SAFE_MODULES = {
        "math": __import__("math"),
        "decimal": __import__("decimal"),
        "statistics": __import__("statistics"),
        "fractions": __import__("fractions"),
        "datetime": __import__("datetime"),
        "itertools": __import__("itertools"),
        "json": __import__("json"),
        "re": __import__("re"),
    }

    @staticmethod
    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ScopedAgent._SAFE_MODULES:
            return ScopedAgent._SAFE_MODULES[name]
        if name in ("__future__",):
            return __import__(name, *args, **kwargs)
        raise ImportError(f"Module {name!r} is not in the safe import whitelist")

    _EXEC_TIMEOUT_SECONDS = 10

    def _build_run_python_tool(self) -> BaseTool:
        step = self.step

        def _run_python(code: str) -> str:
            """Execute Python code in a sandboxed environment and return the result."""
            import signal
            import sys
            import traceback
            from io import StringIO

            local_ns: dict[str, Any] = {}
            for mod_name, mod in ScopedAgent._SAFE_MODULES.items():
                local_ns[mod_name] = mod
            safe_builtins = dict(ScopedAgent._SAFE_BUILTINS)
            safe_builtins["__import__"] = ScopedAgent._safe_import
            local_ns["__builtins__"] = safe_builtins

            stdout_buf = StringIO()
            old_stdout = sys.stdout
            old_handler = None
            error = None

            try:
                compiled = compile(code, "<run_python>", "exec")

                if hasattr(signal, "SIGALM"):
                    old_handler = signal.getsignal(signal.SIGALM)
                    signal.signal(
                        signal.SIGALM,
                        lambda _s, _f: (_ for _ in ()).throw(
                            TimeoutError("Code execution timed out")
                        ),
                    )
                    signal.alarm(ScopedAgent._EXEC_TIMEOUT_SECONDS)

                sys.stdout = stdout_buf
                try:
                    exec(compiled, local_ns, local_ns)
                except Exception:
                    error = traceback.format_exc().strip()
            except SyntaxError:
                error = traceback.format_exc().strip()
            finally:
                sys.stdout = old_stdout
                if hasattr(signal, "SIGALM"):
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALM, old_handler)

            result_val = None
            if error is None:
                for _name in ("ans", "result", "answer", "_"):
                    if _name in local_ns:
                        result_val = local_ns[_name]
                        break

            log.info(
                "step %d (%s) ran_python: %d chars, error=%s",
                step.id, step.agent, len(code), bool(error),
            )

            out: dict[str, Any] = {"success": error is None}
            stdout_text = stdout_buf.getvalue().strip()
            if stdout_text:
                out["stdout"] = stdout_text
            if result_val is not None:
                out["result"] = result_val
            if error is not None:
                out["error"] = error

            if not out.get("stdout") and not out.get("result") and not out.get("error"):
                out["stdout"] = "(no output — use print() or assign to `result`)"

            return json.dumps(out, default=str)

        return StructuredTool.from_function(
            func=_run_python,
            name="run_python",
            description=(
                "Execute Python code in a sandboxed environment for precise "
                "numerical computation. Use this when you need to calculate "
                "financial ratios, apply GAAP/IFRS formulas, compound growth "
                "rates, depreciation schedules, or any arithmetic that must be "
                "exact — LLM-generated numbers are often wrong. Available "
                "modules: math, decimal, statistics, fractions, datetime, "
                "itertools, json, re. No file or network access. Print your "
                "answer or assign it to a variable named `result`. Execution "
                "timeout: 10 seconds."
            ),
            args_schema=_RunPythonArgs,
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
        if self.user_id and self.user_id != "demo":
            user_line = f"### Authenticated user\nuser_id: `{self.user_id}` — pass this as the `user_id` argument to any portfolio tool calls.\n\n"
        else:
            user_line = ""
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
            "- You can run Python code via `run_python(code)` for precise "
            "numerical computation (financial ratios, growth rates, "
            "depreciation, compounding, etc.). **Always prefer "
            "run_python over mental arithmetic** — LLM-generated numbers "
            "are often wrong.\n"
            "- If you need help from another agent, call "
            "`request_assistance(target_agent, reason)` to record a note "
            "for the orchestrator. **Do NOT try to do another agent's "
            "job yourself.**\n\n"
            f"{user_line}"
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
    async def run(self) -> AsyncIterator[PanelEvent]:
        """Invoke the ReAct loop, streaming intermediate events, then yield a final result.

        Yields :class:`PanelEvent` dicts as the agent works:

        * ``{"type": "step_content", "persona": "<agent>", "text": "..."}``
          — streamed LLM output (token deltas).
        * ``{"type": "step_tool_call", "persona": "<agent>", "tool": "<name>", "args": {...}}``
          — agent invoked an MCP or synthetic tool.
        * ``{"type": "step_tool_result", "persona": "<agent>", "tool": "<name>", "result_preview": "..."}``
          — tool returned a result (short preview shown).
        * ``{"type": "_step_result", "result": StepResult}``
          — **terminal** event carrying the final :class:`StepResult`.

        The executor consumes this async iterator, forwarding the
        user-visible events (``step_content``, ``step_tool_call``,
        ``step_tool_result``) upstream and extracting the
        ``StepResult`` from the ``_step_result`` event to commit to
        the scratchpad.
        """
        from src.core.panel import PanelEvent as _PE  # noqa: F811 (re-import for type clarity)

        started_at = time.time()
        agent_name = self.agent_def.name if self.agent_def else self.step.agent
        user_message = (
            f"Please complete step {self.step.id}: "
            f"{self.step.description.strip()}"
        )

        try:
            async for event in self._compiled.astream_events(
                {"messages": [HumanMessage(content=user_message)]},
                version="v2",
                config={"recursion_limit": self.recursion_limit},
            ):
                async for ev in self._translate_event(event, agent_name):
                    yield ev
        except Exception as exc:
            log.exception("step %d crashed during ReAct loop", self.step.id)
            yield {
                "type": "_step_result",
                "result": StepResult(
                    step_id=self.step.id,
                    status="failed",
                    output=None,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    started_at=started_at,
                    completed_at=time.time(),
                    tools_used=[],
                ),
            }
            return

        # After the stream completes, pull the final state from the
        # compiled graph to build the StepResult. We re-invoke
        # ainvoke just for the final message extraction — but that
        # would double the LLM cost. Instead, we collected the
        # messages during streaming and extract from those.
        messages = list(self._streamed_messages)
        final_text = self._extract_final_text(messages)
        tools_used = self._collect_tools_used(messages)

        yield {
            "type": "_step_result",
            "result": StepResult(
                step_id=self.step.id,
                status="complete",
                output={"text": final_text},
                tools_used=tools_used,
                started_at=started_at,
                completed_at=time.time(),
            ),
        }

    async def _translate_event(
        self,
        event: Dict[str, Any],
        agent_name: str,
    ) -> AsyncIterator[PanelEvent]:
        """Convert a LangGraph astream_events chunk into PanelEvents.

        We track messages in ``_streamed_messages`` so we can build
        the final ``StepResult`` without re-invoking the LLM.
        """
        kind = event.get("event")
        _SYNTHETIC_TOOLS = {"get_prior_result", "request_assistance", "run_python"}

        if kind == "on_chat_model_start":
            self._thinking_banner_emitted = getattr(self, "_thinking_banner_emitted", False)
            if not self._thinking_banner_emitted:
                self._thinking_banner_emitted = True
                yield {
                    "type": "step_content",
                    "persona": agent_name,
                    "text": f"\n\n_{self.agent_def.title if self.agent_def else agent_name} is thinking…_\n\n",
                }

        elif kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", None) if chunk is not None else None
            if content and isinstance(content, str) and content.strip():
                yield {
                    "type": "step_content",
                    "persona": agent_name,
                    "text": content,
                }

        elif kind == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            if output is not None:
                msg = output if isinstance(output, AIMessage) else None
                if msg is None and hasattr(output, "generations"):
                    try:
                        msg = output.generations[0][0].message
                    except (IndexError, AttributeError):
                        pass
                if isinstance(msg, AIMessage):
                    self._streamed_messages.append(msg)

        elif kind == "on_tool_start":
            tool_name = ""
            tool_args: Dict[str, Any] = {}
            try:
                tool_name = event.get("name", "")
                tool_args = event.get("data", {}).get("input", {}) or {}
            except Exception:
                pass
            if tool_name and tool_name not in _SYNTHETIC_TOOLS:
                yield {
                    "type": "step_tool_call",
                    "persona": agent_name,
                    "tool": tool_name,
                    "args": tool_args,
                }

        elif kind == "on_tool_end":
            tool_name = event.get("name", "")
            raw_output = event.get("data", {}).get("output")
            if tool_name and tool_name not in _SYNTHETIC_TOOLS:
                preview = self._tool_result_preview(raw_output)
                yield {
                    "type": "step_tool_result",
                    "persona": agent_name,
                    "tool": tool_name,
                    "result_preview": preview,
                }

    # ------------------------------------------------------------------
    # Trajectory parsing helpers (extracted for testability)
    # ------------------------------------------------------------------
    @staticmethod
    def _tool_result_preview(raw: Any, max_len: int = 80) -> str:
        """Short human-readable preview of a tool result for the event stream."""
        if raw is None:
            return "(no output)"
        if isinstance(raw, str):
            text = raw.strip()
            if len(text) <= max_len:
                return text
            return text[: max_len - 1].rstrip() + "…"
        if isinstance(raw, dict):
            for key in ("price", "pe_ttm", "forward_pe", "holding_count", "score"):
                if key in raw:
                    return f"{key}={raw[key]}"
            items = raw.get("news") or raw.get("items")
            if isinstance(items, list):
                return f"{len(items)} item{'s' if len(items) != 1 else ''}"
        try:
            s = str(raw)
            if len(s) <= max_len:
                return s
            return s[: max_len - 1].rstrip() + "…"
        except Exception:
            return "(unreadable)"

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
