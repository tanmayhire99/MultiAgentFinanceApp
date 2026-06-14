"""End-to-end tests for src.core.pipeline — Slice Stage 2c.

These tests stitch the planner + executor together through
:func:`src.core.pipeline.run_pipeline`. The LLM and the ScopedAgent
factory are mocked so the test runs in <50ms with no network and no
GPU; what we're verifying is the **wiring** between the pieces:

* planner-failure path → ``error`` event yielded, no execute attempt
* happy path → planner ran, executor ran every step, synthesizer
  output surfaces as a ``text`` event with persona=synthesizer
* synth-step-failed path → executor ran, but the pipeline emits an
  ``error`` event surfacing the failure
* no-synth path → pipeline emits an ``error`` event explaining the
  malformed plan

Mocking strategy
----------------
``src.core.planner.build_chat_model`` → fake chat model (controlled
``ainvoke`` returns canned ``AIMessage`` content).

``src.core.executor.build_scoped_agent_for_step`` → fake ScopedAgent
whose ``run()`` returns a per-test :class:`StepResult`.

Run via::

    docker exec finai-api python -m unittest tests.test_pipeline_e2e -v
"""
from __future__ import annotations

import asyncio
import json
import time
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.core.pipeline import run_pipeline
from src.core.types import KNOWN_INTENT_FLAGS, PlanStep, StepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PLANNER_PATCH = "src.core.planner.build_chat_model"
_EXECUTOR_PATCH = "src.core.executor.build_scoped_agent_for_step"


def _flags(**overrides: bool) -> Dict[str, bool]:
    base = {f: False for f in KNOWN_INTENT_FLAGS}
    base.update(overrides)
    return base


def _topic_research_plan_json() -> str:
    """A 2-step plan: research_agent → synthesizer. Valid against registry."""
    return json.dumps({
        "schema_version": "1.0",
        "goal": "Topic research on Indian IT outlook",
        "rationale": "research + synthesize report on Indian IT sector",
        "estimated_complexity": "medium",
        "steps": [
            {
                "id": 1,
                "description": "Pull recent news on Indian IT sector",
                "agent": "research_agent",
                "tool_subset": [
                    "research__search_news",
                    "research__search_web",
                ],
                "depends_on": [],
                "max_tool_calls": 6,
            },
            {
                "id": 2,
                "description": "Synthesize a report",
                "agent": "synthesizer",
                "tool_subset": [],
                "depends_on": [1],
                "max_tool_calls": 0,
            },
        ],
    })


def _plan_without_synthesizer_json() -> str:
    """Malformed (semantically) plan — passes schema but has no synth."""
    return json.dumps({
        "schema_version": "1.0",
        "goal": "Topic research on Indian IT outlook",
        "rationale": "research only — bug: no synth step",
        "estimated_complexity": "medium",
        "steps": [
            {
                "id": 1,
                "description": "Pull recent news on Indian IT sector",
                "agent": "research_agent",
                "tool_subset": ["research__search_news"],
                "depends_on": [],
                "max_tool_calls": 6,
            },
        ],
    })


def _panel_plan_json() -> str:
    """A 2-step panel plan: portfolio_agent → panel_agent (TERMINAL).

    No synthesizer — the panel writes its own debate + closing brief.
    Valid against the registry only when intent_flags has
    wants_panel_debate=True (panel_agent's policy gate).
    """
    return json.dumps({
        "schema_version": "1.0",
        "goal": "Run the investor panel over the user's portfolio",
        "rationale": "panel debate requested; portfolio first, then panel_agent (terminal)",
        "estimated_complexity": "high",
        "steps": [
            {
                "id": 1,
                "description": "Fetch holdings + concentration risks",
                "agent": "portfolio_agent",
                "tool_subset": [
                    "portfolio__get_holdings",
                    "portfolio__get_concentration_risks",
                ],
                "depends_on": [],
                "max_tool_calls": 5,
            },
            {
                "id": 2,
                "description": "Run the investor panel debate over the portfolio",
                "agent": "panel_agent",
                "tool_subset": [],
                "depends_on": [1],
                "max_tool_calls": 0,
            },
        ],
    })


def _fake_llm_returning(*responses: str) -> MagicMock:
    """A chat-model factory's return value — ainvoke yields canned content."""
    chat = MagicMock()
    chat.ainvoke = AsyncMock(
        side_effect=[AIMessage(content=r) for r in responses]
    )
    # Some planners call .bind_tools or .with_structured_output;
    # neither is currently used, but keep the mock graceful.
    chat.bind_tools = MagicMock(return_value=chat)
    chat.with_structured_output = MagicMock(return_value=chat)
    return chat


def _make_fake_agent(*, step: PlanStep, result: StepResult) -> MagicMock:
    """A MagicMock whose async run() returns the given StepResult."""
    fake = MagicMock()
    fake.step = step

    async def _run() -> StepResult:
        return result

    fake.run = _run
    return fake


def _make_fake_streaming_panel_agent(*, step: PlanStep, result: StepResult):
    """A panel-style agent whose ``run_streaming`` yields live debate
    events then the ``_step_result`` sentinel.

    Must be a *real class* (not a MagicMock) so the executor's
    ``inspect.isasyncgenfunction(type(agent).run_streaming)`` check sees
    a genuine async-generator function and routes through the streaming
    path — mirroring production ``PanelScopedAgent``.
    """

    class _FakePanelAgent:
        def __init__(self) -> None:
            self.step = step

        async def run_streaming(self):
            yield {"type": "header", "text": "## Investor Panel Debate\n\n",
                   "persona": "moderator"}
            yield {"type": "text", "text": "Buffett: cautious on valuation.\n",
                   "persona": "buffett"}
            yield {"type": "text", "text": "Wood: bullish on the innovation cycle.\n",
                   "persona": "wood"}
            yield {"type": "header", "text": "\n## Closing Brief\n\n",
                   "persona": "moderator"}
            yield {"type": "text", "text": "The panel diverged on valuation.\n",
                   "persona": "moderator"}
            yield {"type": "_step_result", "result": result}

    return _FakePanelAgent()


def _ok_result(step_id: int, *, text: str = "ok") -> StepResult:
    return StepResult(
        step_id=step_id,
        status="complete",
        output={"text": text},
        tools_used=[],
        started_at=time.time(),
        completed_at=time.time() + 0.01,
    )


def _failed_result(step_id: int, *, error: str = "boom") -> StepResult:
    return StepResult(
        step_id=step_id,
        status="failed",
        output=None,
        error=error,
        error_type="TestError",
        started_at=time.time(),
        completed_at=time.time() + 0.01,
    )


def _drain_pipeline(
    query: str,
    *,
    intent_flags: Dict[str, bool],
) -> List[Any]:
    """Run the pipeline and collect every yielded event."""

    async def _go() -> List[Any]:
        events: List[Any] = []
        async for ev in run_pipeline(
            query,
            intent_flags=intent_flags,
            all_mcp_tools=[],  # unused by mocked factory
        ):
            events.append(ev)
        return events

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Happy path — planner succeeds, executor runs, synth output surfaces
# ---------------------------------------------------------------------------
class HappyPathTests(unittest.TestCase):
    def test_full_pipeline_emits_synth_text_event(self):
        synth_text = "## Outlook for Indian IT\n\nThe sector is mixed..."
        fake_llm = _fake_llm_returning(_topic_research_plan_json())

        def fake_factory(*, step: PlanStep, **_kwargs):
            text = synth_text if step.agent == "synthesizer" else "raw findings"
            return _make_fake_agent(
                step=step,
                result=_ok_result(step.id, text=text),
            )

        with patch(_PLANNER_PATCH, return_value=fake_llm), \
                patch(_EXECUTOR_PATCH, side_effect=fake_factory):
            events = _drain_pipeline(
                "Outlook for Indian IT?", intent_flags=_flags()
            )

        # The synth output ends up as a single 'text' event
        text_events = [e for e in events if e.get("type") == "text"]
        self.assertEqual(len(text_events), 1)
        self.assertEqual(text_events[0]["text"], synth_text)
        self.assertEqual(text_events[0]["persona"], "synthesizer")

        # Pipeline-level status events appear at start and end
        statuses = [e["text"] for e in events if e.get("type") == "_status"]
        self.assertTrue(any("Planning" in s for s in statuses))
        self.assertTrue(any("Plan ready" in s for s in statuses))
        self.assertTrue(any("Pipeline complete" in s for s in statuses))

        # No error events on the happy path
        errors = [e for e in events if e.get("type") == "error"]
        self.assertEqual(errors, [])


# ---------------------------------------------------------------------------
# Planner failure — pipeline aborts before executor runs
# ---------------------------------------------------------------------------
class PlannerFailureTests(unittest.TestCase):
    def test_planner_garbage_yields_error_event_no_executor_call(self):
        fake_llm = _fake_llm_returning(
            "garbage 1", "garbage 2", "garbage 3"
        )

        # The factory should NEVER be called if the planner failed
        factory = MagicMock(
            side_effect=AssertionError("executor must not run after planner failure")
        )

        with patch(_PLANNER_PATCH, return_value=fake_llm), \
                patch(_EXECUTOR_PATCH, side_effect=factory):
            events = _drain_pipeline(
                "Outlook for Indian IT?", intent_flags=_flags()
            )

        # We expect a single 'error' event with the planner-failure message
        errors = [e for e in events if e.get("type") == "error"]
        self.assertGreaterEqual(len(errors), 1)
        self.assertTrue(
            any("Planner failed" in e["text"] for e in errors),
            f"Expected 'Planner failed' in error events, got: {errors}",
        )

        # No text events (no synth ran)
        text_events = [e for e in events if e.get("type") == "text"]
        self.assertEqual(text_events, [])

        # The factory was never called
        self.assertEqual(factory.call_count, 0)


# ---------------------------------------------------------------------------
# Synthesizer-step failure — executor finished but synth failed
# ---------------------------------------------------------------------------
class SynthFailureTests(unittest.TestCase):
    def test_synth_failure_yields_error_event_with_partial_handling(self):
        fake_llm = _fake_llm_returning(_topic_research_plan_json())

        def fake_factory(*, step: PlanStep, **_kwargs):
            if step.agent == "synthesizer":
                return _make_fake_agent(
                    step=step,
                    result=_failed_result(step.id, error="synth blew up"),
                )
            return _make_fake_agent(
                step=step, result=_ok_result(step.id, text="raw findings"),
            )

        with patch(_PLANNER_PATCH, return_value=fake_llm), \
                patch(_EXECUTOR_PATCH, side_effect=fake_factory):
            events = _drain_pipeline(
                "Outlook for Indian IT?", intent_flags=_flags()
            )

        errors = [e for e in events if e.get("type") == "error"]
        self.assertGreaterEqual(len(errors), 1)
        # Pipeline surfaces the synth failure with status reason
        self.assertTrue(
            any(
                "Synthesizer step finished with status=failed" in e["text"]
                for e in errors
            ),
            f"Expected synth-failed error, got: {errors}",
        )


# ---------------------------------------------------------------------------
# Plan without synthesizer — pipeline emits a structured error
# ---------------------------------------------------------------------------
class NoSynthStepTests(unittest.TestCase):
    def test_plan_without_synthesizer_yields_error_event(self):
        fake_llm = _fake_llm_returning(_plan_without_synthesizer_json())

        def fake_factory(*, step: PlanStep, **_kwargs):
            return _make_fake_agent(
                step=step, result=_ok_result(step.id, text="raw findings"),
            )

        with patch(_PLANNER_PATCH, return_value=fake_llm), \
                patch(_EXECUTOR_PATCH, side_effect=fake_factory):
            events = _drain_pipeline(
                "Outlook for Indian IT?", intent_flags=_flags()
            )

        errors = [e for e in events if e.get("type") == "error"]
        self.assertGreaterEqual(len(errors), 1)
        self.assertTrue(
            any(
                "no synthesizer or panel_agent step" in e["text"].lower()
                for e in errors
            ),
            f"Expected 'no synthesizer or panel_agent step' error, got: {errors}",
        )


# ---------------------------------------------------------------------------
# Panel-terminal path — Day 10b: panel_agent is the terminal report step,
# streams its debate live, and the pipeline does NOT re-emit it.
# ---------------------------------------------------------------------------
class PanelTerminalTests(unittest.TestCase):
    def test_panel_terminal_streams_live_and_does_not_double_emit(self):
        fake_llm = _fake_llm_returning(_panel_plan_json())

        # The panel's StepResult.output["text"] holds the full transcript
        # (committed to the scratchpad), but the pipeline must NOT re-emit
        # it — it already streamed live during execution.
        panel_result = _ok_result(
            2, text="## Investor Panel Debate\n\n(full buffered transcript)\n",
        )

        def fake_factory(*, step: PlanStep, **_kwargs):
            if step.agent == "panel_agent":
                return _make_fake_streaming_panel_agent(
                    step=step, result=panel_result,
                )
            # portfolio_agent → ordinary buffered agent
            return _make_fake_agent(
                step=step, result=_ok_result(step.id, text="holdings + risks"),
            )

        with patch(_PLANNER_PATCH, return_value=fake_llm), \
                patch(_EXECUTOR_PATCH, side_effect=fake_factory):
            events = _drain_pipeline(
                "What would the investor panel say about my portfolio?",
                intent_flags=_flags(
                    wants_portfolio_data=True, wants_panel_debate=True,
                ),
            )

        # Renderable stream = text + header events (section titles like
        # "## Investor Panel Debate" arrive as header events).
        stream_events = [
            e for e in events if e.get("type") in ("text", "header")
        ]
        joined = "".join(e.get("text", "") for e in stream_events)

        # The debate streamed live through executor → pipeline
        self.assertIn("Investor Panel Debate", joined)
        self.assertIn("Buffett", joined)
        self.assertIn("Wood", joined)
        self.assertIn("Closing Brief", joined)

        # Phase 5 must NOT re-emit the buffered transcript: no
        # synthesizer-tagged report event, and the buffered marker text
        # never appears.
        synth_events = [
            e for e in events
            if e.get("type") == "text" and e.get("persona") == "synthesizer"
        ]
        self.assertEqual(synth_events, [])
        self.assertNotIn("full buffered transcript", joined)

        # Pipeline finished cleanly, no error events
        statuses = [e["text"] for e in events if e.get("type") == "_status"]
        self.assertTrue(any("Pipeline complete" in s for s in statuses))
        errors = [e for e in events if e.get("type") == "error"]
        self.assertEqual(errors, [])

    def test_panel_failure_surfaces_error(self):
        # If the panel step fails, the pipeline surfaces a clean error
        # (and does not crash on the panel-terminal branch).
        fake_llm = _fake_llm_returning(_panel_plan_json())

        def fake_factory(*, step: PlanStep, **_kwargs):
            if step.agent == "panel_agent":
                # Streaming agent that yields a FAILED result sentinel
                return _make_fake_streaming_panel_agent(
                    step=step,
                    result=_failed_result(step.id, error="debate boom"),
                )
            return _make_fake_agent(
                step=step, result=_ok_result(step.id, text="holdings"),
            )

        with patch(_PLANNER_PATCH, return_value=fake_llm), \
                patch(_EXECUTOR_PATCH, side_effect=fake_factory):
            events = _drain_pipeline(
                "Investor panel on my portfolio please",
                intent_flags=_flags(
                    wants_portfolio_data=True, wants_panel_debate=True,
                ),
            )

        errors = [e for e in events if e.get("type") == "error"]
        self.assertGreaterEqual(len(errors), 1)
        self.assertTrue(
            any("status=failed" in e["text"] for e in errors),
            f"Expected a panel-failed error, got: {errors}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
