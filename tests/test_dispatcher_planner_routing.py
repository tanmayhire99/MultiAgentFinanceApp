"""Stage 3 — `/planner` prefix routing through the planner pipeline.

Coverage:

1. ``_strip_planner_prefix`` — leading prefix detection (with/without
   trailing space, case-insensitive, bare ``/planner``).
2. ``FINAI_PLANNER_PREFIX`` env-var gate — when set to a falsy value,
   the prefix is treated as ordinary text and detection is skipped.
3. ``planner_pipeline._derive_intent_flags`` — the deterministic
   classifier-intent → flag-vocab mapping.
4. End-to-end: a ``/planner`` prefixed query routes through
   ``planner_pipeline.run`` instead of the static flow it would
   normally hit.
5. Disclaimer footer still fires for finance flows even when routed
   through the planner pipeline.

Run via::

    docker exec finai-api python -m unittest tests.test_dispatcher_planner_routing -v
"""
from __future__ import annotations

import asyncio
import os
import unittest
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.dispatcher import _strip_planner_prefix, run_analysis
from src.core.flows.planner_pipeline import _derive_intent_flags


# ---------------------------------------------------------------------------
# Prefix detection
# ---------------------------------------------------------------------------
# These tests exercise the parsing logic ASSUMING the env-var gate is on
# (default for development). The demo container ships with
# FINAI_PLANNER_PREFIX=0, which is exercised separately by
# PlannerPrefixGateTests below. Explicitly set the env var here so the
# tests pass regardless of the host environment.
class StripPlannerPrefixTests(unittest.TestCase):
    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"FINAI_PLANNER_PREFIX": "1"})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_with_trailing_space(self):
        q, force = _strip_planner_prefix("/planner did Tesla deliver on FSD?")
        self.assertEqual(q, "did Tesla deliver on FSD?")
        self.assertTrue(force)

    def test_with_tab(self):
        q, force = _strip_planner_prefix("/planner\twhat is the AI hardware market?")
        self.assertEqual(q, "what is the AI hardware market?")
        self.assertTrue(force)

    def test_case_insensitive(self):
        q, force = _strip_planner_prefix("/Planner outlook for Indian IT?")
        self.assertEqual(q, "outlook for Indian IT?")
        self.assertTrue(force)

    def test_bare_prefix(self):
        q, force = _strip_planner_prefix("/planner")
        self.assertEqual(q, "")
        self.assertTrue(force)

    def test_no_prefix(self):
        q, force = _strip_planner_prefix("just a normal question about WDC")
        self.assertEqual(q, "just a normal question about WDC")
        self.assertFalse(force)

    def test_inline_match_does_not_trigger(self):
        # "/planner" must be the LEADING token; mid-message hits don't fire.
        q, force = _strip_planner_prefix("can you /planner this for me?")
        self.assertEqual(q, "can you /planner this for me?")
        self.assertFalse(force)

    def test_attached_text_does_not_trigger(self):
        # ``/plannerhello`` is NOT the prefix — needs a space or tab
        q, force = _strip_planner_prefix("/plannerhello")
        self.assertEqual(q, "/plannerhello")
        self.assertFalse(force)

    def test_leading_whitespace_then_prefix(self):
        q, force = _strip_planner_prefix("   /planner topic question")
        self.assertEqual(q, "topic question")
        self.assertTrue(force)


# ---------------------------------------------------------------------------
# Env-var gate
# ---------------------------------------------------------------------------
class PlannerPrefixGateTests(unittest.TestCase):
    """``FINAI_PLANNER_PREFIX=0`` makes the prefix invisible."""

    def test_disabled_env_returns_query_unchanged(self):
        with patch.dict(os.environ, {"FINAI_PLANNER_PREFIX": "0"}):
            q, force = _strip_planner_prefix("/planner did Tesla deliver?")
            self.assertEqual(q, "/planner did Tesla deliver?")
            self.assertFalse(force)

    def test_disabled_env_also_handles_bare(self):
        with patch.dict(os.environ, {"FINAI_PLANNER_PREFIX": "0"}):
            q, force = _strip_planner_prefix("/planner")
            self.assertEqual(q, "/planner")
            self.assertFalse(force)

    def test_falsy_values_disable(self):
        for val in ("0", "false", "no", "off", "FALSE"):
            with patch.dict(os.environ, {"FINAI_PLANNER_PREFIX": val}):
                _q, force = _strip_planner_prefix("/planner test")
                self.assertFalse(force, f"value {val!r} should disable")

    def test_truthy_values_enable(self):
        for val in ("1", "true", "yes", "on", "True"):
            with patch.dict(os.environ, {"FINAI_PLANNER_PREFIX": val}):
                _q, force = _strip_planner_prefix("/planner test")
                self.assertTrue(force, f"value {val!r} should enable")

    def test_default_unset_is_enabled(self):
        # Remove the env var to make sure the default (ON) kicks in
        env = os.environ.copy()
        env.pop("FINAI_PLANNER_PREFIX", None)
        with patch.dict(os.environ, env, clear=True):
            _q, force = _strip_planner_prefix("/planner test")
            self.assertTrue(force, "default (unset) should be enabled")


# ---------------------------------------------------------------------------
# Intent-flag derivation
# ---------------------------------------------------------------------------
class DeriveIntentFlagsTests(unittest.TestCase):
    def test_none_decision_returns_all_false(self):
        flags = _derive_intent_flags(None)
        self.assertTrue(all(v is False for v in flags.values()))
        self.assertEqual(set(flags.keys()), {
            "wants_claim_tracking",
            "wants_panel_debate",
            "wants_filings",
            "wants_portfolio_data",
            "wants_historical_news",
            "wants_deep_research",
        })

    def test_empty_decision_returns_all_false(self):
        flags = _derive_intent_flags({})
        self.assertTrue(all(v is False for v in flags.values()))

    def test_deep_stock_research_sets_4_flags(self):
        flags = _derive_intent_flags({"intent": "deep_stock_research"})
        self.assertTrue(flags["wants_claim_tracking"])
        self.assertTrue(flags["wants_filings"])
        self.assertTrue(flags["wants_historical_news"])
        self.assertTrue(flags["wants_deep_research"])
        self.assertFalse(flags["wants_portfolio_data"])
        self.assertFalse(flags["wants_panel_debate"])

    def test_portfolio_analysis_sets_portfolio_and_panel(self):
        flags = _derive_intent_flags({"intent": "portfolio_analysis"})
        self.assertTrue(flags["wants_portfolio_data"])
        self.assertTrue(flags["wants_panel_debate"])
        self.assertFalse(flags["wants_claim_tracking"])

    def test_stock_research_no_flags_by_default(self):
        flags = _derive_intent_flags({"intent": "stock_research"})
        self.assertTrue(all(v is False for v in flags.values()))

    def test_topic_research_no_flags(self):
        flags = _derive_intent_flags({"intent": "topic_research"})
        self.assertTrue(all(v is False for v in flags.values()))

    def test_want_panel_overrides_for_any_intent(self):
        # Even on stock_research, want_panel=True should set the flag
        flags = _derive_intent_flags({
            "intent": "stock_research",
            "want_panel": True,
        })
        self.assertTrue(flags["wants_panel_debate"])

    def test_unknown_intent_returns_all_false(self):
        flags = _derive_intent_flags({"intent": "made_up_intent"})
        self.assertTrue(all(v is False for v in flags.values()))


# ---------------------------------------------------------------------------
# Dispatcher routing — /planner prefix
# ---------------------------------------------------------------------------
class DispatcherRoutingTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end via ``run_analysis`` with everything but the flow mocked."""

    def setUp(self):
        # These tests exercise the /planner detection path; force the
        # env-var gate ON regardless of the host's runtime config (the
        # demo container ships with it OFF).
        self._env_patch = patch.dict(os.environ, {"FINAI_PLANNER_PREFIX": "1"})
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    async def _drain(self, query: str) -> List[Any]:
        events: List[Any] = []
        async for ev in run_analysis(query, user_id="test"):
            events.append(ev)
        return events

    async def test_planner_prefix_routes_to_planner_pipeline(self):
        # A query that would normally hit topic_research (no ticker,
        # open question) gets routed through planner_pipeline.run instead
        # because of the /planner prefix.
        fake_decision = {
            "intent": "topic_research",
            "tickers": [],
            "topic": "Indian IT",
            "want_panel": False,
            "rationale": "open-ended sector question",
        }
        called_flows: List[str] = []

        async def fake_planner_run(q, decision=None, user_id="demo"):
            called_flows.append("planner_pipeline")
            yield {"type": "text", "text": "PLANNER OK", "persona": "synthesizer"}

        async def fake_topic_run(q, decision=None, user_id="demo"):
            called_flows.append("topic_research")
            yield {"type": "text", "text": "STATIC OK", "persona": "orchestrator"}

        with patch(
            "src.core.dispatcher.classify_query",
            new=AsyncMock(return_value=fake_decision),
        ), patch(
            "src.core.dispatcher.mcp_servers.get_tools",
            new=AsyncMock(return_value=[]),
        ), patch(
            "src.core.dispatcher.install_tool_cache_wrappers"
        ), patch(
            "src.core.dispatcher.register_tools"
        ), patch(
            "src.core.dispatcher.planner_pipeline.run",
            side_effect=fake_planner_run,
        ), patch(
            "src.core.dispatcher.topic_research.run",
            side_effect=fake_topic_run,
        ):
            events = await self._drain("/planner outlook for Indian IT?")

        # planner_pipeline ran, topic_research did NOT
        self.assertIn("planner_pipeline", called_flows)
        self.assertNotIn("topic_research", called_flows)

        # The "PLANNER OK" event surfaced from the mocked flow
        text_events = [e for e in events if e.get("type") == "text"]
        self.assertTrue(
            any("PLANNER OK" in e.get("text", "") for e in text_events)
        )

    async def test_no_planner_prefix_uses_static_flow(self):
        # Without the prefix, normal routing applies. We patch
        # _FLOW_MAP directly because it's built at import time with
        # bound function references — patching the imported module's
        # attribute does not update the dict.
        fake_decision = {
            "intent": "topic_research",
            "tickers": [],
            "topic": "Indian IT",
            "want_panel": False,
            "rationale": "open-ended sector question",
        }
        called_flows: List[str] = []

        async def fake_planner_run(q, decision=None, user_id="demo"):
            called_flows.append("planner_pipeline")
            yield {"type": "text", "text": "PLANNER OK", "persona": "synthesizer"}

        async def fake_topic_run(q, decision=None, user_id="demo"):
            called_flows.append("topic_research")
            yield {"type": "text", "text": "STATIC OK", "persona": "orchestrator"}

        with patch(
            "src.core.dispatcher.classify_query",
            new=AsyncMock(return_value=fake_decision),
        ), patch(
            "src.core.dispatcher.mcp_servers.get_tools",
            new=AsyncMock(return_value=[]),
        ), patch(
            "src.core.dispatcher.install_tool_cache_wrappers"
        ), patch(
            "src.core.dispatcher.register_tools"
        ), patch(
            "src.core.dispatcher.planner_pipeline.run",
            side_effect=fake_planner_run,
        ), patch.dict(
            "src.core.dispatcher._FLOW_MAP",
            {"topic_research": fake_topic_run},
        ):
            events = await self._drain("outlook for Indian IT?")

        self.assertIn("topic_research", called_flows)
        self.assertNotIn("planner_pipeline", called_flows)

    async def test_planner_prefix_with_finance_intent_keeps_disclaimer(self):
        # When /planner is used on a deep_stock_research query, the
        # disclaimer footer must still fire because the response is
        # still finance content. The dispatcher uses the original
        # classifier intent for the disclaimer check, not the flow.
        fake_decision = {
            "intent": "deep_stock_research",
            "tickers": ["TSLA"],
            "topic": "Tesla",
            "want_panel": False,
            "rationale": "claim tracking on Tesla",
        }

        async def fake_planner_run(q, decision=None, user_id="demo"):
            yield {"type": "text", "text": "## Tesla report", "persona": "synthesizer"}

        with patch(
            "src.core.dispatcher.classify_query",
            new=AsyncMock(return_value=fake_decision),
        ), patch(
            "src.core.dispatcher.mcp_servers.get_tools",
            new=AsyncMock(return_value=[]),
        ), patch(
            "src.core.dispatcher.install_tool_cache_wrappers"
        ), patch(
            "src.core.dispatcher.register_tools"
        ), patch(
            "src.core.dispatcher.planner_pipeline.run",
            side_effect=fake_planner_run,
        ):
            events = await self._drain("/planner did Tesla deliver on FSD?")

        # The disclaimer text appears in the final stream
        full_text = "\n".join(
            e.get("text", "") for e in events if e.get("type") == "text"
        )
        self.assertIn("Disclaimer", full_text)

    async def test_planner_prefix_with_non_finance_intent_no_disclaimer(self):
        # /planner what is EBITDA? — classifier returns educational, so
        # disclaimer should NOT fire even though the planner pipeline
        # ran (educational is not a finance flow).
        fake_decision = {
            "intent": "educational",
            "tickers": [],
            "topic": "EBITDA",
            "want_panel": False,
            "rationale": "concept question",
        }

        async def fake_planner_run(q, decision=None, user_id="demo"):
            yield {"type": "text", "text": "EBITDA is ...", "persona": "synthesizer"}

        with patch(
            "src.core.dispatcher.classify_query",
            new=AsyncMock(return_value=fake_decision),
        ), patch(
            "src.core.dispatcher.mcp_servers.get_tools",
            new=AsyncMock(return_value=[]),
        ), patch(
            "src.core.dispatcher.install_tool_cache_wrappers"
        ), patch(
            "src.core.dispatcher.register_tools"
        ), patch(
            "src.core.dispatcher.planner_pipeline.run",
            side_effect=fake_planner_run,
        ):
            events = await self._drain("/planner what is EBITDA?")

        full_text = "\n".join(
            e.get("text", "") for e in events if e.get("type") == "text"
        )
        self.assertNotIn("Disclaimer", full_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
