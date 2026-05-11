"""Stage 3 — `/planner` prefix routing through the planner pipeline.

Coverage:

1. ``_strip_planner_prefix`` — leading prefix detection (with/without
   trailing space, case-insensitive, bare ``/planner``).
2. Intent-flag derivation — the deterministic classifier-intent →
   flag-vocab mapping.
3. End-to-end: a ``/planner`` prefixed query routes through
   ``planner_pipeline.run``.
4. Disclaimer footer still fires for finance flows even when routed
   through the planner pipeline.

Run via::

    docker exec finai-api python -m unittest tests.test_dispatcher_planner_routing -v
"""
from __future__ import annotations

import os
import unittest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.dispatcher import _strip_planner_prefix, run_analysis
from src.core.flows.planner_pipeline import _derive_intent_flags


# ---------------------------------------------------------------------------
# Prefix detection
# ---------------------------------------------------------------------------
class StripPlannerPrefixTests(unittest.TestCase):
    def test_with_trailing_space(self):
        q = _strip_planner_prefix("/planner did Tesla deliver on FSD?")
        self.assertEqual(q, "did Tesla deliver on FSD?")

    def test_with_tab(self):
        q = _strip_planner_prefix("/planner\twhat is the AI hardware market?")
        self.assertEqual(q, "what is the AI hardware market?")

    def test_case_insensitive(self):
        q = _strip_planner_prefix("/Planner outlook for Indian IT?")
        self.assertEqual(q, "outlook for Indian IT?")

    def test_bare_prefix(self):
        q = _strip_planner_prefix("/planner")
        self.assertEqual(q, "")

    def test_no_prefix(self):
        q = _strip_planner_prefix("just a normal question about WDC")
        self.assertEqual(q, "just a normal question about WDC")

    def test_inline_match_does_not_trigger(self):
        q = _strip_planner_prefix("can you /planner this for me?")
        self.assertEqual(q, "can you /planner this for me?")

    def test_attached_text_does_not_trigger(self):
        q = _strip_planner_prefix("/plannerhello")
        self.assertEqual(q, "/plannerhello")

    def test_leading_whitespace_then_prefix(self):
        q = _strip_planner_prefix(" /planner topic question")
        self.assertEqual(q, "topic question")


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
        flags = _derive_intent_flags({
            "intent": "stock_research",
            "want_panel": True,
        })
        self.assertTrue(flags["wants_panel_debate"])

    def test_unknown_intent_returns_all_false(self):
        flags = _derive_intent_flags({"intent": "made_up_intent"})
        self.assertTrue(all(v is False for v in flags.values()))


# ---------------------------------------------------------------------------
# Dispatcher routing — planner pipeline is default
# ---------------------------------------------------------------------------
class DispatcherRoutingTests(unittest.IsolatedAsyncioTestCase):
    """End-to-end via ``run_analysis`` with everything but the flow mocked."""

    async def _drain(self, query: str) -> List[Any]:
        events: List[Any] = []
        async for ev in run_analysis(query, user_id="test"):
            events.append(ev)
        return events

    async def test_topic_research_routes_to_planner_pipeline(self):
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
            events = await self._drain("outlook for Indian IT?")

        self.assertIn("planner_pipeline", called_flows)

        text_events = [e for e in events if e.get("type") == "text"]
        self.assertTrue(
            any("PLANNER OK" in e.get("text", "") for e in text_events)
        )

    async def test_planner_prefix_with_finance_intent_keeps_disclaimer(self):
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

        full_text = "\n".join(
            e.get("text", "") for e in events if e.get("type") == "text"
        )
        self.assertIn("Disclaimer", full_text)

    async def test_planner_prefix_with_non_finance_intent_no_disclaimer(self):
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
