"""Tests for the opt-in quant (automated-trading) integration — offline.

The quant MCP server runs in the sibling repo's own Python 3.14 runtime; here
we only test FinAI's wiring: the server-config builder, the env gate, and the
QUANT_AGENT catalog entry. No subprocess is spawned and the sibling repo is not
required.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config import mcp_servers
from src.core.agents.registry import QUANT_AGENT, REGISTRY, _quant_enabled


class QuantServerConfigTests(unittest.TestCase):
    def test_builds_stdio_entry(self):
        cfg = mcp_servers.quant_server_config("/venv/bin/python", "/repo/automated-trading")
        self.assertEqual(cfg["command"], "/venv/bin/python")
        self.assertEqual(cfg["args"], ["quant_mcp.py"])
        self.assertEqual(cfg["transport"], "stdio")
        self.assertEqual(cfg["cwd"], "/repo/automated-trading")
        self.assertIn("env", cfg)


class QuantGateTests(unittest.TestCase):
    def test_disabled_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_quant_enabled())

    def test_enabled_with_both_env(self):
        with patch.dict(os.environ, {"QUANT_MCP_PYTHON": "/py", "QUANT_MCP_CWD": "/repo"}, clear=True):
            self.assertTrue(_quant_enabled())

    def test_disabled_with_only_one(self):
        with patch.dict(os.environ, {"QUANT_MCP_PYTHON": "/py"}, clear=True):
            self.assertFalse(_quant_enabled())


class QuantAgentTests(unittest.TestCase):
    def test_agent_shape(self):
        self.assertEqual(QUANT_AGENT.name, "quant_agent")
        self.assertEqual(
            set(QUANT_AGENT.tools),
            {"quant__list_strategies", "quant__backtest_strategy"},
        )
        self.assertIsNone(QUANT_AGENT.policy_gate)  # ungated, read-only

    def test_disabled_by_default_not_in_registry(self):
        # CI/tests don't set the env, so the quant agent must stay out and the
        # base registry stays at its canonical size.
        self.assertNotIn("quant_agent", {a.name for a in REGISTRY})

    def test_tools_unique_vs_base_registry(self):
        base = set()
        for a in REGISTRY:
            base.update(a.tools)
        self.assertEqual(base & set(QUANT_AGENT.tools), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
