"""Tests for the equity-pipeline warehouse integration (offline; mocked DB).

No real Postgres is touched — the DB layer is patched. Covers the opt-in
availability gate, ticker normalization, result mapping, and the INR→USD
quote mapper used by the Indian-stock worker.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.mcp import _warehouse
from src.mcp import indian_stock_mcp


class WarehouseAvailabilityTests(unittest.TestCase):
    def test_unavailable_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_warehouse.is_available())

    def test_unavailable_without_psycopg2(self):
        with patch.dict(os.environ, {"WAREHOUSE_DATABASE_URL": "postgresql://x"}, clear=True):
            with patch.object(_warehouse, "psycopg2", None):
                self.assertFalse(_warehouse.is_available())

    def test_available_with_env_and_psycopg2(self):
        with patch.dict(os.environ, {"WAREHOUSE_DATABASE_URL": "postgresql://x"}, clear=True):
            with patch.object(_warehouse, "psycopg2", object()):
                self.assertTrue(_warehouse.is_available())


class BareTickerTests(unittest.TestCase):
    def test_strips_suffixes_and_normalizes(self):
        self.assertEqual(_warehouse._bare("tcs.ns"), "TCS")
        self.assertEqual(_warehouse._bare("RELIANCE.BO"), "RELIANCE")
        self.assertEqual(_warehouse._bare("  infy  "), "INFY")
        self.assertEqual(_warehouse._bare("HDFCBANK"), "HDFCBANK")


class WarehouseQueryTests(unittest.TestCase):
    _ROW = {
        "ticker": "RELIANCE", "company_name": "Reliance Industries Ltd",
        "sector": "Energy", "latest_date": "2026-06-25", "latest_close": 1318.1,
        "latest_vwap": 1320.81, "latest_volume": 12694362,
        "high_52w": 1611.8, "low_52w": 1253.2, "return_30d_pct": -2.4,
    }

    def test_get_quote_none_when_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(_warehouse.get_quote("RELIANCE"))

    def test_get_quote_returns_first_row(self):
        with patch.object(_warehouse, "_query", return_value=[self._ROW]) as q:
            got = _warehouse.get_quote("RELIANCE.NS")
        self.assertEqual(got["ticker"], "RELIANCE")
        self.assertEqual(got["latest_close"], 1318.1)
        # the bare (suffix-stripped) symbol is what hits the DB
        self.assertEqual(q.call_args.args[1], ("RELIANCE",))

    def test_get_quote_none_when_no_rows(self):
        with patch.object(_warehouse, "_query", return_value=[]):
            self.assertIsNone(_warehouse.get_quote("UNKNOWN"))
        with patch.object(_warehouse, "_query", return_value=None):
            self.assertIsNone(_warehouse.get_quote("UNKNOWN"))

    def test_top_movers_passthrough(self):
        rows = [{"ticker": "ICICIBANK", "return_30d_pct": 11.92}]
        with patch.object(_warehouse, "_query", return_value=rows):
            self.assertEqual(_warehouse.get_top_movers(limit=1), rows)


class IndianWarehouseQuoteMappingTests(unittest.TestCase):
    # Round INR values chosen so /85 is exact for easy assertions.
    _ROW = {
        "ticker": "RELIANCE", "company_name": "Reliance Industries Ltd",
        "sector": "Energy", "latest_date": "2026-06-25", "latest_close": 1700.0,
        "latest_vwap": 1710.0, "latest_volume": 1_000_000,
        "high_52w": 2550.0, "low_52w": 850.0, "return_30d_pct": -2.4,
    }

    def test_maps_and_converts_inr_to_usd(self):
        with patch.object(indian_stock_mcp._warehouse, "get_quote", return_value=self._ROW):
            q = indian_stock_mcp._warehouse_quote("RELIANCE")
        self.assertEqual(q["_source"], "warehouse:equity-pipeline")
        self.assertEqual(q["currency"], "USD")
        self.assertEqual(q["native_currency"], "INR")
        self.assertEqual(q["name"], "Reliance Industries Ltd")
        self.assertEqual(q["exchange"], "NSE")
        # 1700 INR / 85 = 20.0 USD
        self.assertEqual(q["price"], 20.0)
        self.assertEqual(q["52w_high"], 30.0)   # 2550/85
        self.assertEqual(q["52w_low"], 10.0)    # 850/85
        # non-currency fields pass through untouched
        self.assertEqual(q["return_30d_pct"], -2.4)
        self.assertEqual(q["volume"], 1_000_000)
        self.assertEqual(q["as_of_date"], "2026-06-25")

    def test_returns_none_when_no_warehouse_row(self):
        with patch.object(indian_stock_mcp._warehouse, "get_quote", return_value=None):
            self.assertIsNone(indian_stock_mcp._warehouse_quote("ZZZ"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
