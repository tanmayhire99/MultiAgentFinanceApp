"""Tests for event-driven portfolio alerts (src.core.alerts).

Deterministic, offline. The store is isolated per-test via FINAI_ALERTS_DB; the
pure rules are tested with plain dicts; run_scan is tested with a fake portfolio
module so no MCP/network is touched.
"""
from __future__ import annotations

import sys
import types

import pytest

from src.core import alerts
from src.core.alerts import Alert


@pytest.fixture
def alert_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FINAI_ALERTS_DB", str(tmp_path / "alerts.db"))
    yield


_HOLDINGS = [
    {"ticker": "NVDA", "weight": 0.32, "sector": "Tech"},
    {"ticker": "AAPL", "weight": 0.10, "sector": "Tech"},
    {"ticker": "KO", "weight": 0.05, "sector": "Staples"},
]


# ---------------------------------------------------------------------------
# Pure rules
# ---------------------------------------------------------------------------
def test_concentration_flags_only_oversized_positions():
    out = alerts.concentration_alerts("u1", _HOLDINGS, max_position_pct=0.20)
    assert len(out) == 1
    a = out[0]
    assert a.ticker == "NVDA" and a.kind == "concentration" and a.severity == "high"


def test_concentration_severity_medium_near_threshold():
    out = alerts.concentration_alerts("u1", [{"ticker": "X", "weight": 0.25}], max_position_pct=0.20)
    assert out[0].severity == "medium"  # 0.25 < 0.20 * 1.5


def test_price_move_only_for_held_above_threshold():
    out = alerts.price_move_alerts("u1", _HOLDINGS, {"NVDA": -6.2, "AAPL": 1.0, "TSLA": 20.0},
                                   threshold_pct=5.0)
    tickers = {a.ticker for a in out}
    assert tickers == {"NVDA"}            # AAPL below threshold, TSLA not held
    # -6.2% is between the 5% and 10% (2x) bands -> medium, and it's a down move
    assert out[0].severity == "medium" and "down" in out[0].title


def test_price_move_high_severity_on_big_move():
    out = alerts.price_move_alerts("u1", _HOLDINGS, {"NVDA": -12.0}, threshold_pct=5.0)
    assert out[0].severity == "high"


def test_price_move_direction_up():
    out = alerts.price_move_alerts("u1", _HOLDINGS, {"NVDA": 7.5}, threshold_pct=5.0)
    assert "up" in out[0].title


def test_scan_user_combines_rules():
    out = alerts.scan_user("u1", holdings=_HOLDINGS, changes={"NVDA": -6.0})
    kinds = sorted({a.kind for a in out})
    assert kinds == ["concentration", "price_move"]


# ---------------------------------------------------------------------------
# Store + dedup
# ---------------------------------------------------------------------------
def test_raise_and_list(alert_db):
    aid = alerts.raise_alert(Alert("u1", "concentration", "NVDA big", "detail", severity="high", ticker="NVDA"))
    assert isinstance(aid, int)
    rows = alerts.list_alerts("u1")
    assert len(rows) == 1 and rows[0]["title"] == "NVDA big" and rows[0]["read"] is False


def test_dedup_suppresses_repeat(alert_db):
    a = Alert("u1", "concentration", "NVDA big", "d", ticker="NVDA")
    first = alerts.raise_alert(a)
    second = alerts.raise_alert(Alert("u1", "concentration", "NVDA big", "d", ticker="NVDA"))
    assert first is not None and second is None
    assert len(alerts.list_alerts("u1")) == 1


def test_unread_count_and_mark_read(alert_db):
    alerts.raise_alert(Alert("u1", "price_move", "A", "d", ticker="A", dedup_key="k1"))
    alerts.raise_alert(Alert("u1", "price_move", "B", "d", ticker="B", dedup_key="k2"))
    assert alerts.unread_count("u1") == 2
    rows = alerts.list_alerts("u1")
    alerts.mark_read("u1", rows[0]["id"])
    assert alerts.unread_count("u1") == 1
    alerts.mark_read("u1")  # all
    assert alerts.unread_count("u1") == 0


def test_clear_and_user_isolation(alert_db):
    alerts.raise_alert(Alert("u1", "x", "t", "d", dedup_key="z"))
    alerts.raise_alert(Alert("u2", "x", "t", "d", dedup_key="z"))
    alerts.clear("u1")
    assert alerts.list_alerts("u1") == []
    assert len(alerts.list_alerts("u2")) == 1   # other user's alerts untouched


def test_scan_and_store_is_idempotent(alert_db):
    first = alerts.scan_and_store("u1", holdings=_HOLDINGS, changes={"NVDA": -6.0})
    second = alerts.scan_and_store("u1", holdings=_HOLDINGS, changes={"NVDA": -6.0})
    assert len(first) == 2 and second == []     # re-scan of same state dedupes
    assert alerts.unread_count("u1") == 2


# ---------------------------------------------------------------------------
# run_scan adapter (fake portfolio module)
# ---------------------------------------------------------------------------
def test_run_scan_uses_holdings(alert_db, monkeypatch):
    fake = types.ModuleType("src.mcp.portfolio_mcp")
    fake.get_holdings = lambda user_id="demo": {"user_id": user_id, "holdings": _HOLDINGS}
    monkeypatch.setitem(sys.modules, "src.mcp.portfolio_mcp", fake)
    ids = alerts.run_scan("u1")
    assert len(ids) == 1   # concentration on NVDA (no changes -> no price alerts)
    assert alerts.list_alerts("u1")[0]["ticker"] == "NVDA"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
