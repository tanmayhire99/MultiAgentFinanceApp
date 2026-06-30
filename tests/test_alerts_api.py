"""Tests for the /alerts API endpoints.

Calls the endpoint coroutines directly (no HTTP client, matching test_app_ops.py)
with a minimal fake Request. Auth is forced off and the alert store is isolated
per-test via FINAI_ALERTS_DB.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from src import app as app_module
from src.core import alerts


def _req():
    return types.SimpleNamespace(headers={})


@pytest.fixture
def alert_api(tmp_path, monkeypatch):
    monkeypatch.setenv("FINAI_ALERTS_DB", str(tmp_path / "a.db"))
    monkeypatch.setattr(app_module, "is_auth_enabled", lambda: False)
    uid = app_module._resolve_portfolio_user(None)   # the resolved demo user
    app_module._rate_limiter.reset(uid)
    yield uid


def test_get_alerts_returns_unread_and_list(alert_api):
    uid = alert_api
    alerts.raise_alert(alerts.Alert(uid, "concentration", "NVDA big", "d", ticker="NVDA", dedup_key="k1"))
    body = asyncio.run(app_module.get_alerts(_req()))
    assert body["user_id"] == uid
    assert body["unread_count"] == 1
    assert len(body["alerts"]) == 1 and body["alerts"][0]["title"] == "NVDA big"


def test_unread_only_filter(alert_api):
    uid = alert_api
    alerts.raise_alert(alerts.Alert(uid, "x", "t1", "d", dedup_key="k1"))
    rows = alerts.list_alerts(uid)
    alerts.mark_read(uid, rows[0]["id"])
    alerts.raise_alert(alerts.Alert(uid, "x", "t2", "d", dedup_key="k2"))
    body = asyncio.run(app_module.get_alerts(_req(), unread_only=True))
    assert body["unread_count"] == 1
    assert all(a["read"] is False for a in body["alerts"])


def test_mark_read_clears_unread(alert_api):
    uid = alert_api
    alerts.raise_alert(alerts.Alert(uid, "x", "t1", "d", dedup_key="k1"))
    alerts.raise_alert(alerts.Alert(uid, "x", "t2", "d", dedup_key="k2"))
    assert asyncio.run(app_module.get_alerts(_req()))["unread_count"] == 2
    out = asyncio.run(app_module.mark_alerts_read(app_module.MarkReadRequest(), _req()))
    assert out["unread_count"] == 0


def test_scan_endpoint_is_idempotent(alert_api):
    uid = alert_api
    first = asyncio.run(app_module.scan_alerts(_req()))
    assert isinstance(first["new_alerts"], int)
    assert first["unread_count"] >= first["new_alerts"]
    app_module._rate_limiter.reset(uid)
    second = asyncio.run(app_module.scan_alerts(_req()))
    assert second["new_alerts"] == 0   # a re-scan of unchanged holdings dedupes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
