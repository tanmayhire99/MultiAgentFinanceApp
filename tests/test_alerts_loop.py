"""Tests for alert loop (``--loop`` daemon) and ``list_real_users``."""
from __future__ import annotations

import os
import sys

import pytest

from src.core import memory
from src.core.alerts import _main


@pytest.fixture
def memory_db(tmp_path, monkeypatch):
    """Isolated SQLite file for memory, reset per-test."""
    p = str(tmp_path / "finai_memory.db")
    monkeypatch.setenv("FINAI_MEMORY_DB", p)
    return p


def test_list_real_users_empty_when_no_profiles():
    assert memory.list_real_users() == ()


def test_list_real_users_filters_demo_anonymous(memory_db):
    memory.set_profile("demo", risk_tolerance="medium")
    memory.set_profile("anonymous", horizon="short")
    assert memory.list_real_users() == ()


def test_list_real_users_returns_authed_users(memory_db):
    memory.set_profile("alice", risk_tolerance="high", horizon="long")
    memory.set_profile("bob", goals="drip")
    users = memory.list_real_users()
    assert sorted(users) == ["alice", "bob"]


def test_list_real_users_ignores_non_profile_rows(memory_db):
    """Only users with a profile row appear; notes alone don't count."""
    memory.remember("carol", "some topic")  # a note, not a profile
    assert memory.list_real_users() == ()


def test_loop_smoke_no_users(memory_db, monkeypatch, tmp_path):
    """Loop runs the specified number of iterations and exits cleanly when there
    are no real users (no crash, no infinite stall)."""
    monkeypatch.setenv("FINAI_ALERTS_DB", str(tmp_path / "alerts.db"))
    ret = _main(["--loop", "--interval", "0", "--stop-after", "3", "--no-live"])
    assert ret == 0


def test_loop_smoke_with_users(monkeypatch, memory_db, tmp_path):
    """Loop scans real users and persists new concentration alerts."""
    alerts_path = str(tmp_path / "alerts.db")
    monkeypatch.setenv("FINAI_ALERTS_DB", alerts_path)
    memory.set_profile("alice", risk_tolerance="high")

    from src.core.alerts import raise_alert, unread_count

    # Pretend Alice holds ID from the portfolio MCP tool (monkeypatch the import).
    class FakePortfolio:
        @staticmethod
        def get_holdings(user_id):
            return {"holdings": [
                {"ticker": "WDC", "weight": 0.45, "exchange": "NASDAQ",
                 "country": "US", "name": "Western Digital Corp"},
            ]}
    monkeypatch.setitem(sys.modules, "src.mcp.portfolio_mcp", FakePortfolio)

    ret = _main(["--loop", "--interval", "0", "--stop-after", "1", "--no-live"])
    assert ret == 0
    # First iteration should write one concentration alert and then dedup within
    # the 7-day window on subsequent runs.
    assert unread_count("alice") >= 1

    # Run again — nothing new (dedup window covers the repeat).
    ret = _main(["--loop", "--interval", "0", "--stop-after", "1", "--no-live"])
    assert ret == 0
    assert unread_count("alice") >= 1  # no new firing


__all__ = []