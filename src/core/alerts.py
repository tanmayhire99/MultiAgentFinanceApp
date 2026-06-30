"""Event-driven alerts on a user's portfolio (FROM_DEMO_TO_PRODUCT §4, the
engagement multiplier).

A demo answers a question and forgets you; a product proactively tells you when
something about *your* holdings changes. This module is the foundation:

* a durable per-user **alert store** (SQLite), with de-duplication so a repeated
  scan of an unchanged condition doesn't spam;
* **pure rule functions** (concentration, large price move) that take simple
  inputs and return :class:`Alert` objects — deterministic and unit-tested;
* a **scan** that adapts live portfolio data into those rules and persists the
  results, runnable from a scheduler/cron (``python -m src.core.alerts <user>``).

The rules are intentionally pure (no I/O) so they are trivially testable; the
side-effectful adapter (`run_scan`) fetches data and persists, and is the only
part that touches the portfolio tool.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("finai.alerts")

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "finai_alerts.db",
)
_DEDUP_WINDOW_DAYS = 7
_SEVERITIES = ("info", "medium", "high")


@dataclass
class Alert:
    user_id: str
    kind: str               # concentration | price_move | ...
    title: str
    detail: str
    severity: str = "info"  # info | medium | high
    ticker: Optional[str] = None
    dedup_key: str = ""

    def __post_init__(self):
        if self.severity not in _SEVERITIES:
            self.severity = "info"
        if not self.dedup_key:
            self.dedup_key = f"{self.kind}:{self.ticker or ''}:{self.title}"


def _path() -> str:
    return os.environ.get("FINAI_ALERTS_DB", _DEFAULT_PATH)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _connect() -> sqlite3.Connection:
    p = Path(_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=3000")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS alert (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL, kind TEXT NOT NULL, severity TEXT,
            title TEXT, detail TEXT, ticker TEXT, dedup_key TEXT,
            read INTEGER DEFAULT 0, created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_alert_user ON alert(user_id, id);
        """
    )
    return con


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def raise_alert(alert: Alert) -> Optional[int]:
    """Persist an alert, unless an equivalent one fired recently (dedup).

    Returns the new alert id, or None if it was de-duplicated. Never raises.
    """
    try:
        with _connect() as con:
            cutoff = (_now() - timedelta(days=_DEDUP_WINDOW_DAYS)).isoformat()
            dup = con.execute(
                "SELECT 1 FROM alert WHERE user_id=? AND dedup_key=? AND created_at>=? LIMIT 1",
                (alert.user_id, alert.dedup_key, cutoff),
            ).fetchone()
            if dup:
                return None
            cur = con.execute(
                """INSERT INTO alert (user_id, kind, severity, title, detail, ticker,
                                      dedup_key, read, created_at)
                   VALUES (?,?,?,?,?,?,?,0,?)""",
                (alert.user_id, alert.kind, alert.severity, alert.title, alert.detail,
                 alert.ticker, alert.dedup_key, _now().isoformat()),
            )
            return int(cur.lastrowid)
    except Exception:
        log.exception("alerts.raise_alert failed")
        return None


def list_alerts(user_id: str, *, unread_only: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        sql = "SELECT * FROM alert WHERE user_id=?"
        if unread_only:
            sql += " AND read=0"
        sql += " ORDER BY id DESC LIMIT ?"
        with _connect() as con:
            rows = con.execute(sql, (user_id, int(limit))).fetchall()
        return [dict(r) | {"read": bool(r["read"])} for r in rows]
    except Exception:
        log.exception("alerts.list_alerts failed")
        return []


def unread_count(user_id: str) -> int:
    try:
        with _connect() as con:
            return int(con.execute(
                "SELECT COUNT(*) FROM alert WHERE user_id=? AND read=0", (user_id,)).fetchone()[0])
    except Exception:
        log.exception("alerts.unread_count failed")
        return 0


def mark_read(user_id: str, alert_id: Optional[int] = None) -> None:
    """Mark one alert read (by id) or all of the user's alerts read."""
    try:
        with _connect() as con:
            if alert_id is None:
                con.execute("UPDATE alert SET read=1 WHERE user_id=?", (user_id,))
            else:
                con.execute("UPDATE alert SET read=1 WHERE user_id=? AND id=?", (user_id, alert_id))
    except Exception:
        log.exception("alerts.mark_read failed")


def clear(user_id: str) -> None:
    try:
        with _connect() as con:
            con.execute("DELETE FROM alert WHERE user_id=?", (user_id,))
    except Exception:
        log.exception("alerts.clear failed")


# ---------------------------------------------------------------------------
# Pure rules — deterministic, no I/O
# ---------------------------------------------------------------------------
def concentration_alerts(
    user_id: str, holdings: List[dict], *, max_position_pct: float = 0.20,
) -> List[Alert]:
    """Alert when a single holding exceeds ``max_position_pct`` of the portfolio.

    ``holdings`` items use the portfolio tool's shape: ``{ticker, weight, ...}``
    where ``weight`` is a fraction (0..1).
    """
    out: List[Alert] = []
    for h in holdings:
        weight = float(h.get("weight", 0) or 0)
        if weight > max_position_pct:
            ticker = h.get("ticker", "?")
            pct = round(weight * 100, 1)
            out.append(Alert(
                user_id=user_id, kind="concentration",
                severity="high" if weight > max_position_pct * 1.5 else "medium",
                ticker=ticker,
                title=f"{ticker} is {pct}% of your portfolio",
                detail=(f"{ticker} is {pct}% of your portfolio, above the "
                        f"{round(max_position_pct * 100)}% single-position guideline. "
                        "Consider whether this concentration matches your risk tolerance."),
                dedup_key=f"concentration:{ticker}",
            ))
    return out


def price_move_alerts(
    user_id: str, holdings: List[dict], changes: Dict[str, float], *,
    threshold_pct: float = 5.0,
) -> List[Alert]:
    """Alert when a held ticker moved at least ``threshold_pct`` (abs) on the day.

    ``changes`` maps ticker -> percent change (e.g. ``{"NVDA": -6.2}``).
    """
    out: List[Alert] = []
    held = {h.get("ticker") for h in holdings}
    for ticker, change in changes.items():
        if ticker in held and abs(float(change)) >= threshold_pct:
            direction = "up" if change > 0 else "down"
            out.append(Alert(
                user_id=user_id, kind="price_move",
                severity="high" if abs(change) >= threshold_pct * 2 else "medium",
                ticker=ticker,
                title=f"{ticker} moved {direction} {abs(round(change, 1))}%",
                detail=(f"{ticker}, which you hold, moved {direction} "
                        f"{abs(round(change, 1))}% — above your {threshold_pct}% alert threshold."),
                dedup_key=f"price_move:{ticker}:{round(change)}",
            ))
    return out


def scan_user(
    user_id: str, *, holdings: List[dict], changes: Optional[Dict[str, float]] = None,
    max_position_pct: float = 0.20, threshold_pct: float = 5.0,
) -> List[Alert]:
    """Run every applicable rule and return the alerts (does NOT persist)."""
    alerts = concentration_alerts(user_id, holdings, max_position_pct=max_position_pct)
    if changes:
        alerts += price_move_alerts(user_id, holdings, changes, threshold_pct=threshold_pct)
    return alerts


def scan_and_store(user_id: str, **kwargs) -> List[int]:
    """Run :func:`scan_user` and persist each alert (deduped). Returns new ids."""
    ids = []
    for alert in scan_user(user_id, **kwargs):
        new_id = raise_alert(alert)
        if new_id is not None:
            ids.append(new_id)
    return ids


def live_quote_change(ticker: str, holding: Optional[dict] = None) -> Optional[float]:
    """Best-effort 1-day percent change for a held ticker (the live feed).

    Routes by venue: NSE / India holdings use the equity-pipeline warehouse
    (computed from the last two closes — currency-invariant), everything else
    uses the US live quote's ``change_pct_1d``. Returns None on any miss so a
    single unfetchable ticker never breaks the scan. Imports are lazy to keep
    ``alerts`` importable without the MCP stack.
    """
    holding = holding or {}
    exchange = str(holding.get("exchange") or "").upper()
    country = str(holding.get("country") or "").upper()
    try:
        if exchange == "NSE" or country in ("INDIA", "IN") or ticker.upper().endswith(".NS"):
            from src.mcp import _warehouse

            hist = _warehouse.get_history(ticker)
            if hist and len(hist) >= 2:
                prev, last = hist[-2].get("close"), hist[-1].get("close")
                if prev and last:
                    return round((float(last) / float(prev) - 1) * 100, 2)
            return None
        from src.mcp import us_stock_mcp

        quote = us_stock_mcp.get_quote(ticker)
        pct = quote.get("change_pct_1d") if isinstance(quote, dict) else None
        return round(float(pct), 2) if pct is not None else None
    except Exception:
        log.exception("live_quote_change failed for %s", ticker)
        return None


def fetch_day_changes(holdings: List[dict], quote_fn) -> Dict[str, float]:
    """Build a ticker -> 1d-percent-change map by calling ``quote_fn`` per holding.

    Pure orchestration over the ``quote_fn`` seam (tests inject a fake); tickers
    whose change can't be fetched are simply omitted.
    """
    changes: Dict[str, float] = {}
    for h in holdings:
        ticker = h.get("ticker")
        if not ticker:
            continue
        try:
            pct = quote_fn(ticker, h)
        except Exception:
            log.exception("quote_fn raised for %s", ticker)
            pct = None
        if pct is not None:
            changes[ticker] = float(pct)
    return changes


def run_scan(user_id: str, changes: Optional[Dict[str, float]] = None, quote_fn=None) -> List[int]:
    """Fetch the user's live holdings and scan them. Side-effectful adapter.

    Pulls holdings from the portfolio MCP tool. Day moves come from, in order:
    an explicit ``changes`` map; else a ``quote_fn`` live feed (e.g.
    :func:`live_quote_change`) fetched per holding; else nothing (price rule
    skipped). Never raises into the caller.
    """
    try:
        from src.mcp.portfolio_mcp import get_holdings

        data = get_holdings(user_id)
        holdings = data.get("holdings", []) if isinstance(data, dict) else []
    except Exception:
        log.exception("run_scan: could not load holdings for %s", user_id)
        return []
    if changes is None and quote_fn is not None:
        changes = fetch_day_changes(holdings, quote_fn)
    return scan_and_store(user_id, holdings=holdings, changes=changes)


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Scan a user's portfolio for alerts.")
    p.add_argument("user_id")
    p.add_argument("--no-live", action="store_true",
                   help="skip the live day-move feed (concentration checks only)")
    a = p.parse_args(argv)
    ids = run_scan(a.user_id, quote_fn=None if a.no_live else live_quote_change)
    for row in list_alerts(a.user_id, limit=20):
        flag = " " if row["read"] else "*"
        print(f"{flag} [{row['severity']:6s}] {row['title']}")
    print(f"\n{len(ids)} new alert(s); {unread_count(a.user_id)} unread total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "Alert", "raise_alert", "list_alerts", "unread_count", "mark_read", "clear",
    "concentration_alerts", "price_move_alerts", "scan_user", "scan_and_store", "run_scan",
    "fetch_day_changes", "live_quote_change",
]
