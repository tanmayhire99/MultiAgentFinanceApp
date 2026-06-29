"""Read-only client for the sibling `equity-pipeline` PostgreSQL warehouse.

This is the integration seam that lets the Indian-stock MCP worker back its
quote / history with **exchange-sourced NSE EOD data** (official bhavcopy,
warehoused by the `equity-pipeline` project) instead of yfinance's delayed,
third-party feed. It reads ONLY the warehouse's stable analytics layer
(materialized snapshot / views / function) — never the physical tables — so
it mirrors `equity-pipeline/consumer_api.py` and stays decoupled from the
schema.

**Opt-in + fail-safe.** Every function returns ``None`` (and the worker falls
back to its existing source) when:

* ``WAREHOUSE_DATABASE_URL`` is unset (integration disabled), or
* ``psycopg2`` isn't installed, or
* the DB is unreachable / the ticker isn't in the warehouse universe.

So this never hard-couples FinAI to the warehouse being up.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from decimal import Decimal
from typing import Any, Dict, List, Optional

log = logging.getLogger("finai.warehouse")

try:  # psycopg2 is optional — absence simply disables the integration
    import psycopg2
except ImportError:  # pragma: no cover - exercised only without the dep
    psycopg2 = None  # type: ignore[assignment]


_ENV_VAR = "WAREHOUSE_DATABASE_URL"
_conn = None  # cached connection (warehouse is read-only / low-churn)


def is_available() -> bool:
    """True when the integration is configured (env set + psycopg2 present)."""
    return psycopg2 is not None and bool(os.getenv(_ENV_VAR, "").strip())


def _get_conn():
    """Return a live connection, or None. Reconnects if the cache went stale."""
    global _conn
    if not is_available():
        return None
    if _conn is not None and getattr(_conn, "closed", 1) == 0:
        return _conn
    try:
        _conn = psycopg2.connect(os.environ[_ENV_VAR])
        _conn.autocommit = True  # read-only; avoid idle-in-transaction holds
        return _conn
    except Exception as e:  # pragma: no cover - network/credentials
        log.warning("warehouse: connection failed: %s", e)
        _conn = None
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def _query(sql: str, params: Optional[tuple]) -> Optional[List[Dict[str, Any]]]:
    """Run a read query, returning a list of dicts, or None on any failure."""
    conn = _get_conn()
    if conn is None:
        return None
    global _conn
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [
                {c: _jsonable(v) for c, v in zip(cols, row)}
                for row in cur.fetchall()
            ]
    except Exception as e:  # pragma: no cover - drop the stale conn, fall back
        log.warning("warehouse: query failed (%s); falling back", e)
        try:
            if _conn is not None:
                _conn.close()
        except Exception:
            pass
        _conn = None
        return None


def _bare(ticker: str) -> str:
    """Bare NSE symbol the warehouse stores (strip yfinance .NS/.BO suffix)."""
    t = (ticker or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if t.endswith(suffix):
            return t[: -len(suffix)]
    return t


# ---------------------------------------------------------------------------
# Public read API — mirrors equity-pipeline/consumer_api.py
# ---------------------------------------------------------------------------
def get_quote(ticker: str) -> Optional[Dict[str, Any]]:
    """Latest NSE EOD snapshot for a ticker (INR), or None.

    Columns: ticker, company_name, sector, latest_date, latest_close,
    latest_vwap, latest_volume, high_52w, low_52w, return_30d_pct.
    """
    rows = _query(
        """
        SELECT ticker, company_name, sector, latest_date, latest_close,
               latest_vwap, latest_volume, high_52w, low_52w, return_30d_pct
        FROM mv_stock_snapshot
        WHERE ticker = %s
        """,
        (_bare(ticker),),
    )
    return rows[0] if rows else None


def get_history(
    ticker: str,
    from_date: Optional[dt.date] = None,
    to_date: Optional[dt.date] = None,
) -> Optional[List[Dict[str, Any]]]:
    """OHLCV + VWAP + 7d/30d moving averages over a date range (INR), or None."""
    to_date = to_date or dt.date.today()
    from_date = from_date or (to_date - dt.timedelta(days=90))
    return _query(
        """
        SELECT trade_date, open, high, low, close, vwap, volume, ma_7d, ma_30d
        FROM fn_ticker_report(%s, %s, %s)
        """,
        (_bare(ticker), from_date, to_date),
    )


def get_top_movers(limit: int = 10) -> Optional[List[Dict[str, Any]]]:
    """Top gainers over the trailing 30 days across the warehouse universe."""
    return _query(
        """
        SELECT ticker, company_name, sector, close_30d_ago, close_today,
               return_30d_pct
        FROM vw_top_movers_30d
        ORDER BY return_30d_pct DESC
        LIMIT %s
        """,
        (limit,),
    )


def get_sector_performance() -> Optional[List[Dict[str, Any]]]:
    """Average daily return and volume by sector and ISO week."""
    return _query(
        """
        SELECT sector, year, week_number, avg_daily_return_pct,
               total_volume, stocks_in_sector
        FROM vw_sector_weekly_performance
        """,
        None,
    )


__all__ = [
    "is_available",
    "get_quote",
    "get_history",
    "get_top_movers",
    "get_sector_performance",
]
