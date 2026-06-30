"""Indian Stock (NSE / BSE) MCP worker server.

Mirror of :mod:`us_stock_mcp` but backed by the Indian universe. Live
quantitative data comes from :mod:`yfinance` with the ``.NS`` suffix
(so ``"TCS"`` -> ``TCS.NS``). Qualitative narrative fields stay with
the ``indian_stocks`` fixture.

If you want *real-time* NSE/BSE data (not yfinance's 15-min-delayed
feed), swap :func:`src.mcp._live.fetch_and_map` for an
``nsepython``- or ``bsedata``-backed equivalent inside this module;
the tool signatures and response schemas are designed to be stable
across that swap.

Run as::

    python -m src.mcp.indian_stock_mcp
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

from fastmcp import FastMCP

from . import _live
from . import _warehouse
from ._fixtures import load_fixture, lookup


_FIXTURES: Dict[str, Any] = load_fixture("indian_stocks")

# yfinance Indian symbols are suffixed with ``.NS`` (NSE) or ``.BO`` (BSE).
# We default to NSE for listed names; swap to ``.BO`` if a name is BSE-only.
_YFINANCE_SUFFIX = ".NS"


mcp = FastMCP(
    name="indian_stock",
    instructions=(
        "Provides Indian (NSE) equity data - quotes, fundamentals, "
        "growth, defensive, and moat signals. Currency-denominated "
        "fields are returned in **USD** (converted from yfinance's "
        "native INR via a constant FX rate) so every agent in the "
        "pipeline speaks one currency. Quant fields come live from "
        "yfinance with the ``.NS`` suffix (1-hour cached); qualitative "
        "narrative fields come from curated fixtures."
    ),
)


def _fixture_entry(ticker: str) -> Dict[str, Any]:
    try:
        return lookup(_FIXTURES, ticker)
    except KeyError:
        return {}


def _available_tickers() -> List[str]:
    return sorted(k for k in _FIXTURES.keys() if not k.startswith("_"))


def _live_map(ticker: str, mapper) -> Dict[str, Any] | None:
    return _live.fetch_and_map(ticker, mapper, suffix=_YFINANCE_SUFFIX)


def _inr_to_usd(value: Any) -> Any:
    """Convert an INR figure to USD via the constant FX rate, preserving
    ``None`` / non-numeric values untouched."""
    try:
        return round(float(value) * _live.USD_PER_INR, 4) if value is not None else None
    except (TypeError, ValueError):
        return None


def _warehouse_quote(ticker: str) -> Dict[str, Any] | None:
    """Map the equity-pipeline warehouse's NSE EOD snapshot into the
    ``get_quote`` schema, converting INR → USD like the live path so every
    agent still speaks one currency. Returns ``None`` when the warehouse is
    disabled / unreachable / has no row for ``ticker`` (caller falls back).
    """
    row = _warehouse.get_quote(ticker)
    if not row:
        return None
    return {
        "ticker": (row.get("ticker") or ticker).upper(),
        "name": row.get("company_name"),
        "exchange": "NSE",
        "sector": row.get("sector"),
        "price": _inr_to_usd(row.get("latest_close")),
        "vwap": _inr_to_usd(row.get("latest_vwap")),
        "volume": row.get("latest_volume"),
        "52w_high": _inr_to_usd(row.get("high_52w")),
        "52w_low": _inr_to_usd(row.get("low_52w")),
        "return_30d_pct": row.get("return_30d_pct"),
        "currency": "USD",
        "native_currency": "INR",
        "fx_rate_usd_per_inr": round(_live.USD_PER_INR, 6),
        "as_of_date": row.get("latest_date"),
        "_source": "warehouse:equity-pipeline",
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool
def list_supported_tickers() -> Dict[str, Any]:
    """Return the list of Indian tickers the fixture has curated."""
    return {
        "fixture_tickers": _available_tickers(),
        "live_supported": (
            "any NSE symbol via yfinance '.NS' (e.g. TCS.NS, INFY.NS), or "
            "'.BO' for BSE-only listings."
        ),
    }


@mcp.tool
def get_quote(ticker: str) -> Dict[str, Any]:
    """Return the latest quote for an NSE ticker.

    Source preference: **equity-pipeline warehouse** (official NSE bhavcopy
    EOD, exchange-sourced) → live ``yfinance`` (15-min delayed) → curated
    fixture. The warehouse path only kicks in when ``WAREHOUSE_DATABASE_URL``
    is configured and the ticker is in its NIFTY-50 universe; otherwise it
    transparently falls back. All paths return USD (converted from the
    exchange's native INR via :data:`src.mcp._live.USD_PER_INR`) so
    downstream consumers always see one currency.
    """
    warehoused = _warehouse_quote(ticker)
    if warehoused is not None:
        return warehoused
    live = _live_map(ticker, _live.map_quote)
    if live is not None:
        return live
    entry = _fixture_entry(ticker)
    if not entry:
        return {"ticker": ticker.upper(), "error": "no live or fixture data", "_source": "none"}
    return {
        "ticker": ticker.upper(),
        "name": entry.get("name"),
        "currency": "USD",
        "native_currency": "INR",
        **(entry.get("quote") or {}),
        "_source": "fixture:indian_stocks",
    }


@mcp.tool
def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """Return valuation, profitability, and leverage metrics for an NSE ticker."""
    live = _live_map(ticker, _live.map_fundamentals)
    if live is not None:
        return live
    entry = _fixture_entry(ticker)
    if not entry:
        return {"ticker": ticker.upper(), "error": "no live or fixture data", "_source": "none"}
    # Flatten ``quote`` + ``fundamentals`` so downstream renderers see
    # a single response that includes ``price`` and ``market_cap_bn``
    # alongside the valuation / leverage fields. Mirrors the US worker.
    quote = entry.get("quote") or {}
    fundamentals = entry.get("fundamentals") or {}
    return {
        "ticker": ticker.upper(),
        "name": entry.get("name"),
        "sector": entry.get("sector"),
        "industry": entry.get("industry"),
        "currency": "USD",
        "native_currency": "INR",
        "price": quote.get("price"),
        "market_cap_bn": quote.get("market_cap_bn"),
        **fundamentals,
        "_source": "fixture:indian_stocks",
    }


@mcp.tool
def get_growth_metrics(ticker: str) -> Dict[str, Any]:
    """Return growth / innovation indicators for an NSE ticker.

    Merges live 1-year revenue + earnings growth + beta (yfinance) with
    the curated 5-year CAGRs, R&D intensity, addressable market, and
    narrative from the fixture.
    """
    entry = _fixture_entry(ticker)
    fixture_growth = entry.get("growth") or {}
    out: Dict[str, Any] = {
        "ticker": ticker.upper(),
        "name": entry.get("name") or ticker.upper(),
        **fixture_growth,
    }
    live = _live_map(ticker, _live.map_growth)
    if live is not None:
        for k, v in live.items():
            if k.startswith("_") or k in ("ticker", "name"):
                continue
            out[k] = v
        out["_source"] = "live+fixture"
        out["_as_of"] = live.get("_as_of")
    else:
        out["_source"] = "fixture:indian_stocks"
    return out


@mcp.tool
def get_defensive_metrics(ticker: str) -> Dict[str, Any]:
    """Return balance-sheet strength and Graham-style defensive metrics.

    Banks (HDFCBANK) expose different statutory fields (NPA, CAR) that
    yfinance doesn't surface consistently. For such names the fixture
    values for those fields are merged on top of the live response.
    """
    live = _live_map(ticker, _live.map_defensive)
    entry = _fixture_entry(ticker)
    fixture_defensive = entry.get("defensive") or {}

    if live is not None:
        # Overlay any fixture-only keys (bank-specific ratios, dividend_yield
        # for names yfinance marks as None, etc.) without clobbering live
        # values.
        for k, v in fixture_defensive.items():
            if k not in live or live.get(k) is None:
                live[k] = v
        return live

    if not entry:
        return {"ticker": ticker.upper(), "error": "no live or fixture data", "_source": "none"}
    return {
        "ticker": ticker.upper(),
        "name": entry.get("name"),
        **fixture_defensive,
        "_source": "fixture:indian_stocks",
    }


@mcp.tool
def get_moat_signals(ticker: str) -> Dict[str, Any]:
    """Return qualitative moat signals for an NSE ticker (fixture-sourced)."""
    entry = _fixture_entry(ticker)
    if not entry:
        return {
            "ticker": ticker.upper(),
            "error": "no curated moat narrative for this ticker",
            "_source": "none",
        }
    return {
        "ticker": ticker.upper(),
        "name": entry.get("name"),
        "moat_signals": entry.get("moat_signals", []),
        "disruption_score": (entry.get("growth") or {}).get("disruption_score"),
        "narrative": (entry.get("growth") or {}).get("narrative"),
        "_source": "fixture:indian_stocks",
    }


# ---------------------------------------------------------------------------
# Warehouse-backed market-data tools (equity-pipeline; NIFTY-50, NSE EOD)
# ---------------------------------------------------------------------------
# These surface data the warehouse uniquely provides (clean exchange history,
# sector performance, market movers) that yfinance/fixtures don't. They need
# WAREHOUSE_DATABASE_URL configured; without it they return a clear notice so
# the planner knows to skip them rather than the agent erroring.
_WAREHOUSE_OFF = {
    "error": "equity warehouse not configured (set WAREHOUSE_DATABASE_URL)",
    "_source": "none",
}


@mcp.tool
def get_price_history(ticker: str, days: int = 30) -> Dict[str, Any]:
    """Daily OHLCV + VWAP + 7d/30d moving averages for an NSE ticker over the
    trailing ``days`` (default 30, max 365), from the equity-pipeline warehouse
    (official NSE bhavcopy EOD). Prices are converted INR → USD.

    Warehouse-only: returns a notice if the warehouse isn't configured or the
    ticker isn't in its NIFTY-50 universe.
    """
    if not _warehouse.is_available():
        return {"ticker": ticker.upper(), **_WAREHOUSE_OFF}
    days = max(1, min(int(days or 30), 365))
    to_date = dt.date.today()
    rows = _warehouse.get_history(ticker, to_date - dt.timedelta(days=days), to_date)
    if not rows:
        return {
            "ticker": ticker.upper(),
            "error": "no warehouse history (not in the NIFTY-50 universe?)",
            "_source": "warehouse:equity-pipeline",
        }
    bars = [
        {
            "date": r.get("trade_date"),
            "open": _inr_to_usd(r.get("open")),
            "high": _inr_to_usd(r.get("high")),
            "low": _inr_to_usd(r.get("low")),
            "close": _inr_to_usd(r.get("close")),
            "vwap": _inr_to_usd(r.get("vwap")),
            "volume": r.get("volume"),
            "ma_7d": _inr_to_usd(r.get("ma_7d")),
            "ma_30d": _inr_to_usd(r.get("ma_30d")),
        }
        for r in rows
    ]
    return {
        "ticker": ticker.upper(),
        "currency": "USD",
        "native_currency": "INR",
        "fx_rate_usd_per_inr": round(_live.USD_PER_INR, 6),
        "days": days,
        "count": len(bars),
        "bars": bars,
        "_source": "warehouse:equity-pipeline",
    }


@mcp.tool
def get_technicals(ticker: str) -> Dict[str, Any]:
    """Derived technical indicators for an NSE ticker from the equity-pipeline
    warehouse (official NSE bhavcopy EOD): simple moving averages (20/50/200),
    trailing 1m/3m/6m returns, annualised volatility, max drawdown, and trend vs
    the 50-day SMA — computed over ~400 calendar days of warehoused closes.

    SMA values and the latest close are converted INR → USD; returns, volatility
    and drawdown are currency-invariant percentages and pass through unchanged.

    Warehouse-only: returns a notice if the warehouse isn't configured or the
    ticker isn't in its NIFTY-50 universe.
    """
    if not _warehouse.is_available():
        return {"ticker": ticker.upper(), **_WAREHOUSE_OFF}
    t = _warehouse.get_technicals(ticker)
    if not t:
        return {
            "ticker": ticker.upper(),
            "error": "no warehouse history (not in the NIFTY-50 universe?)",
            "_source": "warehouse:equity-pipeline",
        }
    return {
        "ticker": t["ticker"],
        "as_of": t.get("as_of"),
        "currency": "USD",
        "native_currency": "INR",
        "fx_rate_usd_per_inr": round(_live.USD_PER_INR, 6),
        "data_points": t.get("data_points"),
        "latest_close": _inr_to_usd(t.get("latest_close")),
        "sma_20": _inr_to_usd(t.get("sma_20")),
        "sma_50": _inr_to_usd(t.get("sma_50")),
        "sma_200": _inr_to_usd(t.get("sma_200")),
        "trend_vs_sma_50": t.get("trend_vs_sma_50"),
        "return_1m_pct": t.get("return_1m_pct"),
        "return_3m_pct": t.get("return_3m_pct"),
        "return_6m_pct": t.get("return_6m_pct"),
        "annualized_volatility_pct": t.get("annualized_volatility_pct"),
        "max_drawdown_pct": t.get("max_drawdown_pct"),
        "_source": "warehouse:equity-pipeline",
    }


@mcp.tool
def get_top_movers(limit: int = 10) -> Dict[str, Any]:
    """Top gainers over the trailing 30 days across the NIFTY-50 universe, from
    the equity-pipeline warehouse. Prices converted INR → USD; returns are %.

    Warehouse-only: returns a notice if the warehouse isn't configured.
    """
    if not _warehouse.is_available():
        return _WAREHOUSE_OFF
    rows = _warehouse.get_top_movers(limit=max(1, min(int(limit or 10), 50)))
    if rows is None:
        return {"error": "warehouse query failed", "_source": "warehouse:equity-pipeline"}
    movers = [
        {
            "ticker": r.get("ticker"),
            "name": r.get("company_name"),
            "sector": r.get("sector"),
            "close_30d_ago": _inr_to_usd(r.get("close_30d_ago")),
            "close_today": _inr_to_usd(r.get("close_today")),
            "return_30d_pct": r.get("return_30d_pct"),
        }
        for r in rows
    ]
    return {
        "universe": "NIFTY-50",
        "window": "30d",
        "currency": "USD",
        "movers": movers,
        "_source": "warehouse:equity-pipeline",
    }


@mcp.tool
def get_sector_performance(weeks: int = 4) -> Dict[str, Any]:
    """Average daily return (%) and traded volume by sector and ISO week across
    the NIFTY-50 universe, from the equity-pipeline warehouse. Bounded to the
    most recent ``weeks`` ISO weeks (default 4, max 52) so the payload stays
    small — the view itself holds the full backfill.

    Warehouse-only: returns a notice if the warehouse isn't configured.
    """
    if not _warehouse.is_available():
        return _WAREHOUSE_OFF
    rows = _warehouse.get_sector_performance()
    if rows is None:
        return {"error": "warehouse query failed", "_source": "warehouse:equity-pipeline"}
    weeks = max(1, min(int(weeks or 4), 52))
    # Keep only the most recent ``weeks`` (year, ISO-week) buckets.
    recent_keys = sorted(
        {(r.get("year"), r.get("week_number")) for r in rows}, reverse=True
    )[:weeks]
    keyset = set(recent_keys)
    recent = [r for r in rows if (r.get("year"), r.get("week_number")) in keyset]
    recent.sort(
        key=lambda r: (r.get("year") or 0, r.get("week_number") or 0, r.get("sector") or ""),
        reverse=True,
    )
    return {
        "universe": "NIFTY-50",
        "weeks": weeks,
        "rows": recent,
        "_source": "warehouse:equity-pipeline",
    }


if __name__ == "__main__":  # pragma: no cover - entrypoint
    mcp.run()
