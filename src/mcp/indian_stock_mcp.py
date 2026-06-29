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


def _warehouse_quote(ticker: str) -> Dict[str, Any] | None:
    """Map the equity-pipeline warehouse's NSE EOD snapshot into the
    ``get_quote`` schema, converting INR → USD like the live path so every
    agent still speaks one currency. Returns ``None`` when the warehouse is
    disabled / unreachable / has no row for ``ticker`` (caller falls back).
    """
    row = _warehouse.get_quote(ticker)
    if not row:
        return None
    fx = _live.USD_PER_INR

    def _usd(v: Any) -> Any:
        try:
            return round(float(v) * fx, 4) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "ticker": (row.get("ticker") or ticker).upper(),
        "name": row.get("company_name"),
        "exchange": "NSE",
        "sector": row.get("sector"),
        "price": _usd(row.get("latest_close")),
        "vwap": _usd(row.get("latest_vwap")),
        "volume": row.get("latest_volume"),
        "52w_high": _usd(row.get("high_52w")),
        "52w_low": _usd(row.get("low_52w")),
        "return_30d_pct": row.get("return_30d_pct"),
        "currency": "USD",
        "native_currency": "INR",
        "fx_rate_usd_per_inr": round(fx, 6),
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


if __name__ == "__main__":  # pragma: no cover - entrypoint
    mcp.run()
