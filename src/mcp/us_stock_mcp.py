"""US Stock MCP worker server (live yfinance + curated fallback fixtures).

Tools:
    - ``get_quote``             -> price, 52w range, market cap
    - ``get_fundamentals``      -> valuation + margin + return ratios
    - ``get_growth_metrics``    -> 1y revenue/earnings growth + 5y CAGRs narrative
    - ``get_defensive_metrics`` -> liquidity, Graham number, margin of safety
    - ``get_moat_signals``      -> qualitative moat statements + narrative

Quantitative fields come from :mod:`yfinance` (1-hour cached in
:mod:`src.mcp._live`). Qualitative / narrative fields (moat
signals, disruption score, addressable market, analyst takes) stay with
the curated fixture so the persona agents have a consistent story.

Each response includes a ``_source`` marker so callers can see whether a
particular response was served live or from fallback.

Run as a standalone MCP server over stdio::

    python -m src.mcp.us_stock_mcp
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastmcp import FastMCP

from . import _live
from ._fixtures import load_fixture, lookup


_FIXTURES: Dict[str, Any] = load_fixture("us_stocks")


mcp = FastMCP(
    name="us_stock",
    instructions=(
        "Provides US equity data (quotes, fundamentals, growth, defensive "
        "metrics, qualitative moat signals). Quant fields are served live "
        "from yfinance with a 1-hour cache; qualitative narrative fields "
        "come from curated fixtures. Responses include a ``_source`` tag."
    ),
)


def _fixture_entry(ticker: str) -> Dict[str, Any]:
    """Return the fixture entry for ``ticker`` or ``{}`` if not curated."""
    try:
        return lookup(_FIXTURES, ticker)
    except KeyError:
        return {}


def _available_tickers() -> List[str]:
    return sorted(k for k in _FIXTURES.keys() if not k.startswith("_"))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool
def list_supported_tickers() -> Dict[str, Any]:
    """Return the list of US tickers the fixture has curated narratives for.

    Note: live quote/fundamentals/defensive metrics work for *any* valid
    Yahoo Finance US symbol. The curated list is just the set where we
    also have qualitative moat / narrative content.
    """
    return {
        "fixture_tickers": _available_tickers(),
        "live_supported": "any Yahoo Finance US symbol (e.g. AAPL, NFLX, META, AMZN)",
    }


@mcp.tool
def get_quote(ticker: str) -> Dict[str, Any]:
    """Return the latest quote for a US ticker (live yfinance, 1h cached)."""
    live = _live.fetch_and_map(ticker, _live.map_quote)
    if live is not None:
        return live
    # Fallback to fixture
    entry = _fixture_entry(ticker)
    if not entry:
        return {"ticker": ticker.upper(), "error": "no live or fixture data", "_source": "none"}
    return {
        "ticker": ticker.upper(),
        "name": entry.get("name"),
        **(entry.get("quote") or {}),
        "_source": "fixture:us_stocks",
    }


@mcp.tool
def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """Return valuation, profitability, and leverage metrics for a US ticker."""
    live = _live.fetch_and_map(ticker, _live.map_fundamentals)
    if live is not None:
        return live
    entry = _fixture_entry(ticker)
    if not entry:
        return {"ticker": ticker.upper(), "error": "no live or fixture data", "_source": "none"}
    # The fixture splits "price + market cap" under ``quote`` and
    # "P/E + margins + leverage" under ``fundamentals``. Downstream
    # consumers (the Market Snapshot table + persona agents) expect
    # both groups to be flattened into a single response, so merge
    # them here. Without this merge the Price column renders as "—"
    # whenever we're on the fixture fallback path.
    quote = entry.get("quote") or {}
    fundamentals = entry.get("fundamentals") or {}
    return {
        "ticker": ticker.upper(),
        "name": entry.get("name"),
        "sector": entry.get("sector"),
        "industry": entry.get("industry"),
        "price": quote.get("price"),
        "market_cap_bn": quote.get("market_cap_bn"),
        **fundamentals,
        "_source": "fixture:us_stocks",
    }


@mcp.tool
def get_growth_metrics(ticker: str) -> Dict[str, Any]:
    """Return growth / innovation indicators for a US ticker.

    Merges live 1-year revenue + earnings growth and beta (from yfinance)
    with the curated 5-year CAGRs, R&D intensity, addressable market
    and narrative from the fixture.
    """
    entry = _fixture_entry(ticker)
    fixture_growth = entry.get("growth") or {}
    out: Dict[str, Any] = {
        "ticker": ticker.upper(),
        "name": entry.get("name") or ticker.upper(),
        **fixture_growth,
    }
    live = _live.fetch_and_map(ticker, _live.map_growth)
    if live is not None:
        # Merge live 1y metrics (won't clash with 5y fixture names)
        for k, v in live.items():
            if k.startswith("_") or k in ("ticker", "name"):
                continue
            out[k] = v
        out["_source"] = "live+fixture"
        out["_as_of"] = live.get("_as_of")
    else:
        out["_source"] = "fixture:us_stocks"
    return out


@mcp.tool
def get_defensive_metrics(ticker: str) -> Dict[str, Any]:
    """Return balance-sheet strength and Graham-style defensive metrics."""
    live = _live.fetch_and_map(ticker, _live.map_defensive)
    if live is not None:
        return live
    entry = _fixture_entry(ticker)
    if not entry:
        return {"ticker": ticker.upper(), "error": "no live or fixture data", "_source": "none"}
    return {
        "ticker": ticker.upper(),
        "name": entry.get("name"),
        **(entry.get("defensive") or {}),
        "_source": "fixture:us_stocks",
    }


@mcp.tool
def get_moat_signals(ticker: str) -> Dict[str, Any]:
    """Return qualitative moat signals + disruption narrative for a US ticker.

    Always comes from the curated fixture - these are editorial statements
    (CUDA lock-in, ecosystem flywheel, etc.) rather than metrics derivable
    from a price feed.
    """
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
        "_source": "fixture:us_stocks",
    }


if __name__ == "__main__":  # pragma: no cover - entrypoint
    mcp.run()
