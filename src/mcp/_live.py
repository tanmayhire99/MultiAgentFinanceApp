"""Live market data with in-process 1-hour caching.

Backs both the US and Indian stock MCP workers. We use :mod:`yfinance` for
both markets because it handles ``.NS`` / ``.BO`` suffixes cleanly and has
the most complete fundamentals coverage for our tickers. If you want to
swap in ``nsepython`` or ``bsedata`` for real-time NSE/BSE quotes later,
just replace the one-liner inside :func:`_fetch_info`.

All public ``fetch_*`` helpers are wrapped in a :class:`cachetools.TTLCache`
so repeated calls for the same ticker within an hour are served from
memory and never hit Yahoo.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional

from cachetools import TTLCache

try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover - guarded at runtime
    yf = None


log = logging.getLogger("finai.live")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
TTL_SECONDS = 3600  # 1 hour
_INFO_CACHE: TTLCache = TTLCache(maxsize=500, ttl=TTL_SECONDS)


def cache_stats() -> Dict[str, Any]:
    """Useful for debugging; exposed via the MCP worker if desired."""
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate_pct = (100.0 * _CACHE_HITS / total) if total else 0.0
    return {
        "size": len(_INFO_CACHE),
        "maxsize": _INFO_CACHE.maxsize,
        "ttl_seconds": _INFO_CACHE.ttl,
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "hit_rate_pct": round(hit_rate_pct, 1),
    }


def is_live_available() -> bool:
    return yf is not None


# ---------------------------------------------------------------------------
# Raw fetch
# ---------------------------------------------------------------------------
def _normalise_ticker(ticker: str, suffix: str = "") -> str:
    """Return ``<TICKER><suffix>`` upper-cased (no suffix if already present)."""
    t = (ticker or "").strip().upper()
    if suffix and not (t.endswith(".NS") or t.endswith(".BO")):
        return f"{t}{suffix}"
    return t


# Running tallies for cache hits / misses - purely for observability.
# Exposed via :func:`cache_stats` so the main app can report them.
_CACHE_HITS = 0
_CACHE_MISSES = 0


def fetch_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Return ``yf.Ticker(symbol).info`` (cached 1h), or ``None`` on failure.

    Args:
        symbol: the full Yahoo symbol - include ``.NS`` or ``.BO`` for
            Indian tickers (``"RELIANCE.NS"``, ``"INFY.NS"``).

    Emits one ``CACHE-HIT`` or ``CACHE-MISS`` log line per call so the
    caching behaviour is visible in ``docker logs finai-api``.
    """
    global _CACHE_HITS, _CACHE_MISSES

    cached = _INFO_CACHE.get(symbol)
    if cached is not None:
        _CACHE_HITS += 1
        age_s = int(time.time() - (cached.get("_fetched_at") or 0))
        log.info(
            "yfinance CACHE-HIT  %-12s  age=%ds  hits=%d/misses=%d",
            symbol,
            age_s,
            _CACHE_HITS,
            _CACHE_MISSES,
        )
        return cached

    _CACHE_MISSES += 1
    log.info(
        "yfinance CACHE-MISS %-12s  calling yfinance... hits=%d/misses=%d",
        symbol,
        _CACHE_HITS,
        _CACHE_MISSES,
    )

    if yf is None:
        return None
    try:
        info = yf.Ticker(symbol).info
    except Exception as e:  # pragma: no cover - network / scrape flake
        log.warning("yfinance fetch failed for %s: %s", symbol, e)
        return None
    # yfinance sometimes returns a skeleton dict for invalid tickers.
    if not info or (
        info.get("currentPrice") is None
        and info.get("regularMarketPrice") is None
        and info.get("previousClose") is None
    ):
        return None
    # Attach a fetch timestamp for observability.
    info = dict(info)
    info["_fetched_at"] = int(time.time())
    _INFO_CACHE[symbol] = info
    return info


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _round(x: Any, digits: int = 2) -> Optional[float]:
    if x is None:
        return None
    try:
        return round(float(x), digits)
    except (TypeError, ValueError):
        return None


def _pct(ratio: Any, digits: int = 2) -> Optional[float]:
    """Convert a 0.1234 ratio into a 12.34 percentage."""
    if ratio is None:
        return None
    try:
        return round(float(ratio) * 100, digits)
    except (TypeError, ValueError):
        return None


def _bn(raw: Any) -> Optional[float]:
    """Convert a raw nominal value to billions of the base currency."""
    if raw is None:
        return None
    try:
        return round(float(raw) / 1e9, 2)
    except (TypeError, ValueError):
        return None


def _dividend_yield_pct(info: Dict[str, Any]) -> Optional[float]:
    """Extract a dividend yield (%) robust to yfinance's mixed conventions.

    In yfinance >= 0.2.40 the ``dividendYield`` field is already a
    percentage (e.g. 2.44 for 2.44%), while ``trailingAnnualDividendYield``
    stays a fraction (0.0244). We prefer the percentage field but
    heuristically detect fraction-encoded values (< 0.2 is almost
    certainly a fraction, since yields above 20 % are extremely rare).
    """
    dy = info.get("dividendYield")
    if dy is None:
        ttm = info.get("trailingAnnualDividendYield")
        if ttm is None:
            return None
        try:
            return round(float(ttm) * 100, 2)
        except (TypeError, ValueError):
            return None
    try:
        dyf = float(dy)
    except (TypeError, ValueError):
        return None
    # Heuristic: if it looks like a fraction, convert.
    if 0 < dyf < 0.2:
        return round(dyf * 100, 2)
    return round(dyf, 2)


# ---------------------------------------------------------------------------
# Derived quant: Graham number + margin of safety
# ---------------------------------------------------------------------------
def compute_graham_number(info: Dict[str, Any]) -> Optional[float]:
    """Classic Benjamin Graham intrinsic value: sqrt(22.5 * EPS * BVPS)."""
    eps = info.get("trailingEps")
    bvps = info.get("bookValue")
    if eps is None or bvps is None:
        return None
    try:
        eps = float(eps)
        bvps = float(bvps)
    except (TypeError, ValueError):
        return None
    if eps <= 0 or bvps <= 0:
        return None
    try:
        return round(math.sqrt(22.5 * eps * bvps), 2)
    except (ValueError, OverflowError):
        return None


def compute_margin_of_safety_vs_graham_pct(
    price: Optional[float], graham: Optional[float]
) -> Optional[float]:
    """``(graham - price) / price * 100``. Negative means overvalued vs Graham."""
    if price is None or graham is None:
        return None
    try:
        price_f = float(price)
        if price_f == 0:
            return None
        return round(((float(graham) - price_f) / price_f) * 100, 2)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Mappers (yfinance info dict -> our stable MCP tool schema)
# ---------------------------------------------------------------------------
def _best_price(info: Dict[str, Any]) -> Optional[float]:
    for key in ("currentPrice", "regularMarketPrice", "previousClose"):
        v = info.get(key)
        if v is not None:
            return _round(v)
    return None


def map_quote(info: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "exchange": info.get("exchange"),
        "price": _best_price(info),
        "previous_close": _round(info.get("previousClose")),
        "change_pct_1d": _round(info.get("regularMarketChangePercent")),
        "volume": info.get("regularMarketVolume") or info.get("volume"),
        "market_cap_bn": _bn(info.get("marketCap")),
        "52w_high": _round(info.get("fiftyTwoWeekHigh")),
        "52w_low": _round(info.get("fiftyTwoWeekLow")),
    }


def map_fundamentals(info: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or info.get("shortName") or ticker.upper(),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        # Price included here so the Market Snapshot renderer can show it
        # alongside the valuation/margin numbers from a single tool call.
        "price": _best_price(info),
        "market_cap_bn": _bn(info.get("marketCap")),
        "pe_ttm": _round(info.get("trailingPE")),
        "forward_pe": _round(info.get("forwardPE")),
        "pb": _round(info.get("priceToBook")),
        "ps": _round(info.get("priceToSalesTrailing12Months")),
        "roe_pct": _pct(info.get("returnOnEquity")),
        "roa_pct": _pct(info.get("returnOnAssets")),
        "gross_margin_pct": _pct(info.get("grossMargins")),
        "operating_margin_pct": _pct(info.get("operatingMargins")),
        "profit_margin_pct": _pct(info.get("profitMargins")),
        # yfinance reports debtToEquity as a percentage (e.g. 170 = 1.70x);
        # divide to get the conventional ratio our agents expect.
        "debt_to_equity": _round(
            (info.get("debtToEquity") or 0) / 100
            if info.get("debtToEquity") is not None
            else None
        ),
        "revenue_bn_ttm": _bn(info.get("totalRevenue")),
        "net_income_bn_ttm": _bn(info.get("netIncomeToCommon")),
    }


def map_defensive(info: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    price = _best_price(info)
    graham = compute_graham_number(info)
    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or ticker.upper(),
        "current_ratio": _round(info.get("currentRatio")),
        "quick_ratio": _round(info.get("quickRatio")),
        "graham_number": graham,
        "margin_of_safety_vs_graham_pct": compute_margin_of_safety_vs_graham_pct(
            price, graham
        ),
        "dividend_yield_pct": _dividend_yield_pct(info),
        "book_value_per_share": _round(info.get("bookValue")),
        "eps_ttm": _round(info.get("trailingEps")),
        "cash_and_equivalents_bn": _bn(info.get("totalCash")),
    }


def map_growth(info: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    """One-year growth metrics from yfinance.

    Multi-year CAGRs (e.g. 5y revenue CAGR) still come from curated fixtures
    because computing them well requires a full financial-statement pull.
    """
    return {
        "ticker": ticker.upper(),
        "name": info.get("longName") or ticker.upper(),
        "revenue_growth_pct_1y": _pct(info.get("revenueGrowth")),
        "earnings_growth_pct_1y": _pct(info.get("earningsGrowth")),
        "beta": _round(info.get("beta")),
    }


# ---------------------------------------------------------------------------
# USD conversion
# ---------------------------------------------------------------------------
# yfinance returns prices in the exchange's native currency: USD for US
# listings, INR for NSE/BSE (.NS/.BO) listings. Our system displays
# everything in USD (to match the portfolio fixture + US holdings),
# so Indian-stock responses get the currency-denominated fields
# converted on the way out.
#
# The rate is intentionally a constant (matches the portfolio fixture's
# assumption) so the conversion is deterministic and auditable. Real
# production would pull this from a live FX feed.
USD_PER_INR = 1.0 / 85.0


# Fields in the mapper outputs that are denominated in the native
# exchange currency (and therefore need FX conversion for Indian stocks).
_CURRENCY_FIELDS = frozenset(
    {
        # map_quote
        "price",
        "previous_close",
        "52w_high",
        "52w_low",
        "market_cap_bn",
        # map_fundamentals (also has price + market_cap_bn above)
        "revenue_bn_ttm",
        "net_income_bn_ttm",
        # map_defensive
        "graham_number",
        "book_value_per_share",
        "eps_ttm",
        "cash_and_equivalents_bn",
    }
)


def _is_indian_symbol(symbol: str) -> bool:
    s = (symbol or "").upper()
    return s.endswith(".NS") or s.endswith(".BO")


def _apply_fx_to_usd(row: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """If ``symbol`` is an Indian listing, convert currency-denominated fields INR→USD.

    Percentages, ratios, counts and dates are untouched. A ``currency``
    tag is attached to every row so downstream renderers can show the
    unit without guessing from the ticker suffix.
    """
    row["currency"] = "USD"
    if not _is_indian_symbol(symbol):
        return row
    row["native_currency"] = "INR"
    row["fx_rate_usd_per_inr"] = round(USD_PER_INR, 6)
    for field in _CURRENCY_FIELDS:
        v = row.get(field)
        if v is None:
            continue
        try:
            row[field] = round(float(v) * USD_PER_INR, 4)
        except (TypeError, ValueError):
            # Leave non-numeric values alone (shouldn't happen, but defensive).
            pass
    return row


# ---------------------------------------------------------------------------
# Convenience: fetch + map in one call, with source annotation
# ---------------------------------------------------------------------------
def fetch_and_map(
    ticker: str,
    mapper,
    suffix: str = "",
) -> Optional[Dict[str, Any]]:
    """Fetch ``ticker`` (optionally adding market suffix), run ``mapper``, convert to USD.

    Indian-stock rows are converted from INR (yfinance's native currency
    for .NS/.BO) to USD via the constant in :data:`USD_PER_INR` so every
    downstream component speaks USD.

    Returns ``None`` if yfinance couldn't produce a usable info dict; the
    caller can then fall back to fixture data.
    """
    symbol = _normalise_ticker(ticker, suffix=suffix)
    info = fetch_info(symbol)
    if info is None:
        return None
    out = mapper(info, ticker)
    _apply_fx_to_usd(out, symbol)
    out["_source"] = "live:yfinance"
    out["_as_of"] = info.get("_fetched_at")
    return out
