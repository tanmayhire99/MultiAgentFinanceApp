"""Live web research with in-process 1-hour caching.

Backs the Research MCP worker. Uses a **multi-backend** strategy so the
demo works out of the box with zero signup:

    1. Tavily - highest quality (agent-optimised, pre-summarised results).
       Used only when ``TAVILY_API_KEY`` is set in the environment.
    2. DuckDuckGo (``ddgs`` library) - no key required, free.
    3. Curated fixture snippets - last-resort fallback when both live
       backends are unreachable.

All public ``*_cached`` helpers share a :class:`cachetools.TTLCache` keyed
on (tool, normalised-args) so repeated persona calls for the same ticker
don't hammer the upstream providers.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from cachetools import TTLCache

try:
    from tavily import TavilyClient  # type: ignore
except ImportError:  # pragma: no cover - guarded at runtime
    TavilyClient = None  # type: ignore

try:
    from ddgs import DDGS  # type: ignore
except ImportError:  # pragma: no cover - guarded at runtime
    DDGS = None  # type: ignore


log = logging.getLogger("finai.research")


# ---------------------------------------------------------------------------
# Cache (shared across all tools inside this subprocess)
# ---------------------------------------------------------------------------
TTL_SECONDS = 3600  # 1 hour
_CACHE: TTLCache = TTLCache(maxsize=500, ttl=TTL_SECONDS)

# Cache hit/miss tallies - logged on every access so the caching behaviour
# is visible in ``docker logs finai-api``.
_CACHE_HITS = 0
_CACHE_MISSES = 0


def cache_stats() -> Dict[str, Any]:
    total = _CACHE_HITS + _CACHE_MISSES
    hit_rate_pct = (100.0 * _CACHE_HITS / total) if total else 0.0
    return {
        "size": len(_CACHE),
        "maxsize": _CACHE.maxsize,
        "ttl_seconds": _CACHE.ttl,
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "hit_rate_pct": round(hit_rate_pct, 1),
    }


def _cache_key(*parts) -> str:
    return "|".join(str(p) for p in parts)


def _cache_check(kind: str, key: str) -> Optional[Any]:
    """Unified cache lookup with per-call logging."""
    global _CACHE_HITS, _CACHE_MISSES
    if key in _CACHE:
        _CACHE_HITS += 1
        log.info(
            "research CACHE-HIT  %-8s %-60s  hits=%d/misses=%d",
            kind,
            key[:60],
            _CACHE_HITS,
            _CACHE_MISSES,
        )
        return _CACHE[key]
    _CACHE_MISSES += 1
    log.info(
        "research CACHE-MISS %-8s %-60s  hits=%d/misses=%d",
        kind,
        key[:60],
        _CACHE_HITS,
        _CACHE_MISSES,
    )
    return None


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------
def _tavily_client() -> Optional[Any]:
    key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not key or TavilyClient is None:
        return None
    try:
        return TavilyClient(api_key=key)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("Failed to construct Tavily client: %s", e)
        return None


def is_tavily_available() -> bool:
    return _tavily_client() is not None


def is_ddg_available() -> bool:
    return DDGS is not None


def active_backends() -> List[str]:
    out = []
    if is_tavily_available():
        out.append("tavily")
    if is_ddg_available():
        out.append("duckduckgo")
    out.append("fixture")
    return out


# ---------------------------------------------------------------------------
# Tavily adapters
# ---------------------------------------------------------------------------
def _tavily_search(
    query: str,
    *,
    max_results: int = 5,
    topic: str = "general",
    include_answer: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    time_range: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Call Tavily. Returns normalised dict on success, ``None`` on failure.

    Supports Tavily's historical search parameters:

    * ``start_date`` / ``end_date`` (``YYYY-MM-DD``): return only results
      published in that window. Essential for the claim-tracking flow
      (e.g. ``start_date="2024-01-01", end_date="2024-06-30"`` to pull
      the original management guidance before checking whether it was
      actually met).
    * ``time_range`` (``day`` | ``week`` | ``month`` | ``year``):
      coarser shortcut when exact bounds aren't needed.
    * ``topic``: ``general`` (default), ``news``, or ``finance``. The
      ``finance`` topic biases Tavily toward earnings releases,
      analyst reports, and filings.
    """
    client = _tavily_client()
    if client is None:
        return None
    kwargs: Dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "topic": topic,
        "include_answer": include_answer,
        "search_depth": "basic",
    }
    if start_date:
        kwargs["start_date"] = start_date
    if end_date:
        kwargs["end_date"] = end_date
    if time_range:
        kwargs["time_range"] = time_range
    try:
        resp = client.search(**kwargs)
    except Exception as e:
        log.warning("Tavily search failed for %r: %s", query[:60], e)
        return None
    if not isinstance(resp, dict):
        return None
    items = []
    for r in resp.get("results", []) or []:
        items.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content") or "").strip(),
                "score": r.get("score"),
                "published": r.get("published_date"),
            }
        )
    return {
        "backend": "tavily",
        "query": query,
        "answer": resp.get("answer"),
        "items": items,
        "fetched_at": int(time.time()),
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "time_range": time_range,
        } if (start_date or end_date or time_range) else None,
    }


# ---------------------------------------------------------------------------
# DuckDuckGo adapters
# ---------------------------------------------------------------------------
def _ddg_text(query: str, *, max_results: int = 5) -> Optional[Dict[str, Any]]:
    """Call DuckDuckGo text search."""
    if DDGS is None:
        return None
    try:
        with DDGS() as ddg:
            raw = list(ddg.text(query, max_results=max_results))
    except Exception as e:
        log.warning("DDG text search failed for %r: %s", query[:60], e)
        return None
    items = [
        {
            "title": (r.get("title") or "").strip(),
            "url": r.get("href") or r.get("url") or "",
            "snippet": (r.get("body") or "").strip(),
            "score": None,
            "published": None,
        }
        for r in raw
    ]
    return {
        "backend": "duckduckgo",
        "query": query,
        "answer": None,
        "items": items,
        "fetched_at": int(time.time()),
    }


def _ddg_news(query: str, *, max_results: int = 5) -> Optional[Dict[str, Any]]:
    """Call DuckDuckGo news search specifically (date-tagged results)."""
    if DDGS is None:
        return None
    try:
        with DDGS() as ddg:
            raw = list(ddg.news(query, max_results=max_results))
    except Exception as e:
        log.warning("DDG news search failed for %r: %s", query[:60], e)
        # Fall back to text search if news endpoint is flaky
        return _ddg_text(query, max_results=max_results)
    items = [
        {
            "title": (r.get("title") or "").strip(),
            "url": r.get("url") or "",
            "snippet": (r.get("body") or "").strip(),
            "score": None,
            "published": r.get("date") or r.get("published"),
            "source": r.get("source"),
        }
        for r in raw
    ]
    return {
        "backend": "duckduckgo",
        "query": query,
        "answer": None,
        "items": items,
        "fetched_at": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Public helpers (cached, backend-ordered)
# ---------------------------------------------------------------------------
def search_news(ticker: str, max_items: int = 3) -> Optional[Dict[str, Any]]:
    """Recent news for a ticker. Tavily (topic=news) -> DDG.news -> None."""
    key = _cache_key("news", ticker.upper(), max_items)
    hit = _cache_check("news", key)
    if hit is not None:
        return hit

    ticker = ticker.upper()
    # Trim any Yahoo suffix (".NS", ".BO") so searches look natural.
    plain_ticker = ticker.replace(".NS", "").replace(".BO", "")
    query = f"{plain_ticker} stock news latest"

    resp = None
    if is_tavily_available():
        resp = _tavily_search(query, max_results=max_items, topic="news")
    if resp is None and is_ddg_available():
        resp = _ddg_news(query, max_results=max_items)

    if resp is not None:
        _CACHE[key] = resp
    return resp


def search_historical_news(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    max_items: int = 8,
    extra_terms: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Historical news for ``ticker`` between ``start_date`` and ``end_date``.

    Primary use case: **claim tracking**. When a persona needs to verify
    whether a company delivered on a past statement, the agent first
    pulls the news from the window in which the claim was made
    (e.g. "Q1 2024"), then compares against more recent results.

    Dates must be ``YYYY-MM-DD`` strings. Tavily's ``finance`` topic
    biases the ranking toward earnings releases, analyst notes, 10-K
    excerpts, and investor presentations - the "management commentary"
    surface area that claim-tracking actually cares about.

    DuckDuckGo is used as a fallback, but DDG.news has no native date
    filter, so the fallback results may leak outside the requested
    window. The returned payload includes a ``window`` field the
    caller can inspect to detect this.
    """
    ticker = ticker.upper()
    plain_ticker = ticker.replace(".NS", "").replace(".BO", "")
    cache_key = _cache_key(
        "hist_news",
        plain_ticker,
        start_date,
        end_date,
        max_items,
        (extra_terms or "").strip().lower(),
    )
    hit = _cache_check("hist", cache_key)
    if hit is not None:
        return hit

    query_parts = [plain_ticker, "earnings guidance outlook"]
    if extra_terms:
        query_parts.append(extra_terms)
    query = " ".join(query_parts)

    resp = None
    if is_tavily_available():
        # Use topic='finance' (biases toward management + analyst content),
        # fall back to 'news' if finance returns nothing.
        resp = _tavily_search(
            query,
            max_results=max_items,
            topic="finance",
            start_date=start_date,
            end_date=end_date,
        )
        if resp is None or not resp.get("items"):
            resp = _tavily_search(
                query,
                max_results=max_items,
                topic="news",
                start_date=start_date,
                end_date=end_date,
            )
    if (resp is None or not resp.get("items")) and is_ddg_available():
        # DDG has no date filter; we annotate the result so the caller
        # knows the window isn't enforced.
        resp = _ddg_news(f"{query} {start_date[:4]}", max_results=max_items)
        if resp is not None:
            resp["window"] = {
                "start_date": start_date,
                "end_date": end_date,
                "enforced": False,  # DDG can't honour these
            }

    if resp is not None:
        _CACHE[cache_key] = resp
    return resp


def search_web(query: str, max_items: int = 5) -> Optional[Dict[str, Any]]:
    """General web search. Tavily -> DDG.text -> None."""
    key = _cache_key("web", query.strip().lower(), max_items)
    hit = _cache_check("web", key)
    if hit is not None:
        return hit

    resp = None
    if is_tavily_available():
        resp = _tavily_search(query, max_results=max_items, include_answer=True)
    if resp is None and is_ddg_available():
        resp = _ddg_text(query, max_results=max_items)

    if resp is not None:
        _CACHE[key] = resp
    return resp


def company_brief(ticker: str) -> Optional[Dict[str, Any]]:
    """One-shot company overview. Prefer Tavily's `include_answer` if available.

    Returns a dict ``{ticker, summary, sources[]}`` where ``summary`` is
    Tavily's pre-synthesised one-paragraph answer when available, otherwise
    a concatenation of the top DDG snippets.
    """
    key = _cache_key("brief", ticker.upper())
    hit = _cache_check("brief", key)
    if hit is not None:
        return hit

    ticker = ticker.upper()
    plain = ticker.replace(".NS", "").replace(".BO", "")
    query = (
        f"{plain} company overview business model revenue segments "
        f"competitive moat"
    )
    resp = None
    if is_tavily_available():
        resp = _tavily_search(query, max_results=5, include_answer=True)
    if resp is None and is_ddg_available():
        resp = _ddg_text(query, max_results=5)

    if resp is None:
        return None

    # Compose summary
    summary = resp.get("answer")
    if not summary:
        # Fall back to concatenating the first 2 snippets
        snippets = [(it.get("snippet") or "").strip() for it in resp.get("items", [])[:2]]
        summary = " ".join(s for s in snippets if s)
    sources = [
        {"title": it.get("title"), "url": it.get("url")}
        for it in resp.get("items", [])
        if it.get("url")
    ]
    out = {
        "ticker": ticker,
        "summary": summary or "",
        "sources": sources,
        "backend": resp.get("backend"),
        "fetched_at": resp.get("fetched_at"),
    }
    _CACHE[key] = out
    return out
