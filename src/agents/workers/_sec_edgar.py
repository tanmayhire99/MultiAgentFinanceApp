"""SEC EDGAR JSON API wrapper (free, no API key).

Backs the deep stock research flow. Covers **US** filings only - for
NSE / BSE tickers the caller should fall back to Tavily historical news.

What's here
-----------
* ``ticker_to_cik(ticker)``       - resolve a ticker to its Central Index Key
* ``list_filings(ticker, ...)``   - recent filings (10-K / 10-Q / 8-K / ...)
* ``fetch_document_text(url)``    - download + clean a filing document
* ``find_earnings_8k(ticker, ..)`` - convenience wrapper for Item 2.02 8-Ks

Rate limiting
-------------
SEC's published guidance is 10 req/s with a descriptive ``User-Agent``
header. We honour that with a small sleep between successive network
calls from this module. The results are additionally TTL-cached so a
typical agent-driven run hits the network just once per ticker/filing.

Attribution
-----------
Data provided by the U.S. Securities and Exchange Commission.
See https://www.sec.gov/os/webmaster-faq#developers for terms.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from cachetools import TTLCache


log = logging.getLogger("finai.sec_edgar")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# SEC asks for a descriptive User-Agent including an email. We use a
# generic "demo" address; for production you would put a real contact.
_USER_AGENT = "FinAI Demo research@finai.example"
_TIMEOUT = 20.0

# Ticker -> CIK map and filings metadata change slowly; filing documents
# never change once published. Cache them for long periods.
_TICKER_MAP_CACHE: TTLCache = TTLCache(maxsize=1, ttl=86400)   # 24h
_FILINGS_CACHE: TTLCache = TTLCache(maxsize=500, ttl=3600)     # 1h
_DOCUMENT_CACHE: TTLCache = TTLCache(maxsize=200, ttl=86400)   # 24h

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


# Minimum wait between successive network calls - defensive rate limiting.
_MIN_WAIT_S = 0.12  # ~8 req/s, well under the 10 req/s soft limit
_last_request_at: float = 0.0


def _polite_sleep() -> None:
    """Sleep just enough to stay under SEC's published rate cap."""
    global _last_request_at
    now = time.time()
    wait = _MIN_WAIT_S - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


# --------------------------------------------------------------------------
# Ticker -> CIK
# --------------------------------------------------------------------------
def _fetch_ticker_map() -> Dict[str, Dict[str, Any]]:
    """Return ``{TICKER: {cik: int, name: str}}`` from SEC's canonical map."""
    cached = _TICKER_MAP_CACHE.get("map")
    if cached is not None:
        return cached
    _polite_sleep()
    try:
        r = httpx.get(
            _TICKER_MAP_URL,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("SEC ticker-map fetch failed: %s", e)
        return {}
    # The canonical file is shaped ``{"0": {"cik_str": 320193, "ticker": "AAPL",
    # "title": "Apple Inc."}, ...}``. Flatten it to ``{AAPL: {cik: 320193, ...}}``.
    normalized: Dict[str, Dict[str, Any]] = {}
    for _, row in data.items():
        t = (row.get("ticker") or "").upper()
        if not t:
            continue
        try:
            normalized[t] = {"cik": int(row["cik_str"]), "name": row.get("title") or t}
        except (KeyError, TypeError, ValueError):
            continue
    _TICKER_MAP_CACHE["map"] = normalized
    log.info("SEC ticker-map loaded: %d symbols", len(normalized))
    return normalized


def ticker_to_cik(ticker: str) -> Optional[int]:
    """Resolve ``ticker`` to its 10-digit CIK, or ``None`` if not on SEC.

    Indian-exchange suffixes (``.NS``, ``.BO``) are stripped before
    lookup, but tickers that aren't US-listed will simply return
    ``None``. The caller can then fall back to non-SEC sources.
    """
    t = (ticker or "").upper().replace(".NS", "").replace(".BO", "").strip()
    if not t:
        return None
    return (_fetch_ticker_map().get(t) or {}).get("cik")


def _cik_padded(cik: int) -> str:
    return f"{cik:010d}"


# --------------------------------------------------------------------------
# Filings list
# --------------------------------------------------------------------------
def list_filings(
    ticker: str,
    form_types: Optional[List[str]] = None,
    limit: int = 5,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return recent filings for ``ticker``, optionally filtered.

    Args:
        ticker:     e.g. ``"TSLA"``, ``"NVDA"``. ``.NS`` / ``.BO`` suffixes
                    are stripped but non-US tickers return ``[]``.
        form_types: e.g. ``["10-K", "10-Q", "8-K"]``. ``None`` = no filter.
        limit:      max number of filings to return (after filtering).
        since:      ``YYYY-MM-DD`` lower bound on the filing date.

    Returns list of dicts::

        {
            "ticker": "NVDA",
            "cik": 1045810,
            "company_name": "NVIDIA CORP",
            "form": "8-K",
            "filing_date": "2026-02-25",
            "report_date": "2026-02-25",
            "accession": "0001045810-26-000019",
            "primary_doc": "nvda-20260225.htm",
            "items": "2.02,9.01",
            "report_url": "https://www.sec.gov/Archives/edgar/data/1045810/.../nvda-20260225.htm",
            "index_url":  "https://www.sec.gov/cgi-bin/browse-edgar?...",
        }
    """
    cik = ticker_to_cik(ticker)
    if cik is None:
        return []

    key = f"{cik}|{','.join(sorted(form_types or []))}|{limit}|{since or ''}"
    cached = _FILINGS_CACHE.get(key)
    if cached is not None:
        return cached

    _polite_sleep()
    url = f"https://data.sec.gov/submissions/CIK{_cik_padded(cik)}.json"
    try:
        r = httpx.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("SEC submissions fetch failed for %s: %s", ticker, e)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    report_dates = recent.get("reportDate") or []
    accessions = recent.get("accessionNumber") or []
    primary_docs = recent.get("primaryDocument") or []
    items_list = recent.get("items") or []

    form_set = set(form_types or [])

    results: List[Dict[str, Any]] = []
    for i in range(len(forms)):
        if form_set and forms[i] not in form_set:
            continue
        if since and filing_dates[i] < since:
            continue
        accession = accessions[i]
        accession_no_dashes = accession.replace("-", "")
        primary_doc = primary_docs[i] if i < len(primary_docs) else ""
        items = items_list[i] if i < len(items_list) else ""
        report_date = report_dates[i] if i < len(report_dates) else ""
        results.append(
            {
                "ticker": ticker.upper(),
                "cik": cik,
                "company_name": data.get("name"),
                "form": forms[i],
                "filing_date": filing_dates[i],
                "report_date": report_date,
                "accession": accession,
                "primary_doc": primary_doc,
                "items": items,
                "report_url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{accession_no_dashes}/{primary_doc}"
                ),
                "index_url": (
                    f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
                    f"&type={forms[i]}&dateb=&owner=include&count=40"
                ),
            }
        )
        if len(results) >= limit:
            break

    _FILINGS_CACHE[key] = results
    return results


def find_earnings_8k(
    ticker: str,
    since: Optional[str] = None,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    """Return 8-K filings that contain **Item 2.02** (Results of Operations).

    These are the canonical "earnings release" 8-Ks. The body of the
    primary document typically contains the press-release table; the
    accompanying Exhibit 99.1 contains the full earnings release
    narrative (and sometimes the call transcript). For v1 we only
    return the primary document URL; the agent can fetch the index
    page separately if it wants the 99.1.
    """
    eights = list_filings(ticker, form_types=["8-K"], limit=100, since=since)
    filtered = [
        row for row in eights
        if "2.02" in (row.get("items") or "")
    ]
    return filtered[:limit]


# --------------------------------------------------------------------------
# Document fetch
# --------------------------------------------------------------------------
_SCRIPT_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINES_RE = re.compile(r"\n\s*\n+")


def _strip_html(html: str) -> str:
    """Best-effort HTML -> plain text.

    Tries ``bs4`` first because it's vastly more robust on the
    nested-table-heavy filings the SEC publishes, then falls back to
    pure-regex stripping if ``bs4`` is not installed.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        text = _SCRIPT_RE.sub(" ", html)
        text = _TAG_RE.sub("\n", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def fetch_document_text(
    url: str,
    max_chars: int = 40000,
    offset: int = 0,
) -> Optional[str]:
    """Download an EDGAR document and return cleaned plain text.

    Args:
        url: A filing document URL, typically from
             :func:`list_filings` ``report_url``.
        max_chars: Truncate to this size so LLM callers never blow the
            context window. The canonical 10-K runs 200-400k characters
            of HTML; 40k is enough to capture MD&A + guidance tables
            without forcing the caller to chunk. Set to 0 for no cap.
        offset: Start character of the returned slice, relative to the
            cleaned text. Lets the agent "page" through a long document
            without re-fetching it (the full text is cached, so the
            network hit only happens on the first call).

    Returns ``None`` on network failure, else the cleaned text slice.
    If ``offset`` is beyond the document length, returns ``""``.
    """
    cached = _DOCUMENT_CACHE.get(url)
    if cached is not None:
        return _slice(cached, offset, max_chars)

    _polite_sleep()
    try:
        r = httpx.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        r.raise_for_status()
        html = r.text
    except Exception as e:
        log.warning("SEC document fetch failed for %s: %s", url, e)
        return None

    text = _strip_html(html)
    _DOCUMENT_CACHE[url] = text
    return _slice(text, offset, max_chars)


def _slice(text: str, offset: int, max_chars: int) -> str:
    """Return ``text[offset:offset+max_chars]`` with clamping + bounds checks."""
    if offset < 0:
        offset = 0
    if offset >= len(text):
        return ""
    if max_chars <= 0:
        return text[offset:]
    return text[offset : offset + max_chars]


# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------
def cache_stats() -> Dict[str, Any]:
    return {
        "ticker_map": {
            "size": len(_TICKER_MAP_CACHE),
            "ttl_seconds": _TICKER_MAP_CACHE.ttl,
        },
        "filings": {
            "size": len(_FILINGS_CACHE),
            "maxsize": _FILINGS_CACHE.maxsize,
            "ttl_seconds": _FILINGS_CACHE.ttl,
        },
        "documents": {
            "size": len(_DOCUMENT_CACHE),
            "maxsize": _DOCUMENT_CACHE.maxsize,
            "ttl_seconds": _DOCUMENT_CACHE.ttl,
        },
    }
