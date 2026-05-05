"""Indian NSE/BSE filings + aggregator wrapper.

The Indian counterpart to :mod:`src.mcp._sec_edgar`. Provides the
deep-research agent with the raw materials for claim-tracking on NSE/BSE
listed stocks:

* **BSE announcements** via the public ``api.bseindia.com`` JSON endpoint
  (Regulation 30 material events + Reg 33 quarterly results; equivalent to
  the US 8-K stream).
* **BSE attached PDFs** - earnings press releases, earnings call
  transcripts, investor presentations, board meeting outcomes, etc.
* **NSE announcements** via ``nseindia.com`` (Cloudflare-protected; uses
  ``curl_cffi`` with a Chrome TLS fingerprint to bypass the bot check).
* **Screener.in scrape** - 10y of ratios, segment breakdown, machine-
  generated pros/cons, and direct URLs to Annual Reports + concall
  transcripts. The single best free aggregator for Indian stocks.
* **PDF text extraction** via ``pdfplumber`` (layout-aware) with
  ``pypdf`` as a simpler fallback. Supports paging via an ``offset``
  argument, so a 300-page Annual Report doesn't need to be re-fetched
  for each chunk.

Design notes
------------
* Like the SEC worker, this module uses TTL caches so a typical
  agent-driven run hits each upstream once per ticker / URL and replays
  from memory thereafter.
* Ticker-to-BSE-scrip-code mapping is bootstrapped from a small
  hard-coded table of Nifty 50 names, then lazily extended by scraping
  Screener.in's company page header (which exposes both the NSE symbol
  and the BSE code). This avoids shipping a 5000-row master list.
* All three network clients (``httpx``, ``curl_cffi``) use polite
  user-agents. We rate-limit BSE + Screener at ~8 req/s to stay off
  their anti-abuse list.
"""
from __future__ import annotations

import io
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from cachetools import TTLCache


log = logging.getLogger("finai.indian_filings")


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_TIMEOUT = 20.0

_BSE_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/",
    "Origin": "https://www.bseindia.com",
}

_SCREENER_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------
_SCRIP_CACHE: TTLCache = TTLCache(maxsize=500, ttl=86400)      # 24h
_SCREENER_CACHE: TTLCache = TTLCache(maxsize=300, ttl=3600)    # 1h
_BSE_ANN_CACHE: TTLCache = TTLCache(maxsize=300, ttl=1800)     # 30 min
_NSE_ANN_CACHE: TTLCache = TTLCache(maxsize=300, ttl=1800)     # 30 min
_PDF_CACHE: TTLCache = TTLCache(maxsize=100, ttl=86400)        # 24h


# Minimum wait between successive network calls - defensive rate limiting.
_MIN_WAIT_S = 0.12  # ~8 req/s
_last_request_at: float = 0.0


def _polite_sleep() -> None:
    global _last_request_at
    now = time.time()
    wait = _MIN_WAIT_S - (now - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()


# ---------------------------------------------------------------------------
# Ticker -> BSE scrip code
# ---------------------------------------------------------------------------
# Seed list for the demo portfolio + major Nifty 50 members. Anything else
# gets looked up on the fly via the Screener.in company page header.
_BSE_SCRIP_SEED: Dict[str, int] = {
    # Demo portfolio
    "TCS": 532540,
    "INFY": 500209,
    "RELIANCE": 500325,
    "HDFCBANK": 500180,
    "ITC": 500875,
    # Other large-cap names likely to come up
    "WIPRO": 507685,
    "HCLTECH": 532281,
    "TECHM": 532755,
    "LT": 500510,
    "ICICIBANK": 532174,
    "AXISBANK": 532215,
    "SBIN": 500112,
    "KOTAKBANK": 500247,
    "BAJFINANCE": 500034,
    "SUNPHARMA": 524715,
    "DRREDDY": 500124,
    "CIPLA": 500087,
    "ASIANPAINT": 500820,
    "MARUTI": 532500,
    "TATAMOTORS": 500570,
    "M&M": 500520,
    "BHARTIARTL": 532454,
    "ONGC": 500312,
    "COALINDIA": 533278,
    "POWERGRID": 532898,
    "NTPC": 532555,
    "ADANIENT": 512599,
    "ADANIPORTS": 532921,
    "ULTRACEMCO": 532538,
    "GRASIM": 500300,
    "JSWSTEEL": 500228,
    "TATASTEEL": 500470,
    "HINDUNILVR": 500696,
    "NESTLEIND": 500790,
    "BRITANNIA": 500825,
    "TITAN": 500114,
    "BAJAJ-AUTO": 532977,
    "EICHERMOT": 505200,
    "HEROMOTOCO": 500182,
    "DIVISLAB": 532488,
    "APOLLOHOSP": 508869,
}


def _normalise(ticker: str) -> str:
    """Return the bare ticker (no exchange suffix, uppercase)."""
    return (ticker or "").upper().replace(".NS", "").replace(".BO", "").strip()


def ticker_to_bse_scrip(ticker: str) -> Optional[int]:
    """Resolve ``ticker`` to its BSE scrip code, scraping Screener.in if needed."""
    t = _normalise(ticker)
    if not t:
        return None
    if t in _BSE_SCRIP_SEED:
        return _BSE_SCRIP_SEED[t]
    # Cached lookups (same TTLCache so we don't re-scrape Screener per request)
    if t in _SCRIP_CACHE:
        return _SCRIP_CACHE[t]
    scrip = _scrape_screener_scrip(t)
    if scrip is not None:
        _SCRIP_CACHE[t] = scrip
    return scrip


def _scrape_screener_scrip(ticker: str) -> Optional[int]:
    """Look at the Screener.in company page header for the ``BSE: NNNNNN`` line."""
    snap = scrape_screener_company(ticker)
    if snap is None:
        return None
    return snap.get("bse_code")


# ---------------------------------------------------------------------------
# Screener.in scrape
# ---------------------------------------------------------------------------
_SCREENER_BSE_RE = re.compile(r"BSE:\s*</a?[^>]*>\s*(\d{6})|BSE:\s*(\d{6})", re.IGNORECASE)
_SCREENER_NSE_RE = re.compile(r"NSE:\s*</a?[^>]*>\s*([A-Z0-9\-&]{1,20})|NSE:\s*([A-Z0-9\-&]{1,20})", re.IGNORECASE)
_ANNUAL_REPORT_RE = re.compile(
    r"https?://[^\"']*AnnualReport[^\"']+\.pdf|"
    r"https?://[^\"']*annual[\-_]?report[^\"']*\.pdf|"
    r"/bseplus/AnnualReport/\d+/[^\"']+\.pdf",
    re.IGNORECASE,
)
_CONCALL_RE = re.compile(
    r"https?://[^\"']*(?:concall|earnings[-_]?call|transcript|investor[-_]?presentation)[^\"']*\.pdf",
    re.IGNORECASE,
)
_KEY_RATIO_RE = re.compile(
    r"<li[^>]*>\s*<span[^>]*class=\"name[^\"]*\"[^>]*>\s*([^<]+?)\s*</span>\s*"
    r"<span[^>]*class=\"(?:nowrap\s+)?value[^\"]*\"[^>]*>\s*"
    r"(?:<span[^>]*>([^<]*)</span>)?\s*([^<]*?)\s*</span>",
    re.DOTALL,
)
_PROS_CONS_RE = re.compile(
    r"<div[^>]*class=\"(pros|cons)\"[^>]*>.*?<ul>(.*?)</ul>",
    re.DOTALL,
)


def scrape_screener_company(ticker: str) -> Optional[Dict[str, Any]]:
    """Scrape the Screener.in consolidated-view page for ``ticker``.

    Returns a dict with::

        {
          "ticker": "TCS",
          "bse_code": 532540,
          "nse_symbol": "TCS",
          "name": "Tata Consultancy Services Ltd",
          "ratios": {"Market Cap": "\u20b9 9,18,434 Cr.", "Stock P/E": "17.6", ...},
          "pros": ["...", "..."],
          "cons": ["..."],
          "annual_report_urls": [...],
          "concall_urls": [...],
          "screener_url": "https://www.screener.in/company/TCS/consolidated/",
          "_as_of": unix_ts,
        }

    Returns ``None`` on network / parse failure.
    """
    t = _normalise(ticker)
    if not t:
        return None
    cache_key = t
    if cache_key in _SCREENER_CACHE:
        return _SCREENER_CACHE[cache_key]

    # Try consolidated first (what investors actually care about for
    # holdings), fall back to the default (standalone) view.
    urls = [
        f"https://www.screener.in/company/{t}/consolidated/",
        f"https://www.screener.in/company/{t}/",
    ]
    _polite_sleep()
    html = ""
    used_url = ""
    for url in urls:
        try:
            r = httpx.get(url, headers=_SCREENER_HEADERS, timeout=_TIMEOUT)
        except Exception as e:
            log.warning("Screener fetch failed for %s: %s", url, e)
            continue
        if r.status_code == 200 and "Company not found" not in r.text:
            html = r.text
            used_url = url
            break

    if not html:
        return None

    try:
        name = _extract_screener_name(html, t)
        bse_code = _extract_code(html, _SCREENER_BSE_RE)
        nse_symbol = _extract_symbol(html, _SCREENER_NSE_RE) or t
        ratios = _extract_screener_ratios(html)
        pros, cons = _extract_screener_pros_cons(html)
        ar_urls = _extract_screener_urls(html, _ANNUAL_REPORT_RE)
        concall_urls = _extract_screener_urls(html, _CONCALL_RE)
    except Exception as e:
        log.warning("Screener parse failed for %s: %s", t, e)
        return None

    out = {
        "ticker": t,
        "name": name,
        "bse_code": int(bse_code) if bse_code else None,
        "nse_symbol": nse_symbol,
        "ratios": ratios,
        "pros": pros,
        "cons": cons,
        "annual_report_urls": ar_urls,
        "concall_urls": concall_urls,
        "screener_url": used_url,
        "_as_of": int(time.time()),
    }
    _SCREENER_CACHE[cache_key] = out
    # Lazy-populate the BSE scrip cache so subsequent calls skip the
    # double-scrape.
    if out["bse_code"]:
        _SCRIP_CACHE[t] = out["bse_code"]
    return out


def _extract_screener_name(html: str, ticker: str) -> str:
    m = re.search(r"<h1[^>]*>\s*([^<]+?)\s*</h1>", html)
    return (m.group(1).strip() if m else ticker) or ticker


def _extract_code(html: str, pattern: re.Pattern) -> Optional[str]:
    m = pattern.search(html)
    if not m:
        return None
    # Two capturing groups because the regex handles "<a>BSE: 123</a>"
    # and "BSE: 123" in separate alternatives.
    return m.group(1) or m.group(2)


def _extract_symbol(html: str, pattern: re.Pattern) -> Optional[str]:
    return _extract_code(html, pattern)


def _extract_screener_ratios(html: str) -> Dict[str, str]:
    """Pull the small "Ratios" block at the top of a Screener page.

    Not a full parse - we just want the highlighted numbers the investor
    sees first (Market Cap, P/E, ROCE, ROE, Dividend Yield, Book Value).
    """
    # A resilient text-based extraction via BeautifulSoup is cleaner than
    # the regex, but we don't want a hard BS4 dep in this module.
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
    except ImportError:
        return {}

    out: Dict[str, str] = {}
    # The Screener header exposes ratios inside ``<ul class="...">
    # <li><span class="name">Market Cap</span><span class="value">...</span></li>``
    for li in soup.select("ul#top-ratios li, .company-ratios li, #ratio-strip li"):
        name_el = li.select_one(".name") or li.find("span")
        value_el = li.select_one(".value") or (li.find_all("span") or [None])[-1]
        if not name_el or not value_el:
            continue
        name = name_el.get_text(strip=True)
        value = value_el.get_text(" ", strip=True)
        if name and value:
            out[name] = value
    return out


def _extract_screener_pros_cons(html: str) -> Tuple[List[str], List[str]]:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
    except ImportError:
        return [], []
    pros: List[str] = []
    cons: List[str] = []
    pros_block = soup.find(class_=re.compile("pros", re.IGNORECASE))
    cons_block = soup.find(class_=re.compile("cons", re.IGNORECASE))
    if pros_block:
        pros = [li.get_text(" ", strip=True) for li in pros_block.select("li")]
    if cons_block:
        cons = [li.get_text(" ", strip=True) for li in cons_block.select("li")]
    return pros, cons


def _extract_screener_urls(html: str, pattern: re.Pattern) -> List[str]:
    """Extract unique URLs matching ``pattern`` from the Screener page."""
    seen: List[str] = []
    for m in pattern.finditer(html):
        url = m.group(0)
        # Absolutise BSE relative paths.
        if url.startswith("/"):
            url = "https://www.bseindia.com" + url
        if url not in seen:
            seen.append(url)
    return seen


# ---------------------------------------------------------------------------
# BSE announcements
# ---------------------------------------------------------------------------
def list_bse_announcements(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 25,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return recent BSE announcements for ``ticker``.

    Args:
        ticker:     NSE / BSE ticker; Indian suffix is stripped.
        start_date: ``YYYY-MM-DD`` lower bound (inclusive). Defaults to 1y ago.
        end_date:   ``YYYY-MM-DD`` upper bound (inclusive). Defaults to today.
        limit:      Max rows to return after filtering.
        categories: Optional lower-case substrings to match in the
                    ``subject`` field - e.g. ``["transcript", "press release"]``.

    Each returned dict has::

        {
          "ticker": "TCS",
          "scrip_cd": 532540,
          "news_id": "2a83f3bd-7652-4f44-a5db-3b172a778db6",
          "subject": "Announcement under Regulation 30 (LODR)-Earnings Call Transcript",
          "date": "2026-04-14T20:04:50.700",
          "announcement_type": "A",
          "pdf_url": "https://www.bseindia.com/xml-data/corpfiling/AttachHis/<uuid>.pdf",
        }
    """
    scrip = ticker_to_bse_scrip(ticker)
    if scrip is None:
        return []

    today = time.strftime("%Y%m%d")
    one_year_ago = time.strftime(
        "%Y%m%d",
        time.localtime(time.time() - 365 * 86400),
    )
    prev_date = (start_date or "").replace("-", "") or one_year_ago
    to_date = (end_date or "").replace("-", "") or today

    cache_key = f"{scrip}|{prev_date}|{to_date}|{limit}"
    if cache_key in _BSE_ANN_CACHE:
        rows = _BSE_ANN_CACHE[cache_key]
    else:
        _polite_sleep()
        url = (
            "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
            f"?pageno=1&strCat=-1&strPrevDate={prev_date}"
            f"&strScrip={scrip}&strSearch=P&strToDate={to_date}&strType=C"
        )
        try:
            r = httpx.get(url, headers=_BSE_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log.warning("BSE announcements fetch failed for %s (scrip %d): %s",
                        ticker, scrip, e)
            return []
        table = data.get("Table") or data.get("Data") or []
        rows = [_normalise_bse_row(row, scrip, _normalise(ticker)) for row in table]
        _BSE_ANN_CACHE[cache_key] = rows

    if categories:
        wanted = [c.lower() for c in categories]
        rows = [
            r for r in rows
            if any(w in (r.get("subject") or "").lower() for w in wanted)
        ]
    return rows[:limit]


def _normalise_bse_row(row: Dict[str, Any], scrip: int, ticker: str) -> Dict[str, Any]:
    attachment = row.get("ATTACHMENTNAME") or ""
    pdf_url = (
        f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{attachment}"
        if attachment
        else ""
    )
    return {
        "ticker": ticker,
        "scrip_cd": scrip,
        "news_id": row.get("NEWSID"),
        "subject": (row.get("NEWSSUB") or "").strip(),
        "date": row.get("DT_TM") or row.get("NEWS_DT"),
        "announcement_type": row.get("ANNOUNCEMENT_TYPE"),
        "critical": bool(row.get("CRITICALNEWS")),
        "pdf_url": pdf_url,
        "_source": "bse",
    }


# ---------------------------------------------------------------------------
# NSE announcements (Cloudflare-protected; curl_cffi required)
# ---------------------------------------------------------------------------
_nse_session = None  # Lazy init - spawns a TLS fingerprint first time it's used


def _get_nse_session():
    """Return a shared ``curl_cffi`` session impersonating Chrome.

    NSE actively rejects non-browser TLS fingerprints. ``impersonate='chrome'``
    mimics Chrome's ClientHello exactly, which gets us past the 403 wall.
    """
    global _nse_session
    if _nse_session is not None:
        return _nse_session
    try:
        from curl_cffi import requests  # type: ignore
    except ImportError:
        log.warning(
            "curl_cffi not installed; NSE fallback disabled. "
            "Install with 'pip install curl-cffi' to enable."
        )
        return None
    _nse_session = requests.Session(impersonate="chrome")
    try:
        _nse_session.get("https://www.nseindia.com", timeout=10)
    except Exception as e:
        log.warning("NSE session warm-up failed: %s", e)
    return _nse_session


def list_nse_announcements(
    ticker: str,
    limit: int = 25,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return recent NSE corporate announcements for ``ticker``.

    NSE's public API is Cloudflare-protected so this uses
    ``curl_cffi`` with a Chrome TLS fingerprint. A warm-up GET to the
    NSE homepage seeds the session cookies; subsequent API calls reuse
    them.

    The response schema is slightly different from BSE's - no scrip
    code, more text-heavy ``desc`` + ``sm_desc`` fields - so we
    normalise to the same shape as :func:`list_bse_announcements` for
    uniform downstream rendering.
    """
    t = _normalise(ticker)
    if not t:
        return []
    cache_key = f"{t}|{limit}"
    if cache_key in _NSE_ANN_CACHE:
        rows = _NSE_ANN_CACHE[cache_key]
    else:
        s = _get_nse_session()
        if s is None:
            return []
        _polite_sleep()
        url = (
            "https://www.nseindia.com/api/corporate-announcements"
            f"?index=equities&symbol={t}"
        )
        try:
            r = s.get(url, timeout=15)
            data = r.json()
        except Exception as e:
            log.warning("NSE announcements fetch failed for %s: %s", t, e)
            return []
        if not isinstance(data, list):
            return []
        rows = [_normalise_nse_row(row, t) for row in data]
        _NSE_ANN_CACHE[cache_key] = rows

    if categories:
        wanted = [c.lower() for c in categories]
        rows = [
            r for r in rows
            if any(w in (r.get("subject") or "").lower() for w in wanted)
        ]
    return rows[:limit]


def _normalise_nse_row(row: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "scrip_cd": None,
        "news_id": row.get("symbol") + "_" + (row.get("an_dt") or ""),
        "subject": (row.get("desc") or row.get("sm_desc") or "").strip(),
        "date": row.get("an_dt") or row.get("dt"),
        "announcement_type": None,
        "critical": False,
        "pdf_url": row.get("attchmntFile") or "",
        "_source": "nse",
    }


# ---------------------------------------------------------------------------
# PDF fetch + text extraction
# ---------------------------------------------------------------------------
def fetch_pdf_text(
    url: str,
    max_chars: int = 40000,
    offset: int = 0,
) -> Optional[Dict[str, Any]]:
    """Download a PDF and return cleaned text (layout-preserving when possible).

    Args:
        url:       PDF URL - typically a BSE ``AttachHis`` URL or a
                   company IR-page PDF. HTTPS only.
        max_chars: Cap on returned text slice size.
        offset:    Start character of the returned slice, relative to
                   the fully-extracted text. Lets the agent page
                   through long Annual Reports without re-downloading
                   the PDF (cached after the first fetch).

    Returns a dict ``{url, total_chars, offset, chars_returned, text,
    end_of_document, pages, extractor}``, or ``None`` on fetch failure.
    """
    if not (url or "").lower().startswith(("http://", "https://")):
        return None
    if not url.lower().endswith(".pdf") and "pdf" not in url.lower():
        # Defensive: only serve PDFs through this tool
        # (HTML pages should use fetch_sec_document or a web-search tool)
        log.info("fetch_pdf_text: non-PDF URL %s - proceeding anyway", url)

    # Fetch the raw PDF bytes once, cache the **full** extracted text.
    if url in _PDF_CACHE:
        full_text = _PDF_CACHE[url]["text"]
        pages = _PDF_CACHE[url]["pages"]
        extractor = _PDF_CACHE[url]["extractor"]
    else:
        _polite_sleep()
        try:
            r = httpx.get(
                url,
                headers={"User-Agent": _BROWSER_UA},
                timeout=60,
                follow_redirects=True,
            )
            r.raise_for_status()
            blob = r.content
        except Exception as e:
            log.warning("PDF fetch failed for %s: %s", url, e)
            return None

        full_text, pages, extractor = _extract_pdf_text(blob)
        if full_text is None:
            return None
        _PDF_CACHE[url] = {"text": full_text, "pages": pages, "extractor": extractor}

    # Slice + package.
    if offset < 0:
        offset = 0
    total = len(full_text)
    if offset >= total:
        snippet = ""
    elif max_chars <= 0:
        snippet = full_text[offset:]
    else:
        snippet = full_text[offset : offset + max_chars]

    return {
        "url": url,
        "total_chars": total,
        "offset": offset,
        "chars_returned": len(snippet),
        "text": snippet,
        "end_of_document": len(snippet) == 0,
        "pages": pages,
        "extractor": extractor,
        "_source": "pdf",
    }


def _extract_pdf_text(blob: bytes) -> Tuple[Optional[str], int, str]:
    """Extract text from PDF bytes using pdfplumber, falling back to pypdf.

    Returns ``(full_text, page_count, extractor_name)``. ``(None, 0, "")``
    on complete failure.
    """
    # Primary: pdfplumber (preserves table structure better for financials)
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(blob)) as pdf:
            chunks: List[str] = []
            for p in pdf.pages:
                try:
                    txt = p.extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    chunks.append(txt)
            if chunks:
                full = _normalise_pdf_text("\n\n".join(chunks))
                return full, len(pdf.pages), "pdfplumber"
    except Exception as e:
        log.info("pdfplumber failed (%s); falling back to pypdf", e)

    # Fallback: pypdf (faster, less layout-aware)
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(blob))
        chunks = []
        for page in reader.pages:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        full = _normalise_pdf_text("\n\n".join(chunks))
        return full, len(reader.pages), "pypdf"
    except Exception as e:
        log.warning("pypdf also failed on PDF bytes: %s", e)
        return None, 0, ""


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _normalise_pdf_text(text: str) -> str:
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
def cache_stats() -> Dict[str, Any]:
    return {
        "scrip_cache": {"size": len(_SCRIP_CACHE), "ttl_seconds": _SCRIP_CACHE.ttl},
        "screener": {"size": len(_SCREENER_CACHE), "ttl_seconds": _SCREENER_CACHE.ttl},
        "bse_ann": {"size": len(_BSE_ANN_CACHE), "ttl_seconds": _BSE_ANN_CACHE.ttl},
        "nse_ann": {"size": len(_NSE_ANN_CACHE), "ttl_seconds": _NSE_ANN_CACHE.ttl},
        "pdf": {"size": len(_PDF_CACHE), "ttl_seconds": _PDF_CACHE.ttl},
    }
