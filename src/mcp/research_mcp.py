"""Research MCP worker server (live web search + curated fallbacks).

Tools:
    - ``list_supported_tickers``    -> tickers with curated fixture content
    - ``search_news``               -> recent news for a ticker (Tavily / DDG / fixture)
    - ``search_historical_news``    -> news in a specific ``start_date``/``end_date`` window (claim tracking)
    - ``search_web``                -> general web search (Tavily / DDG)
    - ``get_company_brief``         -> one-paragraph company overview (Tavily / DDG)
    - ``get_key_catalysts``         -> curated forward-looking catalysts (fixture)
    - ``get_analyst_takes``         -> curated sell-side ratings + theses (fixture)
    - ``get_sec_filings``           -> SEC EDGAR filings list (10-K / 10-Q / 8-K / ...)
    - ``fetch_sec_document``        -> download + clean a filing document
    - ``extract_forward_claims``    -> LLM-extract forward-looking claims from text
    - ``compare_claim_to_reality``  -> LLM-diff a single claim against recent evidence
    - ``get_indian_filings``        -> BSE / NSE corporate announcements for NSE/BSE tickers
    - ``fetch_indian_document``     -> download + clean a PDF from BSE / IR (Annual Reports, concall transcripts)
    - ``get_screener_snapshot``     -> 10y ratios + Annual Report + concall URLs from Screener.in
    - ``get_indian_concall_urls``   -> shortcut to just the concall transcript URLs
    - ``get_indian_annual_reports`` -> shortcut to just the Annual Report PDF URLs

Backend precedence for live tools: Tavily -> DuckDuckGo -> (fixture for news).
SEC tools use the free ``data.sec.gov`` JSON API with a descriptive
User-Agent; no API key required. Indian tools hit BSE's public
``api.bseindia.com`` JSON endpoint + scrape Screener.in + use
``curl_cffi`` with a Chrome TLS fingerprint for NSE's Cloudflare-
protected APIs.

See :mod:`src.mcp._research`, :mod:`_sec_edgar`,
:mod:`_indian_filings`, and :mod:`_claims` for implementation details.

Run as::

    python -m src.mcp.research_mcp
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from . import _claims, _indian_filings, _research, _sec_edgar
from ._fixtures import load_fixture, lookup


_FIXTURES: Dict[str, Any] = load_fixture("research_snippets")


mcp = FastMCP(
    name="research",
    instructions=(
        "Provides recent news, key catalysts, analyst takes, open-ended web "
        "search, and company briefs. Live tools use Tavily when "
        "TAVILY_API_KEY is set, else fall back to DuckDuckGo; curated "
        "fixtures are the last resort for news and hold the analyst/"
        "catalyst narratives that are editorial rather than searchable."
    ),
)


def _fixture_entry(ticker: str) -> Dict[str, Any]:
    try:
        return lookup(_FIXTURES, ticker)
    except KeyError:
        return {}


def _available_tickers() -> List[str]:
    return sorted(k for k in _FIXTURES.keys() if not k.startswith("_"))


def _fixture_news(ticker: str, max_items: int) -> Dict[str, Any]:
    entry = _fixture_entry(ticker)
    items = (entry.get("news") or [])[: max(0, int(max_items))]
    return {
        "ticker": ticker.upper(),
        "backend": "fixture",
        "news": [
            {
                "headline": n.get("headline"),
                "url": None,
                "date": n.get("date"),
                "sentiment": n.get("sentiment"),
                "snippet": n.get("summary"),
            }
            for n in items
        ],
        "_source": "fixture:research_snippets",
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool
def list_supported_tickers() -> Dict[str, Any]:
    """Return tickers that have curated catalyst/analyst content in the fixture.

    Live tools (``search_news``, ``search_web``, ``get_company_brief``) work
    for *any* ticker or query that the upstream search backend supports.
    """
    return {
        "fixture_tickers": _available_tickers(),
        "active_backends": _research.active_backends(),
        "live_supported": (
            "any ticker for news/brief; any free-form query for search_web"
        ),
    }


@mcp.tool
def search_news(
    ticker: str = "",
    max_items: int = 3,
    query: str = "",
    start_date: str = "",
    end_date: str = "",
) -> Dict[str, Any]:
    """Return recent news items for a TICKER (live web search + cache).

    Primary call pattern: ``search_news(ticker="NVDA", max_items=3)``.
    Pass a stock ticker symbol, NOT a free-text question.

    LLM-mistake forgiveness:

    - If only ``query`` is provided (no ticker), the call is routed to
      :func:`search_web`.
    - If ``start_date`` / ``end_date`` (``YYYY-MM-DD``) are provided,
      the call is routed to :func:`search_historical_news` so the date
      filter actually does something. Prefer calling
      ``search_historical_news`` directly for date-filtered queries.

    Falls back to curated fixture snippets if both live backends fail.
    """
    # Route to historical when date bounds are given - otherwise the
    # agent's date filter would be silently ignored.
    if ticker and (start_date or end_date):
        # Default the missing bound to "today" or "1y ago" as appropriate.
        import time as _t
        today = _t.strftime("%Y-%m-%d")
        one_year_ago = _t.strftime(
            "%Y-%m-%d", _t.localtime(_t.time() - 365 * 86400)
        )
        live = _research.search_historical_news(
            ticker,
            start_date=start_date or one_year_ago,
            end_date=end_date or today,
            max_items=max_items,
        )
        if live is None:
            return {
                "ticker": ticker.upper(),
                "start_date": start_date,
                "end_date": end_date,
                "items": [],
                "_source": "none",
                "error": "No search backend available for historical news.",
            }
        return {
            "ticker": ticker.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "backend": live.get("backend"),
            "window": live.get("window"),
            "items": [
                {
                    "title": it.get("title"),
                    "url": it.get("url"),
                    "date": it.get("published"),
                    "source": it.get("source"),
                    "snippet": it.get("snippet"),
                }
                for it in live.get("items", [])[:max_items]
            ],
            "_source": f"live:{live.get('backend')}",
            "_as_of": live.get("fetched_at"),
            "note": (
                "You called search_news with date bounds; routed to "
                "search_historical_news automatically. Use that directly "
                "next time for explicit date-filtered searches."
            ),
        }
    # Forgive LLM confusion: if only ``query`` was provided, run a
    # free-text search directly via the research backend instead of
    # raising a validation error. We call ``_research`` rather than the
    # decorated ``search_web`` tool to avoid any MCP-wrapper subtlety.
    if not ticker and query:
        live = _research.search_web(query, max_items=max_items)
        if live is None:
            return {
                "query": query,
                "results": [],
                "_source": "none",
                "error": "No search backend available and no ticker supplied.",
            }
        return {
            "query": query,
            "backend": live.get("backend"),
            "results": [
                {
                    "title": it.get("title"),
                    "url": it.get("url"),
                    "snippet": it.get("snippet"),
                    "score": it.get("score"),
                }
                for it in live.get("items", [])[:max_items]
            ],
            "_source": f"live:{live.get('backend')}",
            "_as_of": live.get("fetched_at"),
            "note": (
                "You called search_news without a ticker; routed to "
                "search_web automatically. Use search_web directly next "
                "time for free-text queries."
            ),
        }
    if not ticker:
        return {
            "ticker": "",
            "error": "search_news requires a ticker (e.g. ticker='NVDA').",
            "hint": "For free-text search use search_web(query=...).",
            "_source": "none",
        }

    live = _research.search_news(ticker, max_items=max_items)
    if live is not None and live.get("items"):
        return {
            "ticker": ticker.upper(),
            "backend": live.get("backend"),
            "news": [
                {
                    "headline": it.get("title"),
                    "url": it.get("url"),
                    "date": it.get("published"),
                    "source": it.get("source"),
                    "snippet": it.get("snippet"),
                }
                for it in live.get("items", [])[:max_items]
            ],
            "_source": f"live:{live.get('backend')}",
            "_as_of": live.get("fetched_at"),
        }
    return _fixture_news(ticker, max_items)


@mcp.tool
def search_web(query: str, max_items: int = 5) -> Dict[str, Any]:
    """General open-ended web search (Tavily / DDG).

    Use this for macro, sector, or thematic questions that aren't tied to a
    specific ticker - e.g. "impact of US-China AI chip export controls on
    semiconductor capex".
    """
    live = _research.search_web(query, max_items=max_items)
    if live is None:
        return {
            "query": query,
            "results": [],
            "_source": "none",
            "error": (
                "No search backend available. Set TAVILY_API_KEY or install "
                "ddgs to enable live web search."
            ),
        }
    return {
        "query": query,
        "backend": live.get("backend"),
        "answer": live.get("answer"),
        "results": [
            {
                "title": it.get("title"),
                "url": it.get("url"),
                "snippet": it.get("snippet"),
                "score": it.get("score"),
            }
            for it in live.get("items", [])[:max_items]
        ],
        "_source": f"live:{live.get('backend')}",
        "_as_of": live.get("fetched_at"),
    }


@mcp.tool
def get_company_brief(ticker: str) -> Dict[str, Any]:
    """Return a concise one-paragraph overview of the company behind ``ticker``.

    Uses Tavily's pre-synthesised answer when available (free tier); falls
    back to a concatenation of the top DuckDuckGo snippets otherwise.
    """
    live = _research.company_brief(ticker)
    if live is None:
        return {
            "ticker": ticker.upper(),
            "summary": "",
            "sources": [],
            "_source": "none",
            "error": "No search backend available.",
        }
    return {
        "ticker": live.get("ticker", ticker.upper()),
        "summary": live.get("summary") or "(no summary)",
        "sources": live.get("sources", []),
        "_source": f"live:{live.get('backend')}",
        "_as_of": live.get("fetched_at"),
    }


@mcp.tool
def get_key_catalysts(ticker: str) -> Dict[str, Any]:
    """Return the forward-looking catalysts that analysts are watching.

    Curated fixture content - these are short editorial statements
    (e.g. "Rubin volume ramp through 2026 H2") rather than metrics that
    can be searched for on the web.
    """
    entry = _fixture_entry(ticker)
    return {
        "ticker": ticker.upper(),
        "catalysts": entry.get("key_catalysts", []),
        "_source": "fixture:research_snippets",
    }


@mcp.tool
def get_analyst_takes(ticker: str) -> Dict[str, Any]:
    """Return curated sell-side ratings, price targets, and one-line theses.

    Curated fixture content - sourced from research house notes rather
    than live scraping, so values are stable for the demo.
    """
    entry = _fixture_entry(ticker)
    return {
        "ticker": ticker.upper(),
        "analyst_takes": entry.get("analyst_takes", []),
        "_source": "fixture:research_snippets",
    }


# ---------------------------------------------------------------------------
# Deep-research tools: historical news + SEC filings + claim tracking
# ---------------------------------------------------------------------------
@mcp.tool
def search_historical_news(
    ticker: str,
    start_date: str,
    end_date: str,
    max_items: int = 8,
    extra_terms: str = "",
) -> Dict[str, Any]:
    """Return news articles published between ``start_date`` and ``end_date``.

    ``start_date`` / ``end_date`` are ``YYYY-MM-DD`` strings. Useful for
    the claim-tracking flow: first pull news from the window the claim
    was made (e.g. Q1 2024), then pull the most recent window to
    check whether it was delivered.

    Backends, in order: Tavily (topic=finance with date filter ->
    topic=news with date filter), DuckDuckGo (no date filter; the
    returned payload's ``window.enforced`` field is ``false`` when the
    fallback was used).
    """
    live = _research.search_historical_news(
        ticker,
        start_date=start_date,
        end_date=end_date,
        max_items=max_items,
        extra_terms=(extra_terms or None),
    )
    if live is None:
        return {
            "ticker": ticker.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "items": [],
            "_source": "none",
            "error": "No search backend available.",
        }
    return {
        "ticker": ticker.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "backend": live.get("backend"),
        "window": live.get("window"),
        "items": [
            {
                "title": it.get("title"),
                "url": it.get("url"),
                "date": it.get("published"),
                "source": it.get("source"),
                "snippet": it.get("snippet"),
            }
            for it in live.get("items", [])[:max_items]
        ],
        "_source": f"live:{live.get('backend')}",
        "_as_of": live.get("fetched_at"),
    }


@mcp.tool
def get_sec_filings(
    ticker: str,
    form_types: Optional[List[str]] = None,
    limit: int = 5,
    since: str = "",
) -> Dict[str, Any]:
    """Return recent SEC EDGAR filings for ``ticker``.

    ``form_types`` is an optional list like ``["10-K", "10-Q", "8-K"]``.
    ``since`` is an optional ``YYYY-MM-DD`` lower bound on filing date.

    Covers US SEC-registered entities including NYSE-listed ADRs of
    foreign issuers (e.g. INFY, HDB). For NSE-only tickers (TCS,
    RELIANCE) this returns an empty list; callers should fall back to
    ``search_historical_news`` in that case.

    For earnings releases specifically, prefer 8-Ks with items
    containing ``"2.02"`` (Results of Operations). The primary document
    is linked via ``report_url``.
    """
    rows = _sec_edgar.list_filings(
        ticker,
        form_types=form_types or None,
        limit=limit,
        since=(since or None),
    )
    if not rows:
        cik = _sec_edgar.ticker_to_cik(ticker)
        return {
            "ticker": ticker.upper(),
            "filings": [],
            "_source": "sec-edgar",
            "error": (
                "Ticker not found on SEC EDGAR (probably not US-registered)."
                if cik is None
                else "No filings matched the form_types / since filter."
            ),
        }
    return {
        "ticker": ticker.upper(),
        "cik": rows[0]["cik"],
        "company_name": rows[0]["company_name"],
        "filings": rows,
        "_source": "sec-edgar",
    }


@mcp.tool
def fetch_sec_document(
    url: str,
    max_chars: int = 40000,
    offset: int = 0,
) -> Dict[str, Any]:
    """Download an SEC EDGAR document URL and return cleaned plain text.

    Pass a ``report_url`` from :func:`get_sec_filings`. Returns a text
    blob with HTML tags stripped and whitespace normalised.

    ``max_chars`` caps the slice size so the caller doesn't blow the
    LLM context window. A 10-K is typically 200-400k characters; 40k
    captures MD&A + guidance sections but not the full risk factors or
    appendices.

    ``offset`` lets you page through a long document by fetching the
    next slice without re-downloading: the full text is cached on the
    first call, so subsequent calls with a different ``offset`` on the
    same URL are free. Returns an empty ``text`` when ``offset``
    exceeds the document length (useful as a termination signal).
    """
    if not (url or "").startswith("https://www.sec.gov/"):
        return {
            "url": url,
            "text": "",
            "_source": "sec-edgar",
            "error": "URL must be a https://www.sec.gov/ filing document URL.",
        }
    text = _sec_edgar.fetch_document_text(url, max_chars=max_chars, offset=offset)
    if text is None:
        return {
            "url": url,
            "text": "",
            "_source": "sec-edgar",
            "error": "Document fetch failed (network error or invalid URL).",
        }
    return {
        "url": url,
        "offset": offset,
        "chars_returned": len(text),
        "text": text,
        "end_of_document": len(text) == 0,
        "_source": "sec-edgar",
    }


@mcp.tool
async def extract_forward_claims(
    text: str,
    source_label: str = "",
    max_claims: int = 15,
) -> Dict[str, Any]:
    """Use an LLM to extract FORWARD-LOOKING CLAIMS from a document.

    Returns a list of ``{claim_text, metric, target_value, target_date,
    confidence, source}`` objects. ``source_label`` is attached to
    every extracted claim so downstream rendering can attribute them
    back to their origin (e.g. ``"NVDA Q1 FY25 earnings call"``).

    Only specific, testable commitments are extracted. Hedging
    language ("we aim to", "potentially") is deliberately excluded.
    See :mod:`src.mcp._claims` for the extraction prompt.

    If no claims are found, returns ``{"claims": []}`` with
    ``ok: true``. Errors return ``{"claims": [], "error": ...}``.
    """
    try:
        claims = await _claims.extract_forward_claims(
            text,
            source_label=source_label,
            max_claims=max_claims,
        )
    except Exception as e:
        return {
            "source_label": source_label,
            "claims": [],
            "ok": False,
            "error": f"Claim extraction failed: {e}",
        }
    return {
        "source_label": source_label,
        "claims": claims,
        "count": len(claims),
        "ok": True,
    }


@mcp.tool
async def compare_claim_to_reality(
    claim_text: str,
    claim_metric: str,
    claim_target_value: str,
    claim_target_date: str,
    actuals_context: str,
    claim_source: str = "",
) -> Dict[str, Any]:
    """Use an LLM to produce a verdict on whether a past claim was met.

    Pass the individual claim fields (not a nested dict) so the
    upstream deep-research agent can compose the call directly from a
    previously-extracted claim without needing to serialise JSON.

    ``actuals_context`` should contain the recent evidence the verdict
    should be grounded in: concatenate the most relevant news
    snippets, filing excerpts, and current metrics into a single
    string. 5-8k characters is a sensible target.

    Returns a verdict dict with ``verdict`` in
    ``{met, missed, partial, pending, unknowable}``, plus
    ``variance_pct``, ``variance_time``, ``explanation``,
    ``confidence``, and ``evidence_snippets`` quoting the actuals.
    """
    claim = {
        "claim_text": claim_text,
        "metric": claim_metric,
        "target_value": claim_target_value,
        "target_date": claim_target_date,
        "source": claim_source,
    }
    try:
        verdict = await _claims.compare_claim_to_reality(claim, actuals_context)
    except Exception as e:
        return {
            "claim": claim,
            "verdict": "unknowable",
            "explanation": f"Claim-compare failed: {e}",
            "confidence": "low",
            "evidence_snippets": [],
        }
    return verdict


# ---------------------------------------------------------------------------
# Indian NSE/BSE filings - the claim-tracking surface for NSE/BSE tickers
# ---------------------------------------------------------------------------
@mcp.tool
def get_indian_filings(
    ticker: str,
    source: str = "bse",
    start_date: str = "",
    end_date: str = "",
    limit: int = 20,
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return recent NSE/BSE corporate announcements for an Indian ticker.

    ``source`` selects which exchange to query:

    - ``"bse"`` (default) - BSE public JSON API. Richer, date-filterable,
      friendlier. Covers Reg 30 material events + Reg 33 quarterly
      results + audited annual results. This is the first-choice source.
    - ``"nse"``          - NSE API. Needs ``curl_cffi`` for Cloudflare
      bypass; no date filter on the upstream API, so use ``limit`` to
      cap. Use when BSE is down or misses a filing.

    ``start_date`` / ``end_date`` (``YYYY-MM-DD``) are honoured only on
    BSE; on NSE they're silently ignored.

    ``categories`` is an optional list of lower-case substrings to
    match in the filing subject. Typical values for claim tracking::

        ["transcript"]                 -> earnings-call transcripts only
        ["press release"]              -> quarterly press releases
        ["investor presentation"]      -> slide decks with guidance tables
        ["outcome of board meeting"]   -> results + dividend announcements
        ["regulation 30"]              -> all material events

    Each returned filing has ``{ticker, scrip_cd, news_id, subject, date,
    pdf_url, ...}``; pass ``pdf_url`` to :func:`fetch_indian_document` to
    download + parse the content.

    Works for US ADR tickers too (they have BSE/NSE codes), but SEC
    tools are the primary path for those.
    """
    src = (source or "bse").lower()
    if src not in ("bse", "nse"):
        return {
            "ticker": ticker.upper(),
            "filings": [],
            "error": f"Unknown source {source!r}; use 'bse' or 'nse'.",
            "_source": "none",
        }
    if src == "bse":
        rows = _indian_filings.list_bse_announcements(
            ticker,
            start_date=start_date or None,
            end_date=end_date or None,
            limit=limit,
            categories=categories or None,
        )
    else:
        rows = _indian_filings.list_nse_announcements(
            ticker, limit=limit, categories=categories or None
        )
    if not rows:
        scrip = _indian_filings.ticker_to_bse_scrip(ticker)
        return {
            "ticker": ticker.upper(),
            "scrip_cd": scrip,
            "filings": [],
            "error": (
                "Ticker not on BSE/NSE (try get_sec_filings instead)."
                if scrip is None
                else f"No filings matched the filter on {src.upper()}."
            ),
            "_source": src,
        }
    return {
        "ticker": ticker.upper(),
        "scrip_cd": rows[0].get("scrip_cd"),
        "filings": rows,
        "_source": src,
        "hint": (
            "Pass a filing's pdf_url to fetch_indian_document to get the text. "
            "Transcripts (~30-50 pages) are the richest claim-tracking source."
        ),
    }


@mcp.tool
def fetch_indian_document(
    url: str,
    max_chars: int = 40000,
    offset: int = 0,
) -> Dict[str, Any]:
    """Download a BSE / IR-page PDF and return cleaned plain text.

    Works for:

    - BSE ``AttachHis`` URLs from :func:`get_indian_filings`
    - Annual Report PDFs from :func:`get_screener_snapshot`
    - Concall transcript PDFs from :func:`get_indian_concall_urls`
    - Any other investor-relations-page PDF URL

    Returns ``{url, total_chars, offset, chars_returned, text,
    end_of_document, pages, extractor}``. ``text`` is the plain-text
    slice starting at ``offset`` (so the agent can page through a
    300-page Annual Report without re-fetching it - the full extraction
    is cached on the first call).

    Uses ``pdfplumber`` (layout-aware, better on financial tables) with
    a ``pypdf`` fallback if the primary extractor fails on a tricky PDF.
    """
    result = _indian_filings.fetch_pdf_text(url, max_chars=max_chars, offset=offset)
    if result is None:
        return {
            "url": url,
            "text": "",
            "_source": "pdf",
            "error": "PDF fetch or extraction failed (network error, bad URL, or encrypted PDF).",
        }
    return result


@mcp.tool
def get_screener_snapshot(ticker: str) -> Dict[str, Any]:
    """Return a one-shot Screener.in snapshot for an Indian ticker.

    Bundle of everything Screener.in shows on the top of a company
    page, plus the URLs of every Annual Report and concall transcript
    the page links to. Letting the agent skip the individual BSE /
    IR-page scraping when all it needs is the document list.

    Returns::

        {
          "ticker": "TCS",
          "name": "Tata Consultancy Services Ltd",
          "bse_code": 532540,
          "nse_symbol": "TCS",
          "ratios": {"Market Cap": "...", "Stock P/E": "17.6", ...},
          "pros": [...],          # machine-generated bull points
          "cons": [...],          # machine-generated bear points
          "annual_report_urls": [...],
          "concall_urls": [...],
          "screener_url": "https://www.screener.in/company/TCS/consolidated/",
        }

    Ratios is the same block the Screener page highlights at the top:
    Market Cap, Current Price, Stock P/E, Book Value, Dividend Yield,
    ROCE, ROE, High / Low. Values retain the original Indian number
    formatting (commas, crores).
    """
    snap = _indian_filings.scrape_screener_company(ticker)
    if snap is None:
        return {
            "ticker": ticker.upper(),
            "error": "Screener.in scrape failed (company not found or site change).",
            "_source": "screener",
        }
    return {**snap, "_source": "screener"}


@mcp.tool
def get_indian_concall_urls(ticker: str, limit: int = 8) -> Dict[str, Any]:
    """Return the most recent earnings-call transcript URLs for ``ticker``.

    Fast shortcut that combines two sources:

    1. BSE announcements filtered to ``subject LIKE '%Transcript%'``.
    2. Screener.in's concall URL list (IR-page-hosted transcripts).

    Deduplicated. Prefer the BSE versions when both sources return a
    URL for the same quarter - BSE PDFs are regulatory-grade and won't
    404 later.
    """
    out_urls: List[str] = []
    out_meta: List[Dict[str, Any]] = []

    # 1) BSE announcements (most recent first)
    bse = _indian_filings.list_bse_announcements(
        ticker, limit=30, categories=["transcript"]
    )
    for row in bse:
        url = row.get("pdf_url")
        if url and url not in out_urls:
            out_urls.append(url)
            out_meta.append({
                "url": url,
                "subject": row.get("subject"),
                "date": (row.get("date") or "")[:10],
                "source": "bse",
            })

    # 2) Screener.in (company IR page hosted PDFs; older vintages too)
    snap = _indian_filings.scrape_screener_company(ticker)
    if snap:
        for url in snap.get("concall_urls") or []:
            if url not in out_urls:
                out_urls.append(url)
                out_meta.append({
                    "url": url,
                    "subject": "Concall transcript (Screener)",
                    "date": "",
                    "source": "screener",
                })

    return {
        "ticker": ticker.upper(),
        "count": len(out_meta),
        "transcripts": out_meta[:limit],
        "_source": "bse+screener",
    }


@mcp.tool
def get_indian_annual_reports(ticker: str, limit: int = 6) -> Dict[str, Any]:
    """Return Annual Report PDF URLs for ``ticker`` (newest first).

    Sourced from Screener.in's company page (which aggregates both
    company-hosted and BSE-hosted PDFs). Use
    :func:`fetch_indian_document` to download + parse any of them.

    Annual Reports are the primary source of long-form management
    commentary + CEO letter + MD&A + BRSR (ESG commitments) for
    Indian issuers. Typical size: 200-400 pages.
    """
    snap = _indian_filings.scrape_screener_company(ticker)
    if snap is None:
        return {
            "ticker": ticker.upper(),
            "annual_reports": [],
            "error": "Screener.in scrape failed.",
            "_source": "screener",
        }
    urls = snap.get("annual_report_urls") or []
    return {
        "ticker": ticker.upper(),
        "count": len(urls),
        "annual_reports": urls[:limit],
        "_source": "screener",
        "hint": (
            "For claim tracking, the MD&A + Directors' Report sections "
            "(usually pages 15-80 of the PDF) are the richest source. "
            "Use fetch_indian_document with offset to page through the PDF."
        ),
    }


if __name__ == "__main__":  # pragma: no cover - entrypoint
    mcp.run()
