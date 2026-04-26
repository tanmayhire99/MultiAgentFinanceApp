"""Stock Research flow - focused deep dive on 1-N specific tickers.

Triggered when the router classifies a query as ``stock_research`` -
i.e. the user asks about specific stock(s) rather than their own
portfolio. Typical queries:

    "Research Western Digital"
    "Tell me about NVDA"
    "Should I buy AAPL"
    "Analyse Tata Consultancy"

Sequence (per ticker):

1. Probe which market the ticker belongs to (fixture look-up ->
   live US yfinance -> live Indian yfinance).
2. **Stock Agent** handoffs: ``get_quote`` + ``get_fundamentals``.
3. **Research Agent** handoffs: ``get_company_brief`` + ``search_news``.
4. Render: company overview + live fundamentals table + clickable
   catalysts + one-paragraph analyst synthesis (single LLM call, no
   panel debate unless ``decision.want_panel=True``).
5. If ``want_panel=True``: piggy-back on the investor panel by
   building a **synthetic** :class:`PortfolioContext` where the single
   researched ticker is the only "holding". This lets Buffett / Wood /
   Graham debate the stock without any shared-portfolio assumptions.

If the router extracted more than one ticker, we run sequentially (not
parallel) so the console/SSE output stays readable. Cap at 3 tickers per
request to keep latency reasonable.

MCP tool registration + the final disclaimer are handled by
:mod:`src.core.dispatcher`, so this flow only emits content events.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.personas.base import build_chat_model
from src.agents.personas.moderator import moderator_open_stream
from src.core.resilient_stream import stream_llm_resilient
from src.core.panel import (
    PanelEvent,
    PortfolioContext,
    _call_tool,
)
from src.core.router import RouteDecision


log = logging.getLogger("finai.flows.stock")


MAX_TICKERS_PER_REQUEST = 3


_ANALYST_SYSTEM = (
    "You are a careful, neutral equity research analyst. You have just "
    "been handed live fundamentals, a company overview, and recent news "
    "for ONE specific stock. Write a short structured analyst note in "
    "this exact format:\n\n"
    "**Investment thesis** — 2-3 sentences on what the company does and "
    "why the numbers matter.\n"
    "**What's working** — 2-3 bullets, each citing a specific metric or "
    "news item.\n"
    "**What's not working / risks** — 2-3 bullets, each grounded in the "
    "data or news.\n"
    "**What to watch** — 2-3 bullets of forward-looking indicators.\n\n"
    "Rules:\n"
    "- Aim for 220-320 words total.\n"
    "- Cite specific numbers and/or news headlines; don't generalise.\n"
    "- Neutral tone; no buy / sell calls; no price targets.\n"
    "- If a metric is missing (None or empty), acknowledge it rather "
    "than inventing a value.\n"
    "- Do NOT end with a disclaimer; one is appended separately."
)


# ---------------------------------------------------------------------------
# Market detection
# ---------------------------------------------------------------------------
def _probe_us_fixture(ticker: str) -> bool:
    """Check if the ticker is in the curated US fixture (cheap, no network)."""
    try:
        from src.agents.workers._fixtures import load_fixture, lookup

        _fx = load_fixture("us_stocks")
        try:
            lookup(_fx, ticker)
            return True
        except KeyError:
            return False
    except Exception:  # pragma: no cover - defensive
        return False


def _probe_indian_fixture(ticker: str) -> bool:
    try:
        from src.agents.workers._fixtures import load_fixture, lookup

        _fx = load_fixture("indian_stocks")
        try:
            lookup(_fx, ticker)
            return True
        except KeyError:
            return False
    except Exception:
        return False


async def _resolve_market(
    ticker: str,
) -> Tuple[str, Dict[str, Any]]:
    """Decide which Stock Agent owns ``ticker`` and fetch the quote.

    Returns ``(market, quote_resp)`` where ``market`` is either
    ``"us"`` or ``"in"``. Tries, in order:

    1. US fixture (fast local lookup)
    2. Indian fixture (fast local lookup)
    3. Live US yfinance
    4. Live Indian yfinance

    The first non-empty response wins. Raises ``RuntimeError`` only when
    every probe fails (i.e. ticker isn't listed anywhere we can reach).
    """
    plain = ticker.strip().upper().replace(".NS", "").replace(".BO", "")

    # Fixtures first - no subprocess call overhead.
    if _probe_us_fixture(plain):
        resp = await _call_tool("us_stock__get_quote", {"ticker": plain})
        if isinstance(resp, dict) and not resp.get("error"):
            return "us", resp
    if _probe_indian_fixture(plain):
        resp = await _call_tool("indian_stock__get_quote", {"ticker": plain})
        if isinstance(resp, dict) and not resp.get("error"):
            return "in", resp

    # Fall back to live probes.
    us_resp: Any = None
    try:
        us_resp = await _call_tool("us_stock__get_quote", {"ticker": plain})
    except Exception as e:  # pragma: no cover - network
        log.warning("US stock probe failed for %s: %s", plain, e)
    if (
        isinstance(us_resp, dict)
        and not us_resp.get("error")
        and us_resp.get("_source")
        and us_resp.get("_source") != "none"
    ):
        return "us", us_resp

    in_resp: Any = None
    try:
        in_resp = await _call_tool("indian_stock__get_quote", {"ticker": plain})
    except Exception as e:  # pragma: no cover - network
        log.warning("Indian stock probe failed for %s: %s", plain, e)
    if (
        isinstance(in_resp, dict)
        and not in_resp.get("error")
        and in_resp.get("_source")
        and in_resp.get("_source") != "none"
    ):
        return "in", in_resp

    raise RuntimeError(
        f"Could not find data for '{ticker}' in the US or Indian stock agents "
        f"(fixture + yfinance both returned empty)."
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _v(d: Dict[str, Any], k: str, suffix: str = "") -> str:
    v = d.get(k)
    return f"{v}{suffix}" if v not in (None, "") else "—"


def _fundamentals_table(fund: Dict[str, Any], market: str) -> str:
    # All currency-denominated fields are now USD. For Indian stocks the
    # Stock Agent has already applied FX conversion upstream.
    price = fund.get("price")
    if isinstance(price, (int, float)):
        price_str = f"${price:,.2f}"
    else:
        price_str = "—"
    native = fund.get("native_currency")
    market_label = "Indian listing (native INR, shown in USD)" if native == "INR" else "US listing"
    rows = [
        "| Metric | Value |",
        "|---|---|",
        f"| Market | {market_label} |",
        f"| Price | {price_str} |",
        f"| 52-week range | {_v(fund, 'fifty_two_week_low', '')} – {_v(fund, 'fifty_two_week_high', '')} |",
        f"| Market cap (USD bn) | {_v(fund, 'market_cap_bn', '')} |",
        f"| P/E (TTM) | {_v(fund, 'pe_ttm')} |",
        f"| Forward P/E | {_v(fund, 'forward_pe')} |",
        f"| Price / Book | {_v(fund, 'price_to_book')} |",
        f"| ROE | {_v(fund, 'roe_pct', '%')} |",
        f"| Operating margin | {_v(fund, 'operating_margin_pct', '%')} |",
        f"| Net margin | {_v(fund, 'net_margin_pct', '%')} |",
        f"| Debt / Equity | {_v(fund, 'debt_to_equity')} |",
        f"| Dividend yield | {_v(fund, 'dividend_yield_pct', '%')} |",
    ]
    return "\n".join(rows)


def _render_catalysts(news_items: List[Dict[str, Any]]) -> str:
    if not news_items:
        return "_No recent news found by the Research Agent._\n"
    sentiment_icon = {"positive": "🟢", "negative": "🔴", "mixed": "🟡", "neutral": "⚪"}
    lines: List[str] = []
    for item in news_items:
        icon = sentiment_icon.get((item.get("sentiment") or "").lower(), "•")
        headline = (item.get("headline") or "").strip() or "(no headline)"
        url = (item.get("url") or "").strip()
        date = item.get("date") or ""
        if isinstance(date, str) and len(date) >= 10:
            date = date[:10]
        title_md = f"[{headline}]({url})" if url else headline
        suffix = f" _({date})_" if date else ""
        lines.append(f"- {icon} {title_md}{suffix}")
    return "\n".join(lines) + "\n"


def _format_analyst_input(
    ticker: str,
    market: str,
    fund: Dict[str, Any],
    brief: Dict[str, Any],
    news_items: List[Dict[str, Any]],
) -> str:
    """Compose the analyst's user-message input from the gathered tool data."""
    lines: List[str] = [f"Ticker: {ticker}   |   Market: {market.upper()}"]
    name = fund.get("name") or brief.get("ticker") or ticker
    lines.append(f"Company: {name}")
    if brief.get("summary"):
        lines.append("\nCompany overview (research agent):")
        lines.append(brief["summary"])
    lines.append("\nLive fundamentals (stock agent):")
    for k in (
        "price",
        "pe_ttm",
        "forward_pe",
        "price_to_book",
        "roe_pct",
        "operating_margin_pct",
        "net_margin_pct",
        "debt_to_equity",
        "dividend_yield_pct",
        "market_cap_bn",
    ):
        if fund.get(k) not in (None, ""):
            lines.append(f"- {k}: {fund[k]}")
    if news_items:
        lines.append("\nRecent headlines:")
        for i, n in enumerate(news_items[:5], start=1):
            headline = n.get("headline", "")
            date = n.get("date", "")
            sentiment = n.get("sentiment", "")
            parts = [f"{i}. {headline}"]
            if date:
                parts.append(f"({str(date)[:10]})")
            if sentiment:
                parts.append(f"[{sentiment}]")
            lines.append(" ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic portfolio context (used only when want_panel=True)
# ---------------------------------------------------------------------------
def _synthetic_portfolio_ctx(
    tickers_data: List[Dict[str, Any]],
) -> PortfolioContext:
    """Build a minimal PortfolioContext around the researched tickers.

    Each researched ticker becomes a weight-1/N "holding" so the panel
    personas have a structured context to reason over (same format they
    already know how to consume). This avoids having to fork the panel
    for stock-only runs.
    """
    n = max(1, len(tickers_data))
    weight = round(1.0 / n, 4)
    holdings: List[Dict[str, Any]] = []
    snapshot: Dict[str, Dict[str, Any]] = {}
    catalysts: Dict[str, List[Dict[str, Any]]] = {}
    for row in tickers_data:
        ticker = row["ticker"]
        fund = row.get("fund") or {}
        holdings.append(
            {
                "ticker": ticker,
                "name": fund.get("name") or ticker,
                "sector": fund.get("sector") or "Unknown",
                "country": (row.get("market") or "").upper(),
                "weight": weight,
                "current_value_usd": 0,
            }
        )
        snapshot[ticker] = fund
        catalysts[ticker] = row.get("news") or []

    ctx = PortfolioContext(
        user_id="stock_research",
        summary={
            "portfolio_name": "Stock Research",
            "total_value_usd": 0,
            "absolute_gain_pct": 0,
            "holding_count": len(holdings),
            "geographic_split": {},
        },
        allocation={"grouped_sectors_pct": {}},
        risks={"risks": []},
        score={"score": "—", "band": "—"},
        holdings=holdings,
        market_snapshot=snapshot,
        catalysts=catalysts,
    )
    return ctx


# ---------------------------------------------------------------------------
# Single-ticker research pipeline
# ---------------------------------------------------------------------------
async def _research_single_ticker(
    ticker: str,
    *,
    user_id: str = "demo",
    query: str = "",
) -> AsyncIterator[PanelEvent]:
    """Run the focused research pipeline for one ticker.

    Yield shape (Fix 3 — chat-progress version):

    * ``{"type": "_status", "text": "<short status line>"}`` — chat
      status events the caller emits to LibreChat's chat pane BEFORE
      it opens the artifact. These show real-time progress so the
      user isn't staring at a frozen chat for 60-180s.
    * ``{"type": "text", ...}`` — rendered report sections that the
      caller forwards into the open artifact's body. The first such
      event is the cue for the caller to call ``open_artifact()``.
    * ``{"type": "_ticker_ready", "data": {...}}`` — last event,
      signals the ticker pipeline is done and hands off the gathered
      data so the caller can build a synthetic panel context.

    This split lets the caller (``run``) emit several chat-status lines
    during the fast data-gather phase, then open the artifact and
    let the slower analyst-synthesis LLM call stream inside it.

    ``user_id`` / ``query`` carry through only so the analyst-synthesis
    LLM call at the end can be cached / replayed on NIM blinks (see
    :mod:`src.core.resilient_stream`).
    """
    ticker = ticker.upper().strip()

    # 1) Market resolution
    yield {"type": "_status", "text": f"Resolving market location for {ticker}..."}
    try:
        market, quote = await _resolve_market(ticker)
    except Exception as e:
        yield {
            "type": "error",
            "text": f"{ticker}: could not resolve market — {e}",
        }
        yield {"type": "_ticker_ready", "data": None}
        return

    agent_ns = "us_stock" if market == "us" else "indian_stock"
    market_pretty = "US exchanges" if market == "us" else "Indian exchanges (NSE/BSE)"
    yield {"type": "_status", "text": f"✓ {ticker} listed on {market_pretty}"}

    # 2) Fundamentals
    yield {"type": "_status", "text": f"Pulling live fundamentals for {ticker}..."}
    try:
        fund = await _call_tool(f"{agent_ns}__get_fundamentals", {"ticker": ticker})
    except Exception as e:
        log.exception("Fundamentals failed for %s", ticker)
        yield {"type": "error", "text": f"Fundamentals for {ticker} failed: {e}"}
        fund = {}
    if not isinstance(fund, dict):
        fund = {}
    if fund:
        price = fund.get("price")
        pe = fund.get("pe_ttm") or fund.get("forward_pe")
        bits = []
        if price is not None:
            bits.append(f"${price}")
        if pe is not None:
            bits.append(f"P/E {pe}")
        snapshot = ", ".join(bits) if bits else "data captured"
        yield {"type": "_status", "text": f"✓ {ticker}: {snapshot}"}

    # 3) Company brief
    yield {"type": "_status", "text": f"Fetching company brief for {ticker}..."}
    try:
        brief_resp = await _call_tool("research__get_company_brief", {"ticker": ticker})
    except Exception as e:
        log.exception("Company brief failed for %s", ticker)
        brief_resp = {"summary": "", "sources": []}
    if not isinstance(brief_resp, dict):
        brief_resp = {"summary": "", "sources": []}
    sources = brief_resp.get("sources") or []
    yield {"type": "_status", "text": f"✓ Got company brief ({len(sources)} sources)"}

    # 4) Recent news
    yield {"type": "_status", "text": f"Searching for recent {ticker} catalysts..."}
    try:
        news_resp = await _call_tool(
            "research__search_news", {"ticker": ticker, "max_items": 5}
        )
    except Exception as e:
        log.exception("News failed for %s", ticker)
        news_resp = {}
    if not isinstance(news_resp, dict):
        news_resp = {}
    news_items = news_resp.get("news") or []
    yield {"type": "_status", "text": f"✓ Found {len(news_items)} recent catalysts"}

    # 5) Render the structured report (this is the artifact-body content)
    company_name = fund.get("name") or brief_resp.get("ticker") or ticker
    sections: List[str] = [f"\n## {ticker} — {company_name}\n"]

    # Company overview
    summary = (brief_resp.get("summary") or "").strip()
    if summary:
        sources_md = ""
        if sources:
            top_sources = [
                f"[{(s.get('title') or 'source').strip()[:60]}]({s.get('url')})"
                for s in sources[:3]
                if s.get("url")
            ]
            if top_sources:
                sources_md = f"\n_Sources: {', '.join(top_sources)}_"
        sections.append(f"\n#### Company Overview\n\n{summary}{sources_md}\n")
    else:
        sections.append(
            "\n#### Company Overview\n\n_No company brief available._\n"
        )

    # Fundamentals
    if fund:
        sections.append(
            f"\n#### Live Fundamentals (source: `{fund.get('_source', 'n/a')}`)\n\n"
            + _fundamentals_table(fund, market)
            + "\n"
        )
    else:
        sections.append("\n#### Live Fundamentals\n\n_No data._\n")

    # Catalysts
    sections.append(
        "\n#### Recent Catalysts\n\n" + _render_catalysts(news_items) + "\n"
    )

    yield {
        "type": "text",
        "text": "".join(sections),
        "persona": "orchestrator",
    }

    # 6) Analyst synthesis (single LLM call, no panel). One last
    # chat-status line right before the artifact starts streaming —
    # so the user sees "synthesis happening now, watch the side pane".
    # The header itself is emitted as plain artifact-body text (not a
    # "header" event) so it lands cleanly inside the side-pane markdown.
    yield {
        "type": "_status",
        "text": f"Generating analyst synthesis for {ticker} (streams to artifact)...",
    }
    yield {
        "type": "text",
        "text": "\n### Analyst Synthesis\n\n",
        "persona": "orchestrator",
    }
    analyst_input = _format_analyst_input(ticker, market, fund, brief_resp, news_items)
    analyst_messages = [
        SystemMessage(content=_ANALYST_SYSTEM),
        HumanMessage(
            content=(
                f"Research target: {ticker}\n\n{analyst_input}\n\n"
                "Write the analyst note now, following the system prompt structure."
            )
        ),
    ]

    async def _analyst_stream() -> AsyncIterator[str]:
        llm = build_chat_model(temperature=0.2, max_tokens=700, streaming=True)
        async for chunk in llm.astream(analyst_messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text

    # Cache key includes the ticker so same-ticker re-asks reuse the
    # cached brief when NIM blinks. ``query`` may be empty for very
    # old callers; that's fine — the cache key falls back to just the
    # ticker-scoped agent string.
    cache_query = query or f"research {ticker}"
    async for chunk in stream_llm_resilient(
        stream_factory=_analyst_stream,
        user_id=user_id,
        query=cache_query,
        flow_name="stock_research",
        cache_agent=f"analyst_synthesis:{ticker}",
        cache_agent_title=f"Analyst Synthesis ({ticker})",
        retries=1,
        error_label=f"analyst synthesis for {ticker}",
    ):
        yield {"type": "text", "text": chunk, "persona": "moderator"}

    yield {"type": "text", "text": "\n\n", "persona": "moderator"}

    # Hand off the gathered data to the caller (for synthetic panel ctx)
    yield {
        "type": "_ticker_ready",
        "data": {
            "ticker": ticker,
            "market": market,
            "quote": quote,
            "fund": fund,
            "brief": brief_resp,
            "news": news_items,
        },
    }


# ---------------------------------------------------------------------------
# Flow entry point
# ---------------------------------------------------------------------------
def _build_research_chat_summary(
    gathered: List[Dict[str, Any]],
    *,
    panel_summary: Optional[str] = None,
    wants_artifact: bool = False,
) -> str:
    """Compose the brief chat line shown at the very end of the flow.

    Two phrasings depending on which mode the flow ran in:

    * **artifact mode** — the structured report lives in the side pane,
      so we surface the headline numbers as a quick recap and point the
      user toward the artifact (``Full report in the artifact pane →``).
    * **inline mode** — everything was already streamed inline above
      so a long "Full report in the artifact pane" line would be
      misleading. Instead we emit a minimal "see above" closer
      (just the panel stance summary if any).
    """
    if not gathered:
        if wants_artifact:
            return "Stock research complete. Full report in the artifact pane →\n"
        return "_(No data captured for the requested ticker.)_\n"

    bullets: List[str] = []
    for entry in gathered:
        ticker = entry.get("ticker") or "?"
        fund = entry.get("fund") or {}
        price = fund.get("price")
        pe = fund.get("pe_ttm") or fund.get("forward_pe")
        roe = fund.get("roe_pct")
        bits: List[str] = []
        if price is not None:
            bits.append(f"${price}")
        if pe is not None:
            bits.append(f"{pe:.1f}× P/E" if isinstance(pe, (int, float)) else f"{pe} P/E")
        if roe is not None:
            try:
                bits.append(f"{float(roe):.1f}% ROE")
            except (TypeError, ValueError):
                pass
        line = f"**{ticker}** — {', '.join(bits) if bits else 'data captured'}"
        bullets.append(line)

    if wants_artifact:
        # Side-pane mode: recap headline numbers + point at the pane
        head = " · ".join(bullets) if len(bullets) <= 2 else "\n- " + "\n- ".join(bullets)
        pieces = [head, "Full report in the artifact pane →"]
        if panel_summary:
            pieces.insert(1, panel_summary)
        return "\n\n".join(pieces) + "\n"

    # Inline mode: the user has already seen the full streamed report
    # above this line. Don't repeat the headline numbers (they're
    # right there in the table they just scrolled past). Just append
    # the panel stance summary when present.
    if panel_summary:
        return f"\n\n{panel_summary}\n"
    return ""


async def run(
    query: str,
    decision: Optional[RouteDecision] = None,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Run the Stock Research flow.

    Fix 3 layout: a brief italic chat status line, then a markdown
    artifact in LibreChat's side pane containing the full structured
    report (and panel debate if requested), then a one-line chat
    summary with the headline numbers. No banner headers, no
    "For each ticker we will: 1) Probe..." preamble, no per-tool
    "🔗 Orchestrator → US Stock Agent · ..." narration.

    The decision's ``tickers`` list drives the work. If it's empty, we
    ask the user to be more specific instead of running an expensive
    full panel against the wrong thing.
    """
    from src.core.artifacts import (
        artifact_body,
        chat_text,
        close_artifact,
        open_artifact,
        safe_id,
        status,
    )

    decision = decision or {}
    tickers = decision.get("tickers") or []
    tickers = [t for t in tickers if t][:MAX_TICKERS_PER_REQUEST]

    # Empty-ticker fallback stays inline — no point opening an artifact
    # for a "tell me which ticker" prompt.
    if not tickers:
        yield chat_text(
            "I couldn't identify a specific ticker or company in that "
            "query. Try `Research NVDA` / `Tell me about Tesla` / "
            "`Analyse TCS`.\n"
        )
        return

    label = ", ".join(tickers)
    want_panel = bool(decision.get("want_panel"))
    # ``wants_artifact`` is set by the dispatcher when the user
    # prefixes the query with ``/artifact`` / ``/report`` or includes
    # phrases like "generate report" / "as artifact". When False
    # (the default), everything streams inline in the chat (Claude-
    # style); when True, the heavy structured content is wrapped in a
    # ``:::artifact{}:::`` block for LibreChat's side pane.
    wants_artifact = bool(decision.get("wants_artifact"))

    # 1) Inline status — kept short. The fetch loop below will emit
    #    several more status updates as data comes in.
    if want_panel:
        yield status(
            f"Researching {label} for full investor panel "
            f"(Buffett / Wood / Graham, ~60-180s)"
        )
    else:
        yield status(f"Researching {label}")

    # 2) Iterate each ticker. _research_single_ticker yields:
    #      "_status"        -> chat status line
    #      "_ticker_ready"  -> ticker data handoff for panel context
    #      everything else  -> rendered report content
    #
    # In artifact mode we open the artifact LAZILY on the first content
    # event so all data-gather "_status" lines land in the chat first,
    # then the rendering / streaming phase fills the side pane.
    #
    # In inline mode (the default) we forward content events directly
    # to the chat - no artifact wrapper, no side pane.
    artifact_title = (
        f"Stock Research + Panel: {label}" if want_panel
        else f"Stock Research: {label}"
    )
    artifact_id = safe_id(f"stock-research-{label}")

    artifact_open = False

    def _ensure_artifact_open() -> List[PanelEvent]:
        """Open the artifact on demand (artifact mode only)."""
        nonlocal artifact_open
        if artifact_open or not wants_artifact:
            return []
        artifact_open = True
        return [
            open_artifact(identifier=artifact_id, title=artifact_title),
            artifact_body(f"# {artifact_title}\n\n"),
        ]

    gathered: List[Dict[str, Any]] = []
    for ticker in tickers:
        async for ev in _research_single_ticker(
            ticker, user_id=user_id, query=query
        ):
            etype = ev.get("type")
            if etype == "_status":
                # Chat status — emit OUTSIDE any open artifact. In inline
                # mode this is just chat. In artifact mode we still emit
                # it to chat ONLY if the artifact hasn't opened yet
                # (otherwise it'd land inside the artifact body).
                if not artifact_open:
                    yield status(ev.get("text", ""))
                continue
            if etype == "_ticker_ready":
                if ev.get("data"):
                    gathered.append(ev["data"])  # type: ignore[index]
                continue
            # Any other event is rendered content.
            #   - Artifact mode: lazy-open on first content, then forward
            #   - Inline mode: forward directly to chat (no wrapper)
            for opener in _ensure_artifact_open():
                yield opener
            yield ev

    # 3) Optional panel debate — same routing rules as above.
    panel_summary: Optional[str] = None
    if want_panel and gathered:
        async for ev in _run_panel_on_tickers(query, gathered, user_id=user_id):
            etype = ev.get("type")
            if etype == "_status":
                if not artifact_open:
                    yield status(ev.get("text", ""))
                continue
            if etype == "_panel_summary":
                panel_summary = ev.get("text")  # type: ignore[assignment]
                continue
            for opener in _ensure_artifact_open():
                yield opener
            yield ev

    # 4) Close the artifact (artifact mode only) and emit the chat
    #    summary. In inline mode we just emit the summary; the user
    #    has already seen the full streamed content above.
    if artifact_open:
        yield close_artifact()
    yield chat_text(
        _build_research_chat_summary(
            gathered,
            panel_summary=panel_summary,
            wants_artifact=wants_artifact,
        )
    )


# ---------------------------------------------------------------------------
# Optional panel over the researched tickers
# ---------------------------------------------------------------------------
_STANCE_GLYPHS: Dict[str, str] = {
    "bullish": "🟢",
    "neutral": "⚪",
    "cautious": "🟡",
    "bearish": "🔴",
}


def _build_panel_chat_summary(scratchpad: "PanelScratchpad") -> str:  # type: ignore[name-defined]
    """One-line stance summary for the chat (full debate is in the artifact).

    Reads the FINAL-round entry for each persona and renders a
    compact glyph + stance line, e.g.::

        🟡 Buffett cautious · 🟡 Wood cautious · 🔴 Graham bearish — converged Round 2

    The :class:`PanelScratchpad` stores its data as a flat list of
    ``entries`` (each with ``round``, ``persona``, ``stance`` etc.), so
    we group by persona, pick the highest-round entry per persona, and
    sort them in canonical persona order via ``PERSONA_ORDER``.

    Defensive: if the scratchpad shape isn't what we expect, returns
    "" and the caller falls back to a generic "Full report in the
    artifact pane" line.
    """
    from src.core.debate import PERSONA_ORDER

    try:
        entries = list(getattr(scratchpad, "entries", []) or [])
        if not entries:
            return ""
        max_round = max(e.round for e in entries)

        # Final entry per persona name (latest round they participated in)
        latest_by_persona: Dict[str, Any] = {}
        for entry in entries:
            current = latest_by_persona.get(entry.persona)
            if current is None or entry.round > current.round:
                latest_by_persona[entry.persona] = entry
    except Exception:
        return ""

    parts: List[str] = []
    for persona in PERSONA_ORDER:
        entry = latest_by_persona.get(persona.name)
        if entry is None:
            continue
        # Use the canonical persona name (e.g. "Buffett") rather than
        # the full title with parens (e.g. "Warren Buffett (Value)").
        short_name = persona.title.split("(")[0].strip().split()[-1]
        stance = (entry.stance or "neutral").lower()
        glyph = _STANCE_GLYPHS.get(stance, "⚪")
        parts.append(f"{glyph} {short_name} {stance}")

    if not parts:
        return ""
    return " · ".join(parts) + f" — converged Round {max_round}"


async def _run_panel_on_tickers(
    query: str,
    gathered: List[Dict[str, Any]],
    *,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Run the multi-round sequential debate over the researched ticker(s).

    Builds a synthetic :class:`PortfolioContext` from the gathered
    fundamentals/news so the personas' existing prompts + tools apply
    without code changes, then hands off to the shared
    :func:`src.core.debate.run_debate_loop` used by portfolio_analysis.

    Fix 3: this function streams its content into the open artifact
    (the caller has already opened it). It still uses markdown
    headers like ``## Investor Panel Debate`` and ``### Moderator —
    Opening`` because those render as section headers within the
    artifact pane, not as a banner in chat.

    Yields one bookkeeping event ``{"type": "_panel_summary", "text": "..."}``
    just before returning so the caller can include the stance one-liner
    in the inline chat summary.
    """
    from src.core.debate import PanelScratchpad, run_debate_loop
    from src.core.flows.portfolio_analysis import (
        _DEBATE_SYNTH_SYSTEM,
        _format_scratchpad_for_moderator,
    )

    ctx = _synthetic_portfolio_ctx(gathered)

    yield {
        "type": "text",
        "text": "\n## Investor Panel Debate\n\n### Moderator — Opening\n\n",
        "persona": "orchestrator",
    }
    mod_ctx_block = ctx.moderator_context_block()

    async def _moderator_open_stream() -> AsyncIterator[str]:
        async for chunk in moderator_open_stream(query, mod_ctx_block):
            yield chunk

    async for chunk in stream_llm_resilient(
        stream_factory=_moderator_open_stream,
        user_id=user_id,
        query=query,
        flow_name="stock_panel",
        cache_agent="moderator_opening",
        cache_agent_title="Moderator Opening",
        retries=1,
        error_label="moderator opening",
    ):
        yield {"type": "text", "text": chunk, "persona": "moderator"}
    yield {"type": "text", "text": "\n\n---\n\n", "persona": "moderator"}

    # Sequential multi-round debate on the synthetic portfolio context
    scratchpad: Optional[PanelScratchpad] = None
    async for ev in run_debate_loop(
        query,
        portfolio_ctx=ctx,
        user_id=user_id,
        flow_name="stock_panel",
    ):
        if ev.get("type") == "_debate_done":
            scratchpad = ev.get("scratchpad")  # type: ignore[assignment]
            continue
        yield ev

    if scratchpad is None:
        yield {"type": "error", "text": "Debate loop finished without a scratchpad."}
        return

    # Closing brief grounded in the full transcript - rendered as a
    # markdown header inside the artifact (no banner-style emission).
    yield {
        "type": "text",
        "text": "\n### Moderator — Closing Brief\n\n",
        "persona": "orchestrator",
    }
    transcript = _format_scratchpad_for_moderator(scratchpad)
    # Feed the moderator the FULL synthetic-portfolio context (not just
    # the short moderator brief) so the closing references specific
    # metrics the personas debated.
    full_ctx = ctx.persona_context_block()
    synth_messages = [
        SystemMessage(content=_DEBATE_SYNTH_SYSTEM),
        HumanMessage(
            content=(
                f"Stock(s) under discussion: {mod_ctx_block}\n\n"
                f"Full per-stock context (identical to what the personas saw):\n\n"
                f"{full_ctx}\n\n"
                f"Full debate transcript:\n\n{transcript}\n\n"
                "Write the Closing Brief now, following the system prompt structure."
            )
        ),
    ]

    async def _synth_stream() -> AsyncIterator[str]:
        llm = build_chat_model(temperature=0.2, max_tokens=1000, streaming=True)
        async for chunk in llm.astream(synth_messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text

    async for chunk in stream_llm_resilient(
        stream_factory=_synth_stream,
        user_id=user_id,
        query=query,
        flow_name="stock_panel",
        cache_agent="moderator_synthesis",
        cache_agent_title="Moderator Closing Brief",
        retries=1,
        error_label="moderator synthesis",
    ):
        yield {"type": "text", "text": chunk, "persona": "moderator"}

    # Pass the stance one-liner up to the caller (run) so it can include
    # it in the inline chat summary. This is a bookkeeping event, not
    # rendered to the user directly.
    summary_text = _build_panel_chat_summary(scratchpad)
    if summary_text:
        yield {"type": "_panel_summary", "text": summary_text}
