"""FinAI Investor Panel supervisor with live orchestration trace.

Flow:

    orchestrator pre-panel  (fetch portfolio + deterministic metrics)
        ->  moderator_open          (streamed)
        ->  [buffett | wood | graham] run in parallel; events drained
            in display order so the transcript always reads
            Buffett -> Wood -> Graham
        ->  moderator_synthesise    (streamed)

The orchestrator phase is pure Python - it calls MCP tools directly so
the audience sees the supervisor reaching out to the Portfolio Agent
before the personas ever speak. Each persona then receives the same
portfolio context in its user message and can use its own MCP tools
(US / Indian stock data, research, etc.) for deeper stock-level analysis.

Event types
-----------

``header``           section title (e.g. ``### Warren Buffett (Value)\\n\\n``)
``text``             raw content delta to append to the chat
``tool_call``        an agent invoked another agent's tool (carries
                     ``persona``, ``persona_label``, ``tool``, ``args``)
``tool_result``      a short result summary for orchestrator calls only
``persona_verdict``  structured summary emitted after a persona finishes
``panel_done``       terminal marker (no content)
``error``            surfaced error with an explanatory message
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage

from src.personas.base import (
    PersonaDef,
    PersonaVerdict,
    build_persona_agent,
    parse_verdict,
)
from src.personas.buffett import BUFFETT
from src.personas.graham import GRAHAM
from src.personas.wood import WOOD
from src.config import mcp_servers


log = logging.getLogger("finai.panel")


# Display order is fixed: value -> growth -> defensive.
PERSONA_ORDER: List[PersonaDef] = [BUFFETT, WOOD, GRAHAM]


# LangGraph's ``create_react_agent`` defaults ``recursion_limit`` to 25,
# where each LLM turn AND each tool call counts as one step. Graham, the
# defensive investor, likes to check every holding in the portfolio -
# 10 holdings x (fundamentals + defensive_metrics + quote) = easily 20+
# tool calls, which blows past the default. We set a generous ceiling
# (500) so no realistic persona turn ever trips the limit, while still
# keeping a safety net against a pathological loop.
PERSONA_RECURSION_LIMIT = 500


class PanelEvent(TypedDict, total=False):
    """Single streaming event emitted during a panel run."""

    type: str
    text: str
    persona: str
    persona_label: str
    title: str
    tool: str
    args: Dict[str, Any]
    result_preview: str
    stance: str
    one_liner: str
    confidence: str
    tools_used: List[str]


# Sentinel used to terminate per-persona queues.
_PERSONA_DONE = object()


# ---------------------------------------------------------------------------
# Portfolio context (produced by the orchestrator phase, consumed by the
# moderator and each persona so their reasoning is grounded in real holdings)
# ---------------------------------------------------------------------------
@dataclass
class PortfolioContext:
    """Snapshot of the user's portfolio + deterministic analytics.

    Populated by the orchestrator phase and handed down to the moderator
    and each persona. Contains both the portfolio-level data (holdings,
    sector allocation, concentration flags, diversification score) and a
    per-holding **market snapshot** plus recent **catalysts** for the top
    holdings, so the personas can reason about specific numbers without
    re-fetching them.
    """

    user_id: str
    summary: Dict[str, Any] = field(default_factory=dict)
    allocation: Dict[str, Any] = field(default_factory=dict)
    risks: Dict[str, Any] = field(default_factory=dict)
    score: Dict[str, Any] = field(default_factory=dict)
    holdings: List[Dict[str, Any]] = field(default_factory=list)
    # Ticker -> live fundamentals dict (from stock agents, via orchestrator)
    market_snapshot: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Ticker -> list of recent catalyst dicts (from research agent)
    catalysts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Ticker -> list of moat-signal bullet strings (curated fixture content,
    # fetched by the orchestrator so personas don't each have to re-query).
    moat_signals: Dict[str, List[str]] = field(default_factory=dict)
    # Ticker -> growth_metrics dict (5y CAGR, R&D intensity, addressable
    # market, disruption score, narrative, 1y growth, beta).
    growth: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Ticker -> defensive_metrics dict (Graham number, margin-of-safety,
    # current ratio, interest coverage, book value, EPS, dividend yield).
    defensive: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def has_data(self) -> bool:
        return bool(self.holdings)

    def top_holdings_table_md(self, n: Optional[int] = None) -> str:
        """Markdown table of the top-N holdings by weight.

        Columns: index, ticker, name, weight%, buy price, current price,
        per-holding P&L %, current value. Pass ``n=None`` (or ``n`` >=
        number of holdings) to render every holding.
        """
        if not self.holdings:
            return ""
        sorted_h = sorted(self.holdings, key=lambda h: -h.get("weight", 0))
        if n is not None:
            sorted_h = sorted_h[:n]
        lines = [
            "| # | Ticker | Name | Weight | Buy ($) | Current ($) | P&L % | Value ($) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        def _money(v: Optional[float]) -> str:
            return f"${v:,.2f}" if v is not None else "—"

        def _pnl(pct: Optional[float]) -> str:
            if pct is None:
                return "—"
            icon = "🟢" if pct > 0 else ("🔴" if pct < 0 else "⚪")
            return f"{icon} {pct:+.1f}%"

        for i, h in enumerate(sorted_h, start=1):
            weight_pct = round(h.get("weight", 0) * 100, 1)
            buy = h.get("avg_cost_usd")
            cur = h.get("current_price_usd")
            gain_pct = h.get("absolute_gain_pct")
            val = h.get("current_value_usd", 0)
            lines.append(
                f"| {i} | **{h.get('ticker','?')}** | {h.get('name','?')} | "
                f"{weight_pct}% | {_money(buy)} | {_money(cur)} | "
                f"{_pnl(gain_pct)} | ${val:,.2f} |"
            )
        return "\n".join(lines)

    def sector_summary_md(self) -> str:
        """Markdown list of grouped sector weights."""
        buckets = self.allocation.get("grouped_sectors_pct", {}) or {}
        if not buckets:
            return ""
        lines = []
        for sector, pct in sorted(buckets.items(), key=lambda kv: -kv[1]):
            lines.append(f"- **{sector}**: {pct}%")
        return "\n".join(lines)

    def risks_summary_md(self) -> str:
        items = self.risks.get("risks", []) or []
        if not items:
            return "_No concentration flags at the default thresholds._"
        lines = []
        for r in items:
            icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(r.get("severity"), "•")
            lines.append(f"- {icon} **{r.get('type','?').replace('_',' ')}** — {r.get('detail','')}")
        return "\n".join(lines)

    def score_summary_md(self) -> str:
        if not self.score:
            return ""
        s = self.score
        return (
            f"**Diversification score: {s.get('score','?')} / 100** "
            f"({s.get('band','?')}) — HHI {s.get('herfindahl_index','?')}, "
            f"~{s.get('effective_holdings','?')} effective holdings"
        )

    def market_snapshot_md(self) -> str:
        """Markdown table of live fundamentals for the tickers we fetched.

        Price column is rendered in USD. For Indian listings the Stock
        Agent has already converted from INR to USD via
        :data:`src.mcp._live.USD_PER_INR`, so the values here
        compare apples-to-apples across US and Indian holdings.
        """
        if not self.market_snapshot:
            return ""
        rows = [
            "| Ticker | Price ($) | P/E (TTM) | ROE | Op. Margin | Debt/Eq | Source |",
            "|---|---|---|---|---|---|---|",
        ]

        def _v(d, k, suffix=""):
            v = d.get(k)
            return f"{v}{suffix}" if v not in (None, "") else "—"

        def _price_usd(d):
            v = d.get("price")
            if v in (None, ""):
                return "—"
            try:
                return f"${float(v):,.2f}"
            except (TypeError, ValueError):
                return str(v)

        def _source_label(src):
            if not src:
                return "—"
            if src.startswith("live:"):
                return "live"
            if src.startswith("fixture:"):
                return "fixture"
            return src

        for ticker, fund in self.market_snapshot.items():
            if not fund:
                continue
            rows.append(
                f"| **{ticker}** "
                f"| {_price_usd(fund)} "
                f"| {_v(fund, 'pe_ttm')} "
                f"| {_v(fund, 'roe_pct', '%')} "
                f"| {_v(fund, 'operating_margin_pct', '%')} "
                f"| {_v(fund, 'debt_to_equity')} "
                f"| {_source_label(fund.get('_source'))} |"
            )
        return "\n".join(rows) if len(rows) > 2 else ""

    def catalysts_md(self, items_per_ticker: int = 3) -> str:
        """Markdown list of the N most-recent catalysts for each ticker.

        Each bullet shows the sentiment icon, headline (linked to the
        source when the URL is known), publication date and - when the
        Research Agent included one - a one-line summary. Personas
        should be able to reason about *why* a ticker moved, not just
        that there was a headline, so we include up to
        ``items_per_ticker`` per ticker (default 3).
        """
        if not self.catalysts:
            return ""
        sentiment_icon = {"positive": "🟢", "negative": "🔴", "mixed": "🟡", "neutral": "⚪"}
        lines: List[str] = []
        for ticker, news in self.catalysts.items():
            if not news:
                continue
            lines.append(f"**{ticker}**:")
            for item in news[: max(1, int(items_per_ticker))]:
                icon = sentiment_icon.get((item.get("sentiment") or "").lower(), "•")
                headline = (item.get("headline") or "").strip()
                url = (item.get("url") or "").strip()
                date = item.get("date") or ""
                if isinstance(date, str) and len(date) >= 10:
                    date = date[:10]
                snippet = (item.get("snippet") or "").strip()
                title_md = f"[{headline}]({url})" if url else headline
                head_line = f"  - {icon} {title_md}"
                if date:
                    head_line += f"  _({date})_"
                lines.append(head_line)
                if snippet:
                    short = snippet if len(snippet) <= 220 else snippet[:220].rstrip() + "…"
                    lines.append(f"    _{short}_")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Qualitative enrichment rendering
    # ------------------------------------------------------------------
    def moat_signals_md(self) -> str:
        """Markdown - one block per ticker, 2-4 moat bullets each."""
        if not self.moat_signals:
            return ""
        lines: List[str] = []
        for ticker in self._holding_order():
            signals = self.moat_signals.get(ticker) or []
            if not signals:
                continue
            lines.append(f"**{ticker}**:")
            for s in signals:
                lines.append(f"  - {s}")
        return "\n".join(lines)

    def growth_snapshot_md(self) -> str:
        """Compact growth / innovation table: CAGRs, R&D, addressable market."""
        if not self.growth:
            return ""
        rows = [
            "| Ticker | 5y Rev CAGR | 5y EPS CAGR | R&D % | Disruption | Addressable market |",
            "|---|---|---|---|---|---|",
        ]

        def _pct(v) -> str:
            return f"{v}%" if v not in (None, "") else "—"

        def _num(v) -> str:
            return f"{v}" if v not in (None, "") else "—"

        have_narrative: List[str] = []
        for ticker in self._holding_order():
            g = self.growth.get(ticker) or {}
            if not g:
                continue
            rows.append(
                f"| **{ticker}** "
                f"| {_pct(g.get('revenue_cagr_5y_pct'))} "
                f"| {_pct(g.get('eps_cagr_5y_pct'))} "
                f"| {_pct(g.get('rd_intensity_pct'))} "
                f"| {_num(g.get('disruption_score'))}/5 "
                f"| {(g.get('addressable_market') or '—')[:60]} |"
            )
            narr = g.get("narrative") or ""
            if narr:
                have_narrative.append(f"- **{ticker}**: {narr}")
        out = ["\n".join(rows)]
        if have_narrative:
            out.append("")
            out.append("**Analyst narrative per holding:**")
            out.extend(have_narrative)
        return "\n".join(out)

    def defensive_snapshot_md(self) -> str:
        """Compact defensive / Graham-style table per holding."""
        if not self.defensive:
            return ""
        rows = [
            "| Ticker | Current Ratio | Graham # ($) | Margin of Safety | Int. Coverage | Div Yield |",
            "|---|---|---|---|---|---|",
        ]

        def _val(v) -> str:
            return f"{v}" if v not in (None, "") else "—"

        def _pct(v) -> str:
            return f"{v}%" if v not in (None, "") else "—"

        def _dollar(v) -> str:
            if v in (None, ""):
                return "—"
            try:
                return f"${float(v):,.2f}"
            except (TypeError, ValueError):
                return str(v)

        for ticker in self._holding_order():
            d = self.defensive.get(ticker) or {}
            if not d:
                continue
            rows.append(
                f"| **{ticker}** "
                f"| {_val(d.get('current_ratio'))} "
                f"| {_dollar(d.get('graham_number'))} "
                f"| {_pct(d.get('margin_of_safety_vs_graham_pct'))} "
                f"| {_val(d.get('interest_coverage'))} "
                f"| {_pct(d.get('dividend_yield_pct'))} |"
            )
        return "\n".join(rows)

    def _holding_order(self) -> List[str]:
        """Tickers in portfolio weight order - reused by rendering helpers."""
        ordered = sorted(self.holdings, key=lambda h: -h.get("weight", 0))
        return [h.get("ticker", "?") for h in ordered]

    def persona_context_block(self) -> str:
        """Compact block injected into each persona's user message."""
        if not self.has_data():
            return ""
        sum_ = self.summary or {}
        lines = [
            "## User's current portfolio",
            f"- Portfolio: {sum_.get('portfolio_name', 'demo')}",
            f"- Total value: ${sum_.get('total_value_usd', 0):,.2f} "
            f"(absolute P&L {sum_.get('absolute_gain_pct', 0):+.2f}%)",
            f"- Holdings: {sum_.get('holding_count', len(self.holdings))}",
            f"- Geographic split: "
            + ", ".join(
                f"{k} {v}%" for k, v in (sum_.get("geographic_split", {}) or {}).items()
            ),
            "- All amounts in **USD**. Indian holdings' prices are "
            "converted from INR to USD at the rate embedded in the "
            "portfolio fixture metadata (same rate applied to the live "
            "yfinance feed for Indian tickers).",
            "",
            "### Top holdings (by weight)",
            self.top_holdings_table_md(),
            "",
            "### Sector allocation (grouped)",
            self.sector_summary_md(),
            "",
            "### Concentration risks (auto-detected)",
            self.risks_summary_md(),
            "",
            self.score_summary_md(),
        ]
        # Include the live market snapshot so personas can quote specific
        # current numbers without having to re-fetch them.
        snap_md = self.market_snapshot_md()
        if snap_md:
            lines += [
                "",
                "### Live fundamentals snapshot (fetched by the orchestrator)",
                snap_md,
            ]
        # Qualitative enrichment: moat / growth / defensive, all pre-
        # fetched by the orchestrator so the personas reason over full
        # context rather than making one-off tool calls for each holding.
        moat_md = self.moat_signals_md()
        if moat_md:
            lines += [
                "",
                "### Moat signals per holding (fetched by the orchestrator)",
                moat_md,
            ]
        growth_md = self.growth_snapshot_md()
        if growth_md:
            lines += [
                "",
                "### Growth & innovation snapshot (fetched by the orchestrator)",
                growth_md,
            ]
        defensive_md = self.defensive_snapshot_md()
        if defensive_md:
            lines += [
                "",
                "### Defensive / Graham-style snapshot (fetched by the orchestrator)",
                defensive_md,
            ]
        cat_md = self.catalysts_md()
        if cat_md:
            lines += [
                "",
                "### Recent catalysts per holding (fetched by the orchestrator)",
                cat_md,
            ]
        return "\n".join(lines)

    def moderator_context_block(self) -> str:
        """Shorter block for the moderator opening/synthesis prompt."""
        if not self.has_data():
            return ""
        sum_ = self.summary or {}
        top3 = sorted(self.holdings, key=lambda h: -h.get("weight", 0))[:3]
        top3_str = ", ".join(
            f"{h['ticker']} ({round(h['weight']*100,1)}%)" for h in top3
        )
        grouped = self.allocation.get("grouped_sectors_pct", {}) or {}
        sectors_str = ", ".join(
            f"{k} {v}%" for k, v in sorted(grouped.items(), key=lambda kv: -kv[1])[:4]
        )
        risk_lines = self.risks.get("risks", []) or []
        risks_str = "; ".join(r.get("detail", "") for r in risk_lines) or "none flagged"
        # Aggregate P&L — the panel prompts lean on this heavily so make
        # sure it's front-and-centre in the moderator's brief.
        total_gain_pct = sum_.get("absolute_gain_pct")
        gain_str = (
            f" Aggregate P&L: {total_gain_pct:+.2f}% on cost."
            if total_gain_pct is not None
            else ""
        )
        return (
            f"User portfolio: ${sum_.get('total_value_usd', 0):,.2f} across "
            f"{sum_.get('holding_count', len(self.holdings))} holdings "
            f"(all amounts USD). Top: {top3_str}. Sectors: {sectors_str}. "
            f"Diversification {self.score.get('score','?')}/100 "
            f"({self.score.get('band','?')}). Flags: {risks_str}."
            f"{gain_str}"
        )


# ---------------------------------------------------------------------------
# Orchestrator phase: call Portfolio Agent tools directly, emit visible events
# ---------------------------------------------------------------------------
def _parse_mcp_result(raw: Any) -> Any:
    """Normalise ``langchain-mcp-adapters`` tool return values.

    Depending on adapter version, ``tool.ainvoke(...)`` returns one of:

    * a plain Python object (``dict`` / ``list``) - newer adapter
    * a JSON string
    * a list of content parts, e.g.
      ``[{"type": "text", "text": "<json>"}, ...]``

    We collapse all three to the parsed Python value.
    """
    # list-of-content-parts (observed on langchain-mcp-adapters 0.2.x)
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        text = raw[0].get("text", "")
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return text
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


# ---------------------------------------------------------------------------
# Tool-result cache
# ---------------------------------------------------------------------------
# ``langchain-mcp-adapters`` 0.2.x opens a fresh subprocess per ``tool.ainvoke``
# call (observed in practice: 50+ "Starting MCP server" log lines during a
# single panel run). That means the in-subprocess ``TTLCache`` in
# ``_live.py`` / ``_research.py`` never gets a cache hit - it's wiped the
# moment the subprocess exits.
#
# We fix that by caching at the *main-app* level, keyed on
# ``(tool_name, sorted-args)``. This cache survives across all tool calls
# within the finai-api container lifetime, so the second persona who
# asks for ``us_stock__get_fundamentals(ticker="WDC")`` hits this cache
# and skips the entire subprocess + yfinance round-trip.
from cachetools import TTLCache as _TTLCache  # noqa: E402 (placed here for locality)

_TOOL_RESULT_CACHE: _TTLCache = _TTLCache(maxsize=2000, ttl=3600)  # 1 hour
_TOOL_CACHE_HITS = 0
_TOOL_CACHE_MISSES = 0


def _tool_cache_key(tool_name: str, args: Dict[str, Any]) -> str:
    """Stable key for ``(tool, args)``; args order-insensitive."""
    try:
        payload = json.dumps(args or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = str(sorted((args or {}).items()))
    return f"{tool_name}|{payload}"


def tool_cache_stats() -> Dict[str, Any]:
    """Expose tool-cache stats for the /health root endpoint."""
    total = _TOOL_CACHE_HITS + _TOOL_CACHE_MISSES
    hit_rate = (100.0 * _TOOL_CACHE_HITS / total) if total else 0.0
    return {
        "size": len(_TOOL_RESULT_CACHE),
        "maxsize": _TOOL_RESULT_CACHE.maxsize,
        "ttl_seconds": _TOOL_RESULT_CACHE.ttl,
        "hits": _TOOL_CACHE_HITS,
        "misses": _TOOL_CACHE_MISSES,
        "hit_rate_pct": round(hit_rate, 1),
    }


def _note_tool_cache_hit(tool_name: str) -> None:
    global _TOOL_CACHE_HITS
    _TOOL_CACHE_HITS += 1
    log.info(
        "tool CACHE-HIT  %-44s hits=%d/misses=%d",
        tool_name,
        _TOOL_CACHE_HITS,
        _TOOL_CACHE_MISSES,
    )


def _note_tool_cache_miss(tool_name: str) -> None:
    global _TOOL_CACHE_MISSES
    _TOOL_CACHE_MISSES += 1
    log.info(
        "tool CACHE-MISS %-44s hits=%d/misses=%d",
        tool_name,
        _TOOL_CACHE_HITS,
        _TOOL_CACHE_MISSES,
    )


def install_tool_cache_wrappers(tools: List[Any]) -> None:
    """Wrap each tool's underlying coroutine with a main-process TTL cache.

    langchain ``StructuredTool`` / ``BaseTool`` objects are Pydantic models
    that reject arbitrary attribute assignment (``tool.ainvoke = ...``
    raises ``has no field "ainvoke"``). We bypass Pydantic's setattr by
    patching the tool's internal ``coroutine`` / ``func`` callable, which
    is what both ``ainvoke`` and the ReAct loop ultimately dispatch to.

    Idempotent: tools already marked with ``_finai_cached`` on the
    Pydantic ``__dict__`` are skipped so re-registration within the same
    Python process doesn't nest the wrapper.

    Called by :func:`src.core.dispatcher.run_analysis` once per request,
    after ``mcp_servers.get_tools()`` returns fresh ``BaseTool`` objects.
    """
    for tool in tools:
        # Pydantic-safe marker lookup and set (uses the plain __dict__).
        if tool.__dict__.get("_finai_cached"):
            continue

        tool_name = tool.name

        # Figure out which attribute holds the async callable. MCP
        # adapters on langchain 0.2.x populate ``coroutine``; some
        # adapter versions or sync tools use ``func``. We only need to
        # wrap the first one that's present and callable.
        coro_attr = None
        for attr in ("coroutine", "func"):
            if callable(getattr(tool, attr, None)):
                coro_attr = attr
                break
        if coro_attr is None:
            log.warning(
                "Could not find a callable on tool %s; skipping cache wrap",
                tool_name,
            )
            continue

        original = getattr(tool, coro_attr)

        async def _cached(
            *args,
            _original=original,
            _name=tool_name,
            **kwargs,
        ):
            # Normalise args so ``func("WDC")``, ``func(ticker="WDC")``
            # and ``func({"ticker": "WDC"})`` all produce the same cache
            # key. Dict-shaped positional args are merged into kwargs;
            # scalar positionals are kept under a stable sentinel key.
            merged: Dict[str, Any] = {}
            positional_tail: list = []
            for a in args:
                if isinstance(a, dict):
                    merged.update(a)
                else:
                    positional_tail.append(a)
            if positional_tail:
                merged["__args__"] = positional_tail
            if kwargs:
                merged.update(kwargs)
            key = _tool_cache_key(_name, merged)

            if key in _TOOL_RESULT_CACHE:
                _note_tool_cache_hit(_name)
                return _TOOL_RESULT_CACHE[key]

            _note_tool_cache_miss(_name)
            result = await _original(*args, **kwargs)
            parsed = _parse_mcp_result(result)
            if isinstance(parsed, dict) and parsed.get("error"):
                # Don't poison the cache with errors - next retry may succeed.
                return result
            _TOOL_RESULT_CACHE[key] = result
            return result

        # Pydantic-safe attribute replacement: bypass the model's
        # __setattr__ by writing directly to __dict__.
        try:
            object.__setattr__(tool, coro_attr, _cached)
            object.__setattr__(tool, "_finai_cached", True)
        except Exception as e:
            log.warning("Failed to install cache on tool %s: %s", tool_name, e)


async def _call_tool(tool_name: str, args: Dict[str, Any]) -> Any:
    """Call a namespaced MCP tool by name and return the parsed result.

    With the :func:`install_tool_cache_wrappers` monkey-patch in place,
    the ``ainvoke`` call below automatically hits the main-process TTL
    cache - so repeat calls for the same ``(tool, args)`` skip the MCP
    subprocess and the upstream API entirely.
    """
    tools = await mcp_servers.get_tools()
    target = next((t for t in tools if t.name == tool_name), None)
    if target is None:
        raise RuntimeError(f"Tool '{tool_name}' not found in MCP registry")
    result = await target.ainvoke(args)
    return _parse_mcp_result(result)


async def _orchestrator_fetch_portfolio(
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Pre-panel phase: fetch holdings + analytics, emit visible handoffs.

    Yields a sequence of :class:`PanelEvent` dicts (so the streaming adapter
    can render them in real time) and, as its final yield, a special
    ``{"type": "_portfolio_ready", "ctx": PortfolioContext}`` payload
    the caller consumes to pass context to the moderator and personas.
    """
    # Section header is kept (it's structural markdown, not banner-style
    # narration). The "_The orchestrator is now reaching out..._" dev
    # paragraph that used to live here was Fix-3 cleanup — it was
    # dev-trace bleeding into the user response.
    yield {
        "type": "text",
        "text": "## Portfolio Overview\n\n",
        "persona": "orchestrator",
    }

    ctx = PortfolioContext(user_id=user_id)

    # 1. Holdings
    yield {
        "type": "tool_call",
        "persona": "orchestrator",
        "persona_label": "Orchestrator",
        "tool": "portfolio__get_holdings",
        "args": {"user_id": user_id},
    }
    try:
        holdings_resp = await _call_tool("portfolio__get_holdings", {"user_id": user_id})
        ctx.holdings = holdings_resp.get("holdings", [])
        yield {
            "type": "tool_result",
            "persona": "orchestrator",
            "tool": "portfolio__get_holdings",
            "result_preview": (
                f"{holdings_resp.get('holding_count', '?')} holdings · "
                f"total ${holdings_resp.get('total_value_usd', 0):,.2f}"
            ),
        }
    except Exception as e:
        yield {"type": "error", "text": f"Portfolio get_holdings failed: {e}"}
        yield {"type": "_portfolio_ready", "ctx": ctx}
        return

    # 2. Summary
    yield {
        "type": "tool_call",
        "persona": "orchestrator",
        "persona_label": "Orchestrator",
        "tool": "portfolio__get_portfolio_summary",
        "args": {"user_id": user_id},
    }
    try:
        ctx.summary = await _call_tool(
            "portfolio__get_portfolio_summary", {"user_id": user_id}
        )
        top3 = (ctx.summary.get("top_5_holdings") or [])[:3]
        top_str = ", ".join(f"{h['ticker']} {h['weight_pct']}%" for h in top3)
        yield {
            "type": "tool_result",
            "persona": "orchestrator",
            "tool": "portfolio__get_portfolio_summary",
            "result_preview": f"top 3: {top_str}",
        }
    except Exception as e:
        yield {"type": "error", "text": f"Portfolio summary failed: {e}"}

    # 3. Sector allocation
    yield {
        "type": "tool_call",
        "persona": "orchestrator",
        "persona_label": "Orchestrator",
        "tool": "portfolio__get_sector_allocation",
        "args": {"user_id": user_id},
    }
    try:
        ctx.allocation = await _call_tool(
            "portfolio__get_sector_allocation", {"user_id": user_id}
        )
        grouped = ctx.allocation.get("grouped_sectors_pct", {}) or {}
        top_sector = (
            max(grouped.items(), key=lambda kv: kv[1])
            if grouped
            else ("?", 0)
        )
        yield {
            "type": "tool_result",
            "persona": "orchestrator",
            "tool": "portfolio__get_sector_allocation",
            "result_preview": f"largest bucket: {top_sector[0]} at {top_sector[1]}%",
        }
    except Exception as e:
        yield {"type": "error", "text": f"Sector allocation failed: {e}"}

    # 4. Concentration risks
    yield {
        "type": "tool_call",
        "persona": "orchestrator",
        "persona_label": "Orchestrator",
        "tool": "portfolio__get_concentration_risks",
        "args": {"user_id": user_id},
    }
    try:
        ctx.risks = await _call_tool(
            "portfolio__get_concentration_risks", {"user_id": user_id}
        )
        n = ctx.risks.get("risk_count", 0)
        yield {
            "type": "tool_result",
            "persona": "orchestrator",
            "tool": "portfolio__get_concentration_risks",
            "result_preview": f"{n} risk flag{'s' if n != 1 else ''} detected",
        }
    except Exception as e:
        yield {"type": "error", "text": f"Concentration risks failed: {e}"}

    # 5. Diversification score
    yield {
        "type": "tool_call",
        "persona": "orchestrator",
        "persona_label": "Orchestrator",
        "tool": "portfolio__get_diversification_score",
        "args": {"user_id": user_id},
    }
    try:
        ctx.score = await _call_tool(
            "portfolio__get_diversification_score", {"user_id": user_id}
        )
        yield {
            "type": "tool_result",
            "persona": "orchestrator",
            "tool": "portfolio__get_diversification_score",
            "result_preview": (
                f"score {ctx.score.get('score','?')}/100 ({ctx.score.get('band','?')})"
            ),
        }
    except Exception as e:
        yield {"type": "error", "text": f"Diversification score failed: {e}"}

    # Render a compact "portfolio at a glance" panel for the user
    yield {
        "type": "text",
        "text": (
            "\n#### Your Portfolio at a Glance\n\n"
            + (ctx.top_holdings_table_md() or "_no holdings found_")
            + "\n\n**Sector allocation**\n"
            + (ctx.sector_summary_md() or "_(n/a)_")
            + "\n\n**Concentration flags**\n"
            + ctx.risks_summary_md()
            + "\n\n"
            + ctx.score_summary_md()
            + "\n\n---\n\n"
        ),
        "persona": "orchestrator",
    }

    # ------------------------------------------------------------------
    # Market Snapshot phase: live fundamentals + catalysts for the top
    # holdings. Separate section so the response tells a full story
    # (data -> snapshot -> debate -> synthesis) rather than being
    # entirely panel-driven.
    # ------------------------------------------------------------------
    async for ev in _orchestrator_market_snapshot(ctx):
        yield ev

    # ------------------------------------------------------------------
    # Qualitative Enrichment phase: pre-fetch moat signals, growth
    # metrics and defensive metrics for every holding so the panel
    # personas see the full qualitative picture (not just price +
    # fundamentals) in their context window.
    # ------------------------------------------------------------------
    async for ev in _orchestrator_qualitative_enrichment(ctx):
        yield ev

    # Handshake payload for run_panel to consume
    yield {"type": "_portfolio_ready", "ctx": ctx}


# ---------------------------------------------------------------------------
# Market Snapshot sub-phase
# ---------------------------------------------------------------------------
# Cap on the number of top holdings to deep-dive on in the Market Snapshot.
# Previously defaulted to 5, but personas (especially Graham, who likes to
# check defensive metrics on every holding) were re-fetching the rest
# themselves - burning their recursion budget and duplicating work the
# orchestrator could have done once. Setting this very high effectively
# lets the orchestrator pre-fetch the entire portfolio so personas can
# focus on reasoning over shared data instead of tool-call plumbing. The
# per-holding calls are 1-hour-cached inside the Stock/Research workers,
# so even a full 20-holding portfolio costs seconds after the first run.
_SNAPSHOT_TOP_N = 50


def _stock_tool_for(holding: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Pick which Stock Agent to ask about a given holding + the call args."""
    country = (holding.get("country") or "").upper()
    ticker = holding.get("ticker", "")
    if country == "US":
        return "us_stock__get_fundamentals", {"ticker": ticker}
    # Default to Indian for anything else (our fixture is IN-centric).
    return "indian_stock__get_fundamentals", {"ticker": ticker}


async def _orchestrator_market_snapshot(
    ctx: PortfolioContext,
) -> AsyncIterator[PanelEvent]:
    """Fetch live fundamentals + recent catalysts for the top-N holdings.

    Emits one ``tool_call`` event per handoff so the audience sees the
    orchestrator reaching out to **multiple** named agents (not just
    Portfolio). Populates ``ctx.market_snapshot`` and ``ctx.catalysts``
    in place; the caller will render those blocks inside every persona's
    prompt.
    """
    if not ctx.holdings:
        return

    holdings_to_snap = sorted(ctx.holdings, key=lambda h: -h.get("weight", 0))[
        :_SNAPSHOT_TOP_N
    ]
    snap_count = len(holdings_to_snap)
    # Section header only - the dev-narration paragraph that explained
    # what the orchestrator was about to do was removed in Fix 3 to
    # match Claude-style "show, don't tell" UX.
    yield {
        "type": "text",
        "text": f"\n## Market Snapshot — {snap_count} Holdings\n\n",
        "persona": "orchestrator",
    }

    for holding in holdings_to_snap:
        ticker = holding.get("ticker", "?")
        name = holding.get("name", ticker)
        weight_pct = round(holding.get("weight", 0) * 100, 1)

        yield {
            "type": "text",
            "text": f"**{ticker}** — {name} ({weight_pct}%)\n",
            "persona": "orchestrator",
        }

        # 1) Live fundamentals from the appropriate Stock Agent
        tool_name, tool_args = _stock_tool_for(holding)
        yield {
            "type": "tool_call",
            "persona": "orchestrator",
            "persona_label": "Orchestrator",
            "tool": tool_name,
            "args": tool_args,
        }
        try:
            fund = await _call_tool(tool_name, tool_args)
            ctx.market_snapshot[ticker] = fund or {}
            preview = (
                f"price {fund.get('price') or '-'} · "
                f"P/E {fund.get('pe_ttm') or fund.get('forward_pe') or '-'} · "
                f"{fund.get('_source', 'n/a')}"
            )
            yield {
                "type": "tool_result",
                "persona": "orchestrator",
                "tool": tool_name,
                "result_preview": preview,
            }
        except Exception as e:
            log.warning("Market snapshot fundamentals failed for %s: %s", ticker, e)
            ctx.market_snapshot[ticker] = {}
            yield {
                "type": "error",
                "text": f"Fundamentals for {ticker} failed: {e}",
            }

        # 2) Recent catalysts from the Research Agent
        yield {
            "type": "tool_call",
            "persona": "orchestrator",
            "persona_label": "Orchestrator",
            "tool": "research__search_news",
            "args": {"ticker": ticker, "max_items": 3},
        }
        try:
            news_resp = await _call_tool(
                "research__search_news", {"ticker": ticker, "max_items": 3}
            )
            news_items = news_resp.get("news", []) if isinstance(news_resp, dict) else []
            ctx.catalysts[ticker] = news_items
            preview = (
                f"{len(news_items)} item{'s' if len(news_items) != 1 else ''}"
                + (f" · latest: {news_items[0].get('headline', '')[:55]}" if news_items else "")
            )
            yield {
                "type": "tool_result",
                "persona": "orchestrator",
                "tool": "research__search_news",
                "result_preview": preview,
            }
        except Exception as e:
            log.warning("Market snapshot news failed for %s: %s", ticker, e)
            ctx.catalysts[ticker] = []
            yield {
                "type": "error",
                "text": f"News for {ticker} failed: {e}",
            }

    # Render the consolidated Market Snapshot table + catalyst bullets
    snap_md = ctx.market_snapshot_md()
    cat_md = ctx.catalysts_md()
    chunks: List[str] = [
        f"\n#### Live Fundamentals ({snap_count} holdings)\n"
    ]
    if snap_md:
        chunks.append(snap_md + "\n")
    else:
        chunks.append("_No live fundamentals available._\n")
    chunks.append("\n#### Recent Catalysts\n")
    if cat_md:
        chunks.append(cat_md + "\n")
    else:
        chunks.append("_No catalysts found for these holdings._\n")
    chunks.append("\n---\n\n")
    yield {
        "type": "text",
        "text": "".join(chunks),
        "persona": "orchestrator",
    }


# ---------------------------------------------------------------------------
# Qualitative Enrichment sub-phase
# ---------------------------------------------------------------------------
def _qualitative_tools_for(holding: Dict[str, Any]) -> List[tuple[str, Dict[str, Any]]]:
    """Return the (tool_name, args) triples for a holding's moat / growth / defensive fetches."""
    country = (holding.get("country") or "").upper()
    ticker = holding.get("ticker", "")
    ns = "us_stock" if country == "US" else "indian_stock"
    return [
        (f"{ns}__get_moat_signals", {"ticker": ticker}),
        (f"{ns}__get_growth_metrics", {"ticker": ticker}),
        (f"{ns}__get_defensive_metrics", {"ticker": ticker}),
    ]


async def _orchestrator_qualitative_enrichment(
    ctx: PortfolioContext,
) -> AsyncIterator[PanelEvent]:
    """Fetch moat / growth / defensive data for every holding in parallel.

    Personas used to each re-fetch this per ticker during their ReAct
    loops (blowing through their recursion budget and serialising the
    work three times - once per persona). Doing it once here, in
    parallel, means:

      * Every persona sees the full qualitative picture in-context.
      * Round-1 panel latency drops (personas can skip these tool calls).
      * Cached results benefit the whole debate, not just one persona.

    We fire all ``len(ctx.holdings) * 3`` calls in a single
    :func:`asyncio.gather`; the main-process tool cache keeps repeat
    calls instant, so this phase typically completes in a few seconds.
    """
    if not ctx.holdings:
        return

    # Section header only - dev-narration removed in Fix 3.
    yield {
        "type": "text",
        "text": "\n## Qualitative Enrichment\n\n",
        "persona": "orchestrator",
    }

    # Emit three "bulk" handoff events so the UI shows what's happening
    # without flooding with 33+ individual lines.
    for tool_kind, label in [
        ("get_moat_signals", "moat signals"),
        ("get_growth_metrics", "growth metrics"),
        ("get_defensive_metrics", "defensive metrics"),
    ]:
        yield {
            "type": "tool_call",
            "persona": "orchestrator",
            "persona_label": "Orchestrator",
            "tool": f"us_stock|indian_stock__{tool_kind}",
            "args": {"bulk": f"{len(ctx.holdings)} holdings in parallel"},
        }

    # Fire everything concurrently — but throttle.
    #
    # ``langchain-mcp-adapters`` 0.2.x spawns a fresh subprocess per tool
    # call. Firing 33+ in parallel overwhelms the adapter (we saw silent
    # hangs past the first handful of inflight requests). A small
    # semaphore keeps the concurrency healthy while still running a
    # genuine "enrichment in parallel" story for the UI.
    _ENRICH_CONCURRENCY = 5
    sem = asyncio.Semaphore(_ENRICH_CONCURRENCY)

    async def _limited_call(tool_name: str, args: Dict[str, Any]):
        async with sem:
            return await _call_tool(tool_name, args)

    log.info(
        "ENRICH: creating %d tasks for %d holdings (sem=%d)",
        len(ctx.holdings) * 3, len(ctx.holdings), _ENRICH_CONCURRENCY,
    )
    tasks: Dict[tuple[str, str], Any] = {}
    for h in ctx.holdings:
        ticker = h.get("ticker", "?")
        for tool_name, tool_args in _qualitative_tools_for(h):
            kind = tool_name.split("__", 1)[-1]
            tasks[(ticker, kind)] = asyncio.create_task(
                _limited_call(tool_name, tool_args)
            )
    log.info("ENRICH: %d tasks created, starting collection", len(tasks))

    # Collect results (individual failures don't break the whole phase).
    for (ticker, kind), task in tasks.items():
        try:
            result = await task
        except Exception as e:
            log.warning(
                "ENRICH: %s.%s raised: %s", ticker, kind, e
            )
            continue
        if not isinstance(result, dict):
            log.warning("ENRICH: %s.%s returned non-dict: %s", ticker, kind, type(result).__name__)
            continue
        if kind == "get_moat_signals":
            signals = result.get("moat_signals") or []
            if signals:
                ctx.moat_signals[ticker] = signals
        elif kind == "get_growth_metrics":
            # Strip meta + ticker so the rendering methods don't need
            # to re-filter noise fields.
            slim = {
                k: v
                for k, v in result.items()
                if not k.startswith("_") and k not in ("ticker", "name")
            }
            if slim:
                ctx.growth[ticker] = slim
        elif kind == "get_defensive_metrics":
            slim = {
                k: v
                for k, v in result.items()
                if not k.startswith("_") and k not in ("ticker", "name")
            }
            if slim:
                ctx.defensive[ticker] = slim

    # Emit a result summary for each tool-kind bulk.
    for tool_kind, store_attr in [
        ("get_moat_signals", "moat_signals"),
        ("get_growth_metrics", "growth"),
        ("get_defensive_metrics", "defensive"),
    ]:
        filled = sum(1 for h in ctx.holdings if getattr(ctx, store_attr).get(h.get("ticker")))
        yield {
            "type": "tool_result",
            "persona": "orchestrator",
            "tool": f"us_stock|indian_stock__{tool_kind}",
            "result_preview": f"{filled}/{len(ctx.holdings)} holdings enriched",
        }

    # Render the enrichment so the user sees it in the transcript.
    chunks: List[str] = []
    moat = ctx.moat_signals_md()
    growth = ctx.growth_snapshot_md()
    defensive = ctx.defensive_snapshot_md()
    if moat:
        chunks.append("\n#### Moat Signals\n\n" + moat + "\n")
    if growth:
        chunks.append("\n#### Growth & Innovation Snapshot\n\n" + growth + "\n")
    if defensive:
        chunks.append("\n#### Defensive / Graham-style Snapshot\n\n" + defensive + "\n")
    if chunks:
        chunks.append("\n---\n\n")
        yield {
            "type": "text",
            "text": "".join(chunks),
            "persona": "orchestrator",
        }


# ---------------------------------------------------------------------------
# Per-persona streaming runner (unchanged except for portfolio_context param)
# ---------------------------------------------------------------------------
class VerdictTrimFilter:
    """Streaming text filter that suppresses everything from ``VERDICT:`` onward.

    The persona's final AI message ends with a machine-readable trailer:

        VERDICT: <one sentence>
        STANCE: <bullish|neutral|cautious|bearish>
        CONFIDENCE: <low|medium|high>

    The supervisor re-renders this as a styled verdict card after the
    streamed text, so showing the raw trailer in the live transcript would
    be duplication.
    """

    MARKER = "VERDICT:"

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0
        self.closed = False

    def push(self, delta: str) -> str:
        if self.closed or not delta:
            return ""
        self._buf += delta
        idx = self._buf.find(self.MARKER)
        if idx != -1:
            self.closed = True
            visible = self._buf[self._emitted : idx].rstrip()
            self._emitted = len(self._buf)
            return visible
        safe_end = len(self._buf) - (len(self.MARKER) - 1)
        if safe_end <= self._emitted:
            return ""
        piece = self._buf[self._emitted : safe_end]
        self._emitted = safe_end
        return piece

    def flush(self) -> str:
        if self.closed:
            return ""
        tail = self._buf[self._emitted :]
        self._emitted = len(self._buf)
        return tail


def _compose_persona_user_message(
    query: str, ctx: Optional[PortfolioContext]
) -> str:
    """Build the user message each persona sees, injecting portfolio context."""
    if ctx is None or not ctx.has_data():
        return query
    return (
        f"User's question: {query}\n\n"
        f"{ctx.persona_context_block()}\n\n"
        "Analyse this portfolio through your lens. You may use your MCP tools "
        "to dig deeper on any specific holding. Focus on 1-3 positions that "
        "best illustrate your framework, cite specific numbers from the "
        "portfolio context or your tool calls, and end with an overall view "
        "of the portfolio's health. Remember: this is educational, not "
        "personalised investment advice."
    )


async def _stream_persona_events(
    persona: PersonaDef,
    query: str,
    portfolio_context: Optional[PortfolioContext] = None,
    *,
    user_message_override: Optional[str] = None,
) -> AsyncIterator[PanelEvent]:
    """Run a persona's ReAct loop and yield events in real time.

    ``user_message_override`` lets callers (notably
    :mod:`src.core.debate`) supply a fully-custom user message - e.g.
    one that embeds the multi-round debate scratchpad - without having
    to fork the whole ReAct pipeline. When ``None`` we fall back to the
    standard ``query + portfolio context`` composer.
    """
    runner = build_persona_agent(persona)
    graph = getattr(runner, "graph")
    if user_message_override is not None:
        user_message = user_message_override
    else:
        user_message = _compose_persona_user_message(query, portfolio_context)

    final_text_parts: List[str] = []
    tools_used: List[str] = []
    verdict_filter: Optional[VerdictTrimFilter] = None
    # ReAct agents typically trigger ``on_chat_model_start`` N+1 times (one
    # per tool round-trip). Emitting "<persona> is thinking..." on every
    # single start produced a stutter of identical lines between each tool
    # handoff. We now emit it ONCE - on the first model start for this
    # turn - and rely on the streamed tool_call and text events to carry
    # the visible progress from there.
    thought_banner_emitted = False

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=user_message)]},
        version="v2",
        config={"recursion_limit": PERSONA_RECURSION_LIMIT},
    ):
        kind = event.get("event")

        if kind == "on_chat_model_start":
            # Reset the verdict filter every LLM turn (the final answer
            # may come from any turn, and each produces its own tokens).
            verdict_filter = VerdictTrimFilter()
            if not thought_banner_emitted:
                yield {
                    "type": "text",
                    "text": f"\n\n_{persona.title} is thinking…_\n\n",
                    "persona": persona.name,
                }
                thought_banner_emitted = True

        elif kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", None) if chunk is not None else None
            if content and verdict_filter is not None:
                visible = verdict_filter.push(content)
                if visible:
                    yield {
                        "type": "text",
                        "text": visible,
                        "persona": persona.name,
                    }

        elif kind == "on_chat_model_end":
            if verdict_filter is not None and not verdict_filter.closed:
                tail = verdict_filter.flush()
                if tail:
                    yield {
                        "type": "text",
                        "text": tail,
                        "persona": persona.name,
                    }
            verdict_filter = None
            output = event.get("data", {}).get("output")
            content = getattr(output, "content", None) if output is not None else None
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            if content.strip():
                final_text_parts.append(content)

        elif kind == "on_tool_start":
            tool_name = event.get("name") or "?"
            args = event.get("data", {}).get("input") or {}
            if tool_name not in tools_used:
                tools_used.append(tool_name)
            yield {
                "type": "tool_call",
                "persona": persona.name,
                "persona_label": persona.title,
                "tool": tool_name,
                "args": args,
            }

    final_text = "".join(final_text_parts)
    verdict = parse_verdict(final_text, persona)
    yield {
        "type": "persona_verdict",
        "persona": persona.name,
        "title": persona.title,
        "stance": verdict.get("stance", "neutral"),
        "one_liner": verdict.get("one_liner", ""),
        "confidence": verdict.get("confidence", "low"),
        "tools_used": tools_used,
    }


# ---------------------------------------------------------------------------
# NOTE: the top-level request entry point used to live here as ``run_panel``.
# It is now :func:`src.core.dispatcher.run_analysis`, which:
#
#   1. Classifies the query via :mod:`src.core.router` (one small LLM call).
#   2. Emits a visible "Query Classification" section so the audience sees
#      which flow was picked and why.
#   3. Dispatches to one of four flows in :mod:`src.core.flows`:
#
#          portfolio_analysis  - current full panel flow (lives in
#                                ``src.core.flows.portfolio_analysis``)
#          stock_research      - focused deep dive on specific ticker(s)
#          topic_research      - web research on a macro / sector topic
#          educational         - direct LLM explanation, zero agents
#
#   4. Appends a universal disclaimer footer + terminal event.
#
# The heavy lifting helpers (``PortfolioContext``, orchestrator phases,
# ``_stream_persona_events``, etc.) defined above are now shared across
# all flows that need them.
# ---------------------------------------------------------------------------
