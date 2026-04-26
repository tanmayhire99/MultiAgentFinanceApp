"""Meta-help flow — answers questions about FinAI itself.

Triggered by the router when the user asks meta-style questions about the
*system* rather than the *market* — things like "what can you do?",
"what are your capabilities?", "who are you?", "how does this work?".

This flow is **deliberately LLM-free**. The capabilities listing is a
curated, hand-written description of what the system actually offers.
That gives us three properties the educational flow couldn't:

1. **Zero hallucination.** A real LLM, given a generic-educator prompt,
   confidently said "I don't have real-time internet access" — which is
   blatantly false (FinAI has Yahoo Finance live quotes + Tavily/DDG
   web search + SEC EDGAR + BSE/Screener filings). Static text avoids
   that failure mode entirely.
2. **Sub-second response.** No model spin-up, no streaming latency.
   Meta-questions are usually impatient ("can this even do X?"),
   so a snappy answer is what the user wants.
3. **Always accurate.** When we add a new flow / agent / data source,
   we update this file; the user always sees the current truth.

If a future need surfaces for *personalised* meta answers (e.g. answering
"can you do X for ticker Y?" with a tailored response), we can add a
small LLM tail to this flow — but the curated trunk should always run
first so the basics are guaranteed correct.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from src.core.panel import PanelEvent
from src.core.router import RouteDecision


log = logging.getLogger("finai.flows.meta_help")


# ---------------------------------------------------------------------------
# The capabilities answer.
#
# Edit this when you add/remove a flow, change a data source, or alter the
# panel composition. The tests in tests/test_flows_meta_help.py verify a few
# invariants (mentions every flow + every data source + the persona panel).
# ---------------------------------------------------------------------------
_HEADER = "# 🤖 FinAI — What this assistant can do\n\n"

_INTRO = (
    "I'm **FinAI** — a multi-agent financial-analysis assistant. I'm not "
    "a generic chatbot: every query is routed to one of five specialised "
    "flows, each backed by real data sources, real LLM agents, and "
    "(usually) a structured markdown report.\n\n"
    "**What's different from a plain LLM:** I have live market data, SEC "
    "filings, Indian regulator filings (BSE / Screener / NSE), web search, "
    "a multi-persona investor panel, and a claim-tracking sub-system. I "
    "do not invent prices or fundamentals — every number you see is "
    "fetched at request time from a named data source.\n\n"
)

_FLOWS_TABLE = (
    "## The 5 query types I route to\n\n"
    "| If you ask… | I route to | What you get |\n"
    "|---|---|---|\n"
    "| _\"How's my portfolio looking?\"_ | **Portfolio Analysis** | Full Buffett / "
    "Wood / Graham investor panel debate over your holdings, plus a "
    "synthesised verdict. ~60–180 s. |\n"
    "| _\"Tell me about NVDA\"_ / _\"Should I look at TCS?\"_ | "
    "**Stock Research** | Live quote, fundamentals, growth metrics, "
    "valuation, recent news, optional persona views. US + Indian tickers. "
    "~15–40 s. |\n"
    "| _\"Did Tesla deliver on FSD?\"_ / _\"Has TCS kept its margin "
    "guidance?\"_ | **Deep Stock Research** | Multi-step claim tracking: "
    "SEC EDGAR filings (10-K / 10-Q / 8-K) + Indian filings (BSE / annual "
    "reports / concall transcripts) + historical news archaeology + "
    "promises-vs-reality scorecard. ~2–6 minutes. |\n"
    "| _\"Impact of US-China chip tensions\"_ / _\"Outlook for Indian IT "
    "sector\"_ | **Topic Research** | Open-ended web research with "
    "Tavily / DuckDuckGo + an LLM-synthesised brief. ~20–40 s. |\n"
    "| _\"Explain compound interest\"_ / _\"What is beta?\"_ | "
    "**Educational** | A direct concept explanation — no tool calls, no "
    "market data. Just a structured definition + worked example. ~3–8 s. |"
    "\n\n"
)

_PANEL = (
    "## The investor panel (Buffett / Wood / Graham)\n\n"
    "When you ask for a panel — _\"panel view\"_, _\"what would the experts "
    "say?\"_, or name any of the personas — I run three sub-agents in "
    "parallel:\n\n"
    "* **Warren Buffett** — value & moat lens. Looks at unit economics, "
    "competitive advantage, and price-vs-intrinsic-value.\n"
    "* **Cathie Wood** — innovation & growth lens. Looks at TAM, S-curve "
    "adoption, and disruption dynamics.\n"
    "* **Benjamin Graham** — defensive lens. Margin of safety, balance "
    "sheet conservatism, downside scenarios.\n\n"
    "Each persona is a real ReAct agent with its own MCP tool budget; they "
    "produce structured verdicts (stance + one-liner + confidence) which a "
    "**Moderator** agent then synthesises into a single ranked decision.\n\n"
)

_DATA_SOURCES = (
    "## Data sources I can pull from\n\n"
    "| Source | What it covers |\n"
    "|---|---|\n"
    "| **Yahoo Finance (live)** | Real-time quotes, fundamentals, "
    "historical prices for US + Indian tickers. |\n"
    "| **SEC EDGAR** | Full-text 10-K, 10-Q, 8-K, S-1 for US-listed "
    "companies (claim tracking source). |\n"
    "| **BSE Filings** | Indian regulatory filings, results announcements, "
    "investor presentations. |\n"
    "| **Screener.in** | Indian fundamentals, peer comparisons, "
    "shareholding patterns. |\n"
    "| **NSE** | Indian quotes + corporate announcements. |\n"
    "| **Tavily / DuckDuckGo** | Web search for news, analyst takes, "
    "macro context. Date-filtered when claim-tracking. |\n"
    "| **Curated fixtures** | Demo-safety fallback if a live API is down. "
    "Always disclosed in the response. |\n\n"
)

_LIMITS = (
    "## What I won't do\n\n"
    "* **No personalised investment advice.** I'll show you the numbers "
    "and let three personas debate, but the final decision is yours (or "
    "your SEBI-registered advisor's).\n"
    "* **No buy/sell recommendations.** A panel verdict is _analysis_, "
    "not a trade signal.\n"
    "* **No execution.** I don't connect to a broker. I don't place "
    "orders.\n"
    "* **No personal-data retention across sessions.** Every query is "
    "stateless.\n\n"
)

_FOLLOWUP = (
    "## Try one of these to see me in action\n\n"
    "* `tell me about WDC` — Stock Research (US ticker)\n"
    "* `look at INFY with a panel view` — Stock Research + Investor Panel\n"
    "* `analyse my portfolio` — Portfolio Analysis (uses the demo fixture)\n"
    "* `did Tesla deliver on FSD timelines?` — Deep Stock Research "
    "(claim tracking; ~3-5 min)\n"
    "* `what is the difference between IRR and ROI?` — Educational\n"
)


# ---------------------------------------------------------------------------
# Special case: user asked for a report but didn't name a subject
# ---------------------------------------------------------------------------
# A query like "generate detailed report" trips ``wants_artifact=True``
# in the dispatcher (because the user clearly wants the artifact pane)
# but the LLM router can't pin a subject - no ticker, no portfolio
# reference, no concept word - so it falls back to ``meta_help``. The
# right answer is NOT to dump the full capabilities listing on them;
# it's to ask one focused question: "what subject?".
_REPORT_REQUEST_NUDGE = (
    "## Sure — what would you like a report on?\n\n"
    "I can generate the report into the artifact side pane, but I "
    "need a subject. Try one of these:\n\n"
    "| Want… | Type something like… |\n"
    "|---|---|\n"
    "| A single-stock deep-dive | `tell me about WDC, generate report` |\n"
    "| A portfolio review | `analyse my portfolio, generate report` |\n"
    "| Promises-vs-reality on a stock | `did Tesla deliver on FSD?, generate report` |\n"
    "| A macro / sector brief | `outlook for Indian IT, generate report` |\n\n"
    "_Without a subject I can't pull live data — every number FinAI "
    "shows is fetched at request time from a named source, so I need "
    "to know which company / portfolio / topic to fetch._\n"
)

_REPORT_KEYWORDS = ("report", "artifact", "document")


def _looks_like_subjectless_report_request(
    query: str, decision: Optional[RouteDecision]
) -> bool:
    """True iff the user clearly asked for a report but named no subject.

    Two signals must coincide:
    * the dispatcher set ``decision["wants_artifact"]=True`` (user
      intent to see the side pane), AND
    * the query mentions a report / artifact / document keyword (so
      the request was really about a *report*, not about FinAI itself).

    If either signal is missing, the user is just asking generic
    "what can you do?" - we show the full capabilities listing.
    """
    if not decision:
        return False
    if not decision.get("wants_artifact"):
        return False
    q_lower = query.lower()
    return any(kw in q_lower for kw in _REPORT_KEYWORDS)


def _capabilities_text() -> str:
    """Single source of truth for the static capabilities response.

    Exposed at module level (rather than inlined into ``run``) so tests
    can import and assert against the rendered text without spinning up
    a flow.
    """
    return _HEADER + _INTRO + _FLOWS_TABLE + _PANEL + _DATA_SOURCES + _LIMITS + _FOLLOWUP


async def run(
    query: str,
    decision: Optional[RouteDecision] = None,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Emit the curated FinAI capabilities answer.

    Two response shapes:

    * Plain meta query ("what can you do?", "tell me about FinAI") →
      stream the full capabilities listing (the original Fix 1
      behaviour).
    * Report-without-subject ("generate detailed report",
      "/report ", "make me a comprehensive document") → stream a
      short, focused nudge asking *what* the user wants the report on.
      Suppresses the long capabilities tour because the user has
      shown they know FinAI exists - they just need a subject prompt.

    Both paths make ZERO LLM calls (static markdown only).
    """
    log.debug("meta_help flow invoked for query=%r", query[:80])

    if _looks_like_subjectless_report_request(query, decision):
        yield {
            "type": "text",
            "text": _REPORT_REQUEST_NUDGE,
            "persona": "orchestrator",
        }
        return

    yield {
        "type": "text",
        "text": _capabilities_text(),
        "persona": "orchestrator",
    }
