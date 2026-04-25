"""Intent router for the FinAI multi-agent system.

A small LLM call (GPT-OSS-120B on NVIDIA NIM, same model the personas use)
classifies each incoming user query into one of four flows so the system
can invoke the *right* agents instead of running the full investor panel
for every request.

=========================  ==================================================
Intent                     Triggers / purpose
=========================  ==================================================
``portfolio_analysis``     User asks about their own portfolio / holdings.
                           Full panel flow: Portfolio Agent ->
                           Market Snapshot -> 3 persona debate -> synthesis.
``stock_research``         User asks about specific ticker(s). Focused
                           deep dive (no portfolio, no debate unless
                           explicitly requested via ``want_panel``).
``topic_research``         Open-ended macro / sector / thematic question.
                           Research Agent web search + LLM summary.
``educational``            Concept explanation (no market data needed).
                           Single LLM call, zero agents, zero tools.
=========================  ==================================================

Conversation context
--------------------
The router can optionally read the **previous assistant reply** so
ambiguous follow-up queries like "do panel analysis as well" or "now
look at P/E" chain correctly onto the prior turn (inheriting the
previous ticker, flipping ``want_panel=True``, etc.). Context is
extracted from our own classification card in the prior reply, not
from freeform text, so the router's input stays small and structured.

Reliability
-----------
* Shorter system prompt + ``max_tokens=800`` keeps output well inside
  the budget for a full JSON response with a rationale.
* JSON output schema puts ``intent`` + ``tickers`` first so even a
  truncated completion preserves the critical routing information.
* Every failure mode (LLM error, unparseable JSON, unknown intent,
  empty tickers in a clear stock query) falls back to a deterministic
  rule that avoids the "every query runs the panel" failure mode.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.personas.base import build_chat_model


log = logging.getLogger("finai.router")


INTENTS = (
    "portfolio_analysis",
    "stock_research",
    "topic_research",
    "educational",
    "deep_stock_research",
)


INTENT_LABELS = {
    "portfolio_analysis": "Portfolio Analysis",
    "stock_research": "Stock Research",
    "topic_research": "Topic Research",
    "educational": "Educational",
    "deep_stock_research": "Deep Stock Research",
}


class RouteDecision(TypedDict, total=False):
    """Structured routing decision returned by :func:`classify_query`."""

    intent: str  # one of INTENTS
    tickers: List[str]  # uppercase, e.g. ["WDC", "NVDA"]
    topic: str  # short subject phrase, e.g. "Western Digital"
    want_panel: bool  # user explicitly requested persona views / debate
    rationale: str  # one-line explanation of the intent choice


# ---------------------------------------------------------------------------
# System prompt (kept compact so the whole call fits inside max_tokens)
# ---------------------------------------------------------------------------
_SYSTEM = """You route queries for a multi-agent finance assistant. Pick ONE intent and emit strict JSON.

## INTENTS

1) portfolio_analysis — user refers to THEIR OWN portfolio / holdings.
   Signals: "my portfolio", "my holdings", "my stocks", "my positions", "my investments", "what do I own".

2) stock_research — user asks about specific stock(s), not necessarily held.
   Signals: company name / ticker, "research X", "tell me about X", "analyse X", "should I buy X".

3) deep_stock_research — user wants a LONG, multi-step deep dive on ONE (or a few) tickers that goes BEYOND current metrics: historical claim tracking (did management deliver on past guidance?), SEC filings review (10-K / 10-Q / 8-K), multi-quarter news archaeology, promises-vs-reality analysis. This is a BATCH-mode flow (2-5 minutes); only pick it when the user is asking for that kind of depth.
   Signals: "deep dive", "deep research", "deep analysis", "thorough research", "claim tracking", "claims vs reality", "promises vs reality", "promised vs delivered", "verify guidance", "did they deliver", "follow through on", "stood on their claims", "what did they promise", "past claims", "long-form research", "full research report", "10-K review", "earnings history", "track record of guidance".
   This intent ALWAYS implies tickers; if no ticker is found, prefer stock_research instead and it will ask the user.

4) topic_research — open-ended macro / sector / theme question, no single ticker in focus.
   Signals: "impact of", "trends in", "outlook on", "what's happening in", "sector view".

5) educational — concept / definition question. No market data needed.
   Signals: "explain", "what is", "how does", "teach me", "difference between".

## PANEL FLAG

In THIS finance app, "panel" / "panel analysis" / "panel view" / "panel debate" / "what do the experts think" / "Buffett / Wood / Graham's view" ALWAYS refers to the investor panel (Buffett / Wood / Graham).

Set want_panel=true whenever the user asks for a panel, debate, experts' opinions, or the three personas by name. Otherwise false.

Routing rules when want_panel is true:
- If the user also names a specific ticker / company → stock_research.
- Else if a Previous turn has stock tickers → stock_research (inherit those tickers).
- Else if a Previous turn is portfolio_analysis or refers to "my portfolio" → portfolio_analysis.
- Otherwise → portfolio_analysis (default: run the panel on the user's portfolio).

Do NOT route "panel" queries to topic_research or educational. Exception: statistical "panel data" / "panel regression" / "fixed-effects panel" / "econometric panel" queries are educational (statistical technique, not investor panel) — those explicitly say "data", "regression", or "fixed effects".

## CONVERSATION CONTEXT

If a "Previous turn" block is provided, the current query might be a follow-up. Rules:

- If the current query is short and refers to "it / this / that / also / as well / now" → INHERIT the previous intent AND the previous tickers.
- If the current query adds "panel / debate / opinions" to a prior stock_research or portfolio_analysis → keep the prior intent, set want_panel=true, inherit tickers.
- If the current query explicitly names a DIFFERENT ticker / company / topic → switch to the new focus; do NOT inherit.
- If no previous turn is given, ignore these rules.

## ENTITY EXTRACTION

- tickers: list of stock tickers mentioned in the query OR inferred from well-known company names ("Western Digital" → "WDC"). Uppercase only. Include inherited tickers from context when the current query is a follow-up (see above). Empty list if nothing applies.
- topic: 2-6 word phrase of the subject.
- want_panel: boolean per the PANEL FLAG section.
- rationale: ONE short sentence (<= 20 words) naming the intent and the signal you used.

## OUTPUT

Return a single JSON object. No markdown, no code fences, no prose. Put fields in this exact order so truncation can't corrupt the intent:

{"intent": "...", "tickers": [...], "want_panel": true|false, "topic": "...", "rationale": "..."}"""


_MAX_TICKERS = 5
_MAX_RATIONALE_CHARS = 220


# ---------------------------------------------------------------------------
# Conversation-context extraction
# ---------------------------------------------------------------------------
_CLASSIFICATION_INTENT_RE = re.compile(
    r"\*\*Intent\*\*\s*\|\s*`?(?P<intent>\w+)`?", re.IGNORECASE
)
_CLASSIFICATION_TICKERS_RE = re.compile(
    r"\*\*Detected tickers\*\*\s*\|\s*(?P<tickers>[^\n|]+)", re.IGNORECASE
)
_CLASSIFICATION_TOPIC_RE = re.compile(
    r"\*\*Detected topic\*\*\s*\|\s*(?P<topic>[^\n|]+)", re.IGNORECASE
)
_CLASSIFICATION_PANEL_RE = re.compile(
    r"\*\*Panel requested\??\*\*\s*\|\s*(?P<panel>[^\n|]+)", re.IGNORECASE
)


def _parse_previous_card(assistant_text: str) -> Optional[Dict[str, Any]]:
    """Extract the prior classification from the assistant's last reply.

    We emit the classification card in a fixed markdown-table shape, so
    parsing it back is trivial and robust - no need to re-run an LLM on
    arbitrary prose.
    """
    if not assistant_text or "Query Classification" not in assistant_text:
        return None
    # Slice to the classification card section so we don't accidentally
    # match inside the flow content below.
    end = assistant_text.find("\n---", assistant_text.find("Query Classification"))
    section = assistant_text if end == -1 else assistant_text[:end]

    intent_m = _CLASSIFICATION_INTENT_RE.search(section)
    tickers_m = _CLASSIFICATION_TICKERS_RE.search(section)
    topic_m = _CLASSIFICATION_TOPIC_RE.search(section)
    panel_m = _CLASSIFICATION_PANEL_RE.search(section)

    if not intent_m:
        return None

    tickers: List[str] = []
    if tickers_m:
        raw = tickers_m.group("tickers").strip()
        if raw and raw != "—":
            # Tickers rendered as `AAPL`, `WDC` etc.
            for t in re.findall(r"[A-Z][A-Z0-9.]{0,6}", raw):
                if t not in tickers:
                    tickers.append(t)

    want_panel = False
    if panel_m:
        want_panel = panel_m.group("panel").strip().lower().startswith("y")

    topic = topic_m.group("topic").strip() if topic_m else ""
    if topic in {"—", "-", ""}:
        topic = ""

    return {
        "intent": intent_m.group("intent").strip().lower(),
        "tickers": tickers,
        "topic": topic,
        "want_panel": want_panel,
    }


def _last_user_before(history: List[Dict[str, Any]], current_query: str) -> str:
    """Find the previous user turn's content (skipping the current one)."""
    found_current = False
    for msg in reversed(history):
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "user":
            if not found_current and content.strip() == current_query.strip():
                found_current = True
                continue
            return content
    return ""


def _last_assistant(history: List[Dict[str, Any]]) -> str:
    for msg in reversed(history):
        if (msg.get("role") or "").lower() == "assistant":
            return msg.get("content") or ""
    return ""


def _build_context_hint(
    history: Optional[List[Dict[str, Any]]],
    current_query: str,
) -> str:
    """Return a compact "Previous turn" block for the router, or ``""``."""
    if not history:
        return ""
    prior_assistant = _last_assistant(history)
    if not prior_assistant:
        return ""
    parsed = _parse_previous_card(prior_assistant)
    if not parsed:
        return ""
    prior_user = _last_user_before(history, current_query)

    lines = ["", "## Previous turn"]
    if prior_user:
        clean_user = " ".join(prior_user.strip().split())[:160]
        lines.append(f'- Previous user query: "{clean_user}"')
    lines.append(f"- Previous intent: {parsed['intent']}")
    lines.append(
        f"- Previous tickers: {', '.join(parsed['tickers']) if parsed['tickers'] else '(none)'}"
    )
    if parsed["topic"]:
        lines.append(f"- Previous topic: {parsed['topic']}")
    lines.append(f"- Previous want_panel: {str(parsed['want_panel']).lower()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def classify_query(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> RouteDecision:
    """Classify ``query`` (optionally in the context of prior turns).

    ``history`` is the full OpenAI-style messages list from the client,
    including the current user message. Passing it lets the router chain
    follow-ups like "do panel analysis as well" onto the previous turn.
    """
    llm = build_chat_model(
        temperature=0.1,
        max_tokens=800,
        streaming=False,
        response_format={"type": "json_object"},
    )
    context_hint = _build_context_hint(history, query)
    user_content = f'Query: "{query.strip()}"' + context_hint + "\n\nClassify now."
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=user_content),
    ]

    # Retry once on transient connection failures before falling back to
    # the deterministic rules. NVIDIA NIM occasionally drops an individual
    # request; a single retry with a short delay recovers from the vast
    # majority of those. JSON-parse failures and other response-shape
    # errors are NOT retried — those are deterministic issues with the
    # model output, not the transport.
    resp = None
    last_exc: Optional[BaseException] = None
    for attempt in range(2):
        try:
            resp = await llm.ainvoke(messages)
            last_exc = None
            break
        except Exception as e:
            last_exc = e
            if attempt == 0:
                log.warning(
                    "Router LLM call failed on attempt 1/2 (%s) — retrying in 0.5s",
                    e,
                )
                await asyncio.sleep(0.5)
                continue
            log.exception("Router LLM call failed after retry: %s", e)

    if resp is None:
        return _safe_fallback(
            query, history, reason=f"router LLM call failed: {last_exc}"
        )

    content = getattr(resp, "content", "") or ""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        # Best-effort repair: sometimes the model emits trailing prose
        # after the closing brace; try to carve out the first JSON object.
        repaired = _try_repair_json(content)
        if repaired is not None:
            data = repaired
        else:
            log.warning("Router JSON parse failed: %s | content=%r", e, content[:300])
            return _safe_fallback(
                query, history, reason="router returned non-JSON output"
            )

    if not isinstance(data, dict):
        return _safe_fallback(query, history, reason="router returned non-object JSON")

    intent = str(data.get("intent") or "").strip()
    if intent not in INTENTS:
        log.warning("Router returned unknown intent %r, falling back", intent)
        return _safe_fallback(query, history, reason=f"unknown intent: {intent!r}")

    tickers_raw = data.get("tickers") or []
    if not isinstance(tickers_raw, list):
        tickers_raw = []
    tickers: List[str] = []
    for t in tickers_raw[:_MAX_TICKERS]:
        if isinstance(t, (str, int)):
            sym = str(t).strip().upper()
            if sym and sym not in tickers:
                tickers.append(sym)

    topic = str(data.get("topic") or "").strip()
    want_panel = bool(data.get("want_panel", False))
    rationale = str(data.get("rationale") or "").strip() or "No rationale provided."
    if len(rationale) > _MAX_RATIONALE_CHARS:
        rationale = rationale[:_MAX_RATIONALE_CHARS].rstrip() + "…"

    return {
        "intent": intent,
        "tickers": tickers,
        "topic": topic,
        "want_panel": want_panel,
        "rationale": rationale,
    }


def _try_repair_json(raw: str) -> Optional[Dict[str, Any]]:
    """Return the first balanced JSON object found in ``raw``, or ``None``."""
    if not raw:
        return None
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        c = raw[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _safe_fallback(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    *,
    reason: str,
) -> RouteDecision:
    """Deterministic fallback when the router LLM is unavailable.

    Rules, in order:
      1. If the query mentions a panel keyword AND the prior turn had
         tickers → ``stock_research`` with ``want_panel=true``,
         inheriting the tickers.
      2. If the query mentions a panel keyword with NO prior ticker
         context → ``portfolio_analysis`` (the user probably wants the
         full panel on their portfolio).
      3. If the prior intent was portfolio/stock and the current query
         is short/referential → inherit the prior intent + tickers.
      4. Short uppercase-only input (<=5 chars, alpha) → ``stock_research``.
      5. Everything else → ``educational``.

    We deliberately avoid defaulting to ``portfolio_analysis`` in cases
    3-5 because that would reintroduce the "every query runs the panel"
    failure mode this module is designed to prevent.
    """
    q = query.strip()
    panel_re = re.compile(
        r"\b(panel|debate|experts?|opinions?|"
        r"buffett|wood|graham|personas?)\b",
        re.IGNORECASE,
    )
    deep_re = re.compile(
        r"\b(deep\s+dive|deep\s+research|deep\s+analysis|thorough\s+research|"
        r"claim\s+tracking|claims?\s+vs\s+reality|promises?\s+vs\s+reality|"
        r"verify\s+guidance|did\s+they\s+deliver|stood\s+on\s+(?:their|his|her)\s+claims?|"
        r"promised\s+vs\s+delivered|follow\s+through\s+on|"
        r"track\s+record\s+of\s+guidance|earnings\s+history)\b",
        re.IGNORECASE,
    )
    refers_to_prior = re.compile(
        r"\b(it|this|that|also|as\s+well|now|too|same|"
        r"again|continue|on\s+these?|on\s+those?)\b",
        re.IGNORECASE,
    )

    prior: Optional[Dict[str, Any]] = None
    if history:
        prior_text = _last_assistant(history)
        if prior_text:
            prior = _parse_previous_card(prior_text)

    # Rule 0: deep-research keyword -> deep_stock_research if we have tickers
    if deep_re.search(q):
        # Try to detect a bare ticker in the query first (uppercase word, 2-5 chars)
        ticker_candidates = re.findall(r"\b([A-Z]{2,5})\b", q)
        # Strip common English words that happen to be uppercase
        _stop = {"AI", "IT", "OR", "AND", "THE", "CEO", "CFO", "USA", "UK", "FSD", "FY"}
        tickers_in_q = [t for t in ticker_candidates if t not in _stop]
        # Inherit prior tickers if the user didn't repeat them
        if not tickers_in_q and prior and prior.get("tickers"):
            tickers_in_q = list(prior["tickers"])
        if tickers_in_q:
            return {
                "intent": "deep_stock_research",
                "tickers": tickers_in_q[:3],
                "topic": ", ".join(tickers_in_q[:3]),
                "want_panel": False,
                "rationale": (
                    f"Fallback: deep-research keyword + ticker(s) {tickers_in_q[:3]} "
                    f"({reason})."
                ),
            }
        # No tickers found — send to stock_research which will ask the user
        return {
            "intent": "stock_research",
            "tickers": [],
            "topic": "deep research (no ticker)",
            "want_panel": False,
            "rationale": (
                f"Fallback: deep-research keyword without a ticker, "
                f"deferring to stock_research ({reason})."
            ),
        }

    # Rule 1 & 2: panel keyword
    if panel_re.search(q):
        if prior and prior.get("tickers"):
            return {
                "intent": "stock_research",
                "tickers": list(prior["tickers"]),
                "topic": prior.get("topic") or ", ".join(prior["tickers"]),
                "want_panel": True,
                "rationale": (
                    f"Fallback: 'panel' follow-up on previous tickers "
                    f"{prior['tickers']} ({reason})."
                ),
            }
        return {
            "intent": "portfolio_analysis",
            "tickers": [],
            "topic": "portfolio panel (fallback)",
            "want_panel": True,
            "rationale": f"Fallback: 'panel' keyword with no prior tickers ({reason}).",
        }

    # Rule 3: referential follow-up inherits the previous turn
    if prior and prior.get("intent") in INTENTS and refers_to_prior.search(q):
        return {
            "intent": prior["intent"],
            "tickers": list(prior.get("tickers", [])),
            "topic": prior.get("topic") or q[:60],
            "want_panel": bool(prior.get("want_panel", False)),
            "rationale": (
                f"Fallback: referential follow-up inherited prior intent "
                f"{prior['intent']} ({reason})."
            ),
        }

    # Rule 4: bare ticker
    if 1 <= len(q) <= 5 and q.isupper() and q.isalpha():
        return {
            "intent": "stock_research",
            "tickers": [q],
            "topic": q,
            "want_panel": False,
            "rationale": f"Fallback: short uppercase query treated as ticker ({reason}).",
        }

    # Rule 5: educational default
    return {
        "intent": "educational",
        "tickers": [],
        "topic": q[:60],
        "want_panel": False,
        "rationale": f"Fallback to educational flow ({reason}).",
    }


def render_decision_card(decision: RouteDecision, query: str) -> str:
    """Markdown rendering of the routing decision for the chat transcript."""
    intent = decision.get("intent", "unknown")
    label = INTENT_LABELS.get(intent, intent)
    tickers = ", ".join(f"`{t}`" for t in decision.get("tickers", [])) or "—"
    topic = decision.get("topic") or "—"
    panel = "Yes" if decision.get("want_panel") else "No"
    rationale = decision.get("rationale") or "—"
    q = query.strip().rstrip("?.!")

    return (
        "## 🎯 Query Classification\n\n"
        f"**Your query:** _{q}_\n\n"
        "| Field | Value |\n"
        "|---|---|\n"
        f"| **Intent** | `{intent}` ({label}) |\n"
        f"| **Detected tickers** | {tickers} |\n"
        f"| **Detected topic** | {topic} |\n"
        f"| **Panel requested?** | {panel} |\n"
        f"| **Rationale** | {rationale} |\n\n"
        "_A small LLM call picked this flow from four options; a different "
        "query (or follow-up) would have taken a different route through "
        "the agent graph._\n\n"
        "---\n\n"
    )
