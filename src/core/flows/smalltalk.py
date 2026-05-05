"""Smalltalk flow — short conversational replies for greetings/chitchat.

Triggered when the user sends a casual message that doesn't need any
agent / tool work: "hi", "thanks", "ok", "got it", "good morning". The
goal is to feel like a chat with a normal assistant — no headers, no
classification card, no disclaimer, no agent fanfare. Just a short,
friendly reply that nudges them toward something useful if they've not
asked anything yet.

Hybrid implementation
---------------------
1. **Static-reply fast path** for the most common, unambiguous phrasings
   (``hi``, ``hello``, ``thanks``, ``ok``, ``bye``). Matched by regex,
   replied to with a hand-written one-liner. Zero LLM cost, zero
   latency. We deliberately keep these responses **slightly varied**
   per-category but **not random**: greeting → invite, thanks →
   acknowledge + offer next step, etc.

2. **LLM fallback** for anything else routed here by the classifier
   (e.g. a more nuanced greeting like "morning! how are things?"). One
   small LLM call with a brief, friendly system prompt — capped at
   ``max_tokens=200`` so it stays a one-or-two-line reply.

Why this shape
--------------
The user feedback that prompted this flow: the previous behaviour
classified "hi" as ``educational``, which then ran the financial-educator
prompt and produced a weird structured definition response. The fix is to
stop pretending every query is a finance question - some are just
greetings.

Importantly, this flow emits **NO** disclaimer, **NO** classification
card, and **NO** persona-style flourish. The dispatcher already gates
those off when ``verbose_trace`` is False (which is the default). For
``smalltalk`` specifically, the dispatcher also skips the disclaimer
because a "hi" doesn't need a SEBI warning.
"""
from __future__ import annotations

import logging
import re
from typing import AsyncIterator, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.personas.base import build_chat_model
from src.core.panel import PanelEvent
from src.core.router import RouteDecision


log = logging.getLogger("finai.flows.smalltalk")


# ---------------------------------------------------------------------------
# Static-reply fast path
# ---------------------------------------------------------------------------
# Each tuple is (regex pattern, reply category). Patterns are compiled with
# IGNORECASE. We anchor to start-of-string and allow trailing punctuation /
# emoji-ish characters because real users say "hi!" / "thanks :)" / "ok.".
#
# Order matters when patterns overlap - first match wins. Keep specific
# patterns above general ones.
_STATIC_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # Time-of-day greetings (specific, before generic "hi")
    (re.compile(
        r"^\s*(?:good\s+)?(?:morning|afternoon|evening|night)\s*[!.,?]*\s*$",
        re.IGNORECASE,
    ), "greeting_time"),
    # Generic greetings - bare or with light punctuation, NO trailing words
    (re.compile(
        r"^\s*(?:hi|hello|hey|hola|yo|sup|hi\s+there|hey\s+there|hello\s+there)\s*[!.,?]*\s*$",
        re.IGNORECASE,
    ), "greeting"),
    # Thanks
    (re.compile(
        r"^\s*(?:thanks(?:\s+a\s+lot)?|thank\s+you|thx|ty|cheers|appreciate\s+(?:it|that|you))\s*[!.,?]*\s*$",
        re.IGNORECASE,
    ), "thanks"),
    # Acknowledgments
    (re.compile(
        r"^\s*(?:ok|okay|got\s+it|alright|cool|nice|sounds?\s+good|"
        r"makes?\s+sense|sure|right|yes|yeah|yep|yup)\s*[!.,?]*\s*$",
        re.IGNORECASE,
    ), "acknowledge"),
    # Goodbyes
    (re.compile(
        r"^\s*(?:bye|goodbye|see\s+you|see\s+ya|cya|later|talk\s+soon|"
        r"good\s+night|gn|night)\s*[!.,?]*\s*$",
        re.IGNORECASE,
    ), "goodbye"),
)


# Static replies per category. Hand-written so they sound natural and
# always nudge the user toward something useful (instead of leaving the
# conversation stuck at "hi"). We DON'T randomise these per-call
# because consistency is nicer than novelty for a financial assistant.
_STATIC_REPLIES: dict[str, str] = {
    "greeting": (
        "Hey 👋 — what would you like to look at? I can pull a stock "
        "(US or Indian), analyse your portfolio, run a Buffett / Wood / "
        "Graham panel debate, dig through SEC or BSE filings, or explain "
        "a finance concept."
    ),
    "greeting_time": (
        "Good to see you 👋 — I can look at a stock, your portfolio, "
        "run a panel debate, or explain a concept. What's on your mind?"
    ),
    "thanks": (
        "Anytime — let me know what you'd like to look at next."
    ),
    "acknowledge": (
        "Got it. Want me to look at a stock, your portfolio, or a concept?"
    ),
    "goodbye": (
        "See you next time. Bookmark me for the next time you're "
        "researching a stock or thinking about your portfolio."
    ),
}


def _match_static(query: str) -> Optional[str]:
    """Return the static reply category if ``query`` matches a fast-path pattern."""
    for pattern, category in _STATIC_PATTERNS:
        if pattern.search(query):
            return category
    return None


# ---------------------------------------------------------------------------
# LLM fallback for nuanced casual messages
# ---------------------------------------------------------------------------
# Triggered when the classifier routed a query here but our regex didn't
# match. Tight system prompt + small max_tokens => fast, predictable,
# never hallucinates capabilities (the meta_help flow handles that).
_LLM_SYSTEM = (
    "You are FinAI, a multi-agent financial-analysis assistant. The user "
    "just sent a CASUAL / CONVERSATIONAL message — a greeting, a thanks, "
    "an acknowledgment, or some chitchat. They have NOT asked you to "
    "analyse anything yet.\n\n"
    "Reply briefly (1-2 sentences) in a warm, natural tone. End with a "
    "short, optional nudge toward something useful: e.g. 'want me to "
    "look at a stock or run a panel?'.\n\n"
    "Strict rules:\n"
    "- DO NOT use markdown headers (no #, ##), tables, or bullet lists.\n"
    "- DO NOT describe your full capabilities — that's a different flow.\n"
    "- DO NOT add a disclaimer.\n"
    "- DO NOT pretend to be a generic AI assistant — you are FinAI "
    "specifically.\n"
    "- Keep it under 40 words."
)


async def run(
    query: str,
    decision: Optional[RouteDecision] = None,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Emit a short conversational reply.

    Fast path: regex match → curated static reply (no LLM).
    Fallback: tiny LLM call with the smalltalk system prompt.
    """
    log.debug("smalltalk flow invoked for query=%r", query[:80])

    # 1) Try the static fast-path first.
    category = _match_static(query)
    if category is not None:
        yield {
            "type": "text",
            "text": _STATIC_REPLIES[category],
            "persona": "orchestrator",
        }
        return

    # 2) LLM fallback - kept tight (max_tokens=200) so it can't drift
    #    into a long-form response. Streamed so the user sees the
    #    text appear naturally instead of waiting for the whole reply.
    llm = build_chat_model(
        temperature=0.5,
        max_tokens=200,
        streaming=True,
    )
    messages = [
        SystemMessage(content=_LLM_SYSTEM),
        HumanMessage(content=query.strip() or "(empty)"),
    ]
    try:
        async for chunk in llm.astream(messages):
            text = getattr(chunk, "content", None)
            if text:
                yield {"type": "text", "text": text, "persona": "orchestrator"}
    except Exception as e:
        log.exception("Smalltalk LLM fallback failed")
        # Last-ditch static reply so the user doesn't see a raw error
        # for a "hi". Failing visibly on a greeting is much worse than
        # a slightly-generic recovery.
        yield {
            "type": "text",
            "text": (
                "Hey — I'm here. Want me to look at a stock, your "
                "portfolio, or a concept?"
            ),
            "persona": "orchestrator",
        }
