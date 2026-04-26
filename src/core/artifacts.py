"""Helpers for emitting LibreChat artifact blocks from FinAI flows.

LibreChat (Claude.ai-style) renders content placed inside a special
"container directive" block in a side panel instead of inline:

::

    :::artifact{identifier="finai-wdc" type="text/markdown" title="WDC — Stock Research"}
    ```
    # Western Digital — Q4 FY26
    ...full markdown report...
    ```
    :::

Anything OUTSIDE the ``:::artifact{...}:::`` block stays inline in the
chat. So a flow can have a conversational opening, an artifact-wrapped
report, and a brief inline summary at the end, all in the same SSE
stream.

Why triple backticks
--------------------
LibreChat's auto-open regex only fires when the artifact body is
wrapped in triple backticks::

    /:::artifact(?:\\{[^}]*\\})?(?:\\s|\\n)*(?:```[\\s\\S]*?```(?:\\s|\\n)*)?:::/m

Without them the directive still parses (via ``remark-directive``)
but the side pane won't auto-open on first render. We always include
the backticks so the panel pops out immediately when an artifact
arrives. The downside is that if our markdown ever contains a triple
backtick code fence we'd break the wrapper - the
:func:`_sanitise_markdown` helper rewrites such fences to four
backticks defensively (markdown renders this identically) before
they reach LibreChat.

How the flows use this
----------------------
Each heavy flow follows this shape::

    yield status("Looking up Western Digital...")
    yield status("Pulling US fundamentals...")
    yield from open_artifact(
        identifier=safe_id("finai-wdc"),
        title="WDC — Stock Research",
    )
    yield chat_text("# Western Digital...")          # this goes INSIDE the artifact
    ...stream the full structured report...
    yield from close_artifact()
    yield chat_text("WDC trades at $404, 38× P/E. Full report in the artifact pane.")

Anything yielded between ``open_artifact`` and ``close_artifact`` is
captured by LibreChat into the side pane. Anything before / after
is inline chat.

The helpers all return :class:`PanelEvent`-shaped dicts so they can
be ``yield``-ed straight from any flow without further wrapping.
"""
from __future__ import annotations

import re
from typing import Iterator

from src.core.panel import PanelEvent


# ---------------------------------------------------------------------------
# Identifier hygiene
# ---------------------------------------------------------------------------
_KEBAB_RE = re.compile(r"[^a-z0-9]+")
_TRIPLE_BACKTICK_RE = re.compile(r"```")


def safe_id(text: str, *, max_len: int = 50, prefix: str = "finai") -> str:
    """Convert ``text`` to a kebab-case identifier safe for the directive.

    LibreChat doesn't enforce hard rules on the identifier, but we want
    deterministic IDs (so a re-run of the same query updates the same
    artifact instead of stacking new ones) and short ones (the
    identifier appears in DOM IDs).

    A leading ``prefix`` is always prepended so artifact IDs are
    namespaced under ``finai-`` and won't collide with whatever
    LibreChat generates internally.
    """
    cleaned = _KEBAB_RE.sub("-", text.lower()).strip("-")
    cleaned = cleaned[:max_len].rstrip("-") or "report"
    return f"{prefix}-{cleaned}"


# ---------------------------------------------------------------------------
# Title hygiene
# ---------------------------------------------------------------------------
def _safe_title(title: str) -> str:
    """Strip characters that would break the directive's quoted attribute.

    Double quotes terminate the title attribute, so we replace them
    with single quotes. Newlines would break the directive parser, so
    we collapse to a single line.
    """
    one_line = " ".join(title.split())
    return one_line.replace('"', "'")


# ---------------------------------------------------------------------------
# Markdown body hygiene
# ---------------------------------------------------------------------------
def _sanitise_markdown(body: str) -> str:
    """Avoid triple-backtick collisions with the artifact's outer fence.

    The artifact wrapper looks like::

        :::artifact{...}
        ```
        body
        ```
        :::

    If ``body`` itself contains ``\\`\\`\\``` it would close the wrapper
    prematurely. Markdown lets us use 4+ backticks for nested fences
    without changing the rendering, so we rewrite any literal triple
    backticks in the body to four. This is a no-op for ordinary
    finance reports (no code blocks); it's a safety net.
    """
    return _TRIPLE_BACKTICK_RE.sub("````", body)


# ---------------------------------------------------------------------------
# PanelEvent emitters
# ---------------------------------------------------------------------------
def status(text: str) -> PanelEvent:
    """Brief italic chat message during a long-running flow.

    Used for "_Looking up WDC..._", "_Pulling fundamentals..._" etc.
    Visible in the chat (not in the artifact pane).
    """
    return {
        "type": "text",
        "text": f"_{text.strip()}_\n\n",
        "persona": "orchestrator",
    }


def chat_text(text: str, *, persona: str = "orchestrator") -> PanelEvent:
    """Plain inline chat text (not italicised).

    Used for the conversational opening / summary line. NOT for the
    artifact body - call :func:`open_artifact` first if you want
    content to land in the side pane.
    """
    return {"type": "text", "text": text, "persona": persona}


def open_artifact(
    *,
    identifier: str,
    title: str,
    content_type: str = "text/markdown",
) -> PanelEvent:
    """Return a single PanelEvent that opens an artifact block.

    Callers ``yield`` this directly from an async generator. After
    yielding it, all subsequent ``text``-typed events emitted by the
    flow are captured into LibreChat's artifact pane until
    :func:`close_artifact` is yielded.

    Returns a single dict (not a generator) because Python's
    ``yield from`` is forbidden inside ``async def`` functions, and
    the flows are all async generators.

    .. note::
       Only ONE artifact per assistant message is supported by
       LibreChat. Don't open a new one before closing the previous.
    """
    safe_title_v = _safe_title(title)
    return {
        "type": "text",
        "text": (
            f':::artifact{{identifier="{identifier}" '
            f'type="{content_type}" '
            f'title="{safe_title_v}"}}\n'
            "```\n"
        ),
        "persona": "orchestrator",
    }


def artifact_body(text: str) -> PanelEvent:
    """Yield a chunk of artifact body content (with backtick safety).

    Use this for any content that should land inside the open
    artifact - tables, prose, bullets, headers. Triple backticks in
    the body are rewritten to four to avoid breaking the wrapper.

    For streaming LLM output, you can either pre-sanitise once and
    yield the chunks raw, or run each chunk through this helper.
    """
    return {
        "type": "text",
        "text": _sanitise_markdown(text),
        "persona": "orchestrator",
    }


def close_artifact() -> PanelEvent:
    """Return a single PanelEvent that closes the artifact block.

    Always pair with a prior :func:`open_artifact`. After yielding
    this, subsequent text events return to inline chat.
    """
    return {
        "type": "text",
        "text": "\n```\n:::\n",
        "persona": "orchestrator",
    }


__all__ = [
    "safe_id",
    "status",
    "chat_text",
    "open_artifact",
    "artifact_body",
    "close_artifact",
]
