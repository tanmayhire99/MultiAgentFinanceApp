"""Topic Research flow - open-ended web research on a theme or macro question.

Triggered by the router when the user asks *about* a topic that isn't
tied to a specific ticker (e.g. "Impact of US-China tariffs on semis",
"Trends in Indian IT services"). The flow:

1. Emit a focused "Topic Research" header.
2. Hand off to the **Research Agent** (``research__search_web``). One
   visible MCP handoff so the audience sees the routing.
3. Render the top search results as clickable Markdown links.
4. If the backend provided a synthesised answer (Tavily only), show it.
5. Ask GPT-OSS-120B for a short LLM summary grounded in the results.

No portfolio data, no stock agents, no investor panel. Run time is
typically 5-12 s end-to-end: ~2 s for search, ~3 s for the LLM summary.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.personas.base import build_chat_model
from src.core.panel import PanelEvent, _call_tool
from src.core.router import RouteDecision


log = logging.getLogger("finai.flows.topic")


_SYSTEM = (
    "You are a concise financial research analyst. You have been given the "
    "top web-search results for a user's topic question. Write a short "
    "briefing following this structure exactly:\n\n"
    "**Summary** — 2-3 sentences of what these results collectively say.\n"
    "**Key points** — 3-5 bullets citing the most important findings. "
    "When a point comes from a specific source, reference it by linked "
    "title (markdown: [title](url)). Do NOT invent sources; only cite "
    "links that appear in the supplied results.\n"
    "**What to watch next** — 1-3 bullets of forward-looking indicators "
    "or signals the user should monitor.\n\n"
    "Rules:\n"
    "- Aim for 160-260 words total.\n"
    "- Be neutral and factual. No hype, no investment advice.\n"
    "- If the supplied results are thin or contradictory, say so "
    "honestly rather than padding.\n"
    "- Do NOT end with a disclaimer; one is appended separately."
)


def _build_topic_query(user_query: str, decision: Optional[RouteDecision]) -> str:
    """Shape the user's query into a high-recall search string."""
    topic = ""
    if decision:
        topic = (decision.get("topic") or "").strip()
    if topic and topic.lower() not in user_query.lower():
        return f"{user_query.strip()} ({topic})"
    return user_query.strip()


def _format_results_for_llm(results: List[Dict[str, Any]], answer: Optional[str]) -> str:
    lines: List[str] = []
    if answer:
        lines.append(f"Synthesised answer from the backend:\n{answer}\n")
    lines.append("Top results:")
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip() or "(no title)"
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            # Keep snippets compact so the LLM context doesn't blow up
            lines.append(f"   Snippet: {snippet[:500]}")
        lines.append("")
    return "\n".join(lines)


def _render_results_md(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "_No results returned by the search backend._\n"
    rows = ["#### Top Web Results\n"]
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip() or "(untitled)"
        url = (r.get("url") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        if url:
            heading = f"{i}. **[{title}]({url})**"
        else:
            heading = f"{i}. **{title}**"
        rows.append(heading)
        if snippet:
            short = snippet[:220]
            if len(snippet) > 220:
                short += "…"
            rows.append(f"   {short}")
        rows.append("")
    return "\n".join(rows)


async def run(
    query: str,
    decision: Optional[RouteDecision] = None,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Run a single Research Agent web search + LLM summary."""
    topic_label = ""
    if decision:
        topic_label = (decision.get("topic") or "").strip()

    heading = "# 🔎 Topic Research"
    if topic_label:
        heading += f": {topic_label}"
    yield {"type": "header", "text": f"{heading}\n\n"}

    yield {
        "type": "text",
        "text": (
            f"**Your query:** _{query.strip().rstrip('?.!')}_\n\n"
            "_This query was classified as an open-ended topic question, "
            "so only the **Research Agent** is being consulted. No "
            "portfolio, stock, or panel agents are invoked._\n\n"
        ),
        "persona": "orchestrator",
    }

    search_query = _build_topic_query(query, decision)

    # Single visible Research Agent handoff
    yield {
        "type": "tool_call",
        "persona": "orchestrator",
        "persona_label": "Orchestrator",
        "tool": "research__search_web",
        "args": {"query": search_query, "max_items": 6},
    }
    try:
        resp = await _call_tool(
            "research__search_web", {"query": search_query, "max_items": 6}
        )
    except Exception as e:
        log.exception("Topic search_web failed")
        yield {"type": "error", "text": f"Research Agent search failed: {e}"}
        return

    if not isinstance(resp, dict):
        yield {"type": "error", "text": "Research Agent returned an unexpected payload."}
        return

    results = resp.get("results") or []
    answer = resp.get("answer") or ""
    backend = resp.get("backend") or (resp.get("_source") or "").replace("live:", "")

    yield {
        "type": "tool_result",
        "persona": "orchestrator",
        "tool": "research__search_web",
        "result_preview": (
            f"{len(results)} result{'s' if len(results) != 1 else ''} "
            f"via {backend or 'unknown backend'}"
        ),
    }

    # Render the raw results so the audience can see what the LLM will
    # be grounded in.
    if answer:
        yield {
            "type": "text",
            "text": (
                "\n#### Backend Synthesis\n\n"
                f"_Via {backend or 'live backend'}_\n\n"
                f"> {answer}\n\n"
            ),
            "persona": "orchestrator",
        }

    yield {
        "type": "text",
        "text": "\n" + _render_results_md(results) + "\n",
        "persona": "orchestrator",
    }

    # LLM summary grounded in the results
    if not results and not answer:
        yield {
            "type": "text",
            "text": (
                "_The Research Agent returned no usable results, so no "
                "summary is being generated._\n\n"
            ),
            "persona": "orchestrator",
        }
        return

    yield {"type": "header", "text": "\n### Analyst Briefing\n\n"}
    yield {
        "type": "text",
        "text": "_Synthesising the top results into a briefing…_\n\n",
        "persona": "moderator",
    }

    llm = build_chat_model(temperature=0.2, max_tokens=1400, streaming=True)
    summary_input = _format_results_for_llm(results, answer)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"User's question: {query.strip()}\n\n"
                f"{summary_input}\n\n"
                "Write the briefing now, following the system prompt structure."
            )
        ),
    ]
    try:
        async for chunk in llm.astream(messages):
            text = getattr(chunk, "content", None)
            if text:
                yield {"type": "text", "text": text, "persona": "moderator"}
    except Exception as e:
        log.exception("Topic LLM summary failed")
        yield {"type": "error", "text": f"Summary LLM call failed: {e}"}
        return

    yield {"type": "text", "text": "\n\n", "persona": "moderator"}
