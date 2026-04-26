"""Deep Stock Research flow - 2-5 minute multi-step agent with claim tracking.

Picked by the dispatcher when the router classifies a query as
``deep_stock_research``. Unlike ``stock_research`` (fast single-pass
ReAct over current metrics), this flow is a **batch-mode deep agent**
orchestrated by LangChain's :mod:`deepagents`. It pulls SEC filings,
historical news, extracts forward-looking claims from past management
commentary, fetches the latest actuals, and produces a verdict report
on whether the company delivered on its past promises.

Pipeline (executed autonomously by the agent, not scripted)::

    1. Scope  - agent reads the query, plans via ``write_todos``
    2. Gather - agent fetches SEC filings (10-K / 10-Q / 8-K),
                historical news (Tavily date-range), current metrics
                (yfinance), stashes raw content to the virtual
                filesystem
    3. Extract - agent runs ``extract_forward_claims`` on historical
                 documents, saves the claim JSON as a file
    4. Diff   - agent runs ``compare_claim_to_reality`` on each claim
                vs the latest evidence, saves verdicts
    5. Write  - agent writes the final report in one shot, quoting
                the snippets it captured during steps 2-4

Design notes
------------
* Uses ``deepagents.create_deep_agent`` as the harness. That gives us
  the ``write_todos`` planning tool, the virtual filesystem (read /
  write / edit / ls / grep), and sub-agent spawning for free.
* Works with the NIM ``gpt-oss-120b`` model the rest of the system
  uses. Planning quality on gpt-oss-120b is lower than on gpt-4.1, but
  for a demo it's acceptable; users can swap in a frontier model by
  overriding ``FINAI_DEEP_RESEARCH_MODEL`` in the environment.
* Streams events via LangGraph's :func:`astream_events` (v2) and
  translates them into :class:`PanelEvent` dicts so the existing SSE
  renderer and LibreChat UI work without changes.
* Long runs are unavoidable - most of the latency is real tool calls
  (SEC EDGAR + LLM extraction + LLM comparison). The orchestrator
  banner promises 2-5 minutes upfront so the user isn't surprised.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from src.config import mcp_servers
from src.core.panel import PanelEvent
from src.core.router import RouteDecision


log = logging.getLogger("finai.flows.deep_stock_research")


# Wall-clock cap on a single deep-research run. The user agreed to a
# 3-5 minute batch mode, but the agent will sometimes wander into
# 10+ minute territory (seen in testing: 96 tool calls over 11 min on
# TSLA). At this cap we stop accepting new agent events and force the
# fallback "final assistant message" path so the user gets something
# useful rather than waiting indefinitely. Override via the
# ``FINAI_DEEP_RESEARCH_MAX_SECONDS`` environment variable.
_DEFAULT_MAX_SECONDS = 420.0  # 7 minutes - slightly over the promised budget


def _max_seconds() -> float:
    raw = os.environ.get("FINAI_DEEP_RESEARCH_MAX_SECONDS", "")
    try:
        v = float(raw)
        return v if v > 10 else _DEFAULT_MAX_SECONDS
    except (TypeError, ValueError):
        return _DEFAULT_MAX_SECONDS


# ---------------------------------------------------------------------------
# System prompt - the "detailed prompt" that turns a shallow ReAct agent
# into a Deep Agent per Harrison Chase's description.
# ---------------------------------------------------------------------------
_DEEP_RESEARCH_SYSTEM = """You are FinAI's **Deep Stock Research Agent**.

Your job is to produce a grounded, claim-tracking research report on a public company. This is a BATCH investigation, not a chat - the user has already agreed to wait 2-5 minutes for a thorough answer. Work slowly and carefully; quote specific numbers and sources.

## Output contract

Your FINAL assistant message must be a markdown report in this exact structure:

```
# 🔬 Deep Research: <TICKER> — <Company Name>

## TL;DR
2-4 sentences summarising the company's core business and your overall verdict on whether management delivers on its commitments.

## Current Snapshot
Key metrics (price, P/E, margins) and what the company actually sells. Use fundamentals from the yfinance / us_stock / indian_stock tools.

## Past Commitments Scorecard
A markdown table with columns: **Claim | Target | Verdict | Variance | Source**.
Each row is one forward-looking claim management made in a past earnings call, filing, or press release, compared against what actually happened.
Use the ⬛ verdict icons: 🟢 met · 🟡 partial · 🔴 missed · ⚪ pending · ❓ unknowable.
Aim for 4-8 rows. Cite the source filing/call for each.

## Case Study: <one bold claim>
A 2-paragraph deep dive on the single most interesting claim-vs-reality finding (either the biggest "they did deliver" or the biggest "they missed"). Quote the original claim text and the actuals evidence.

## Recent Catalysts (last 90 days)
3-5 bullets with dates, grounded in the most recent news you pulled.

## Bull / Bear
Two short lists (3 bullets each) of the strongest bull and bear arguments, grounded in the claims scorecard and the catalysts above.

## Sources
A numbered list of every URL (filings, news articles) you actually fetched.
```

Do NOT write a buy/sell recommendation. Do NOT give a price target.

## How to gather data

You have a virtual filesystem (``write_file``, ``read_file``, ``ls``, ``grep``) AND a todo list (``write_todos``). Use them. A reasonable plan:

1. Call ``write_todos`` to lay out the 5-7 step plan.
2. Call ``get_sec_filings`` (if the ticker is US-registered) to get the list of recent 10-K / 10-Q / 8-K filings. Grab the URLs.
3. Call ``fetch_sec_document`` on 1-2 of those URLs and ``write_file`` the cleaned text to e.g. ``filings/10q_latest.txt``. Keep ``max_chars`` reasonable (30-40k per fetch).
4. Call ``search_historical_news`` with a window of 6-18 months ago (``start_date`` / ``end_date`` in ``YYYY-MM-DD``). Save the top results to ``news/historical.json``.
5. Call ``search_news`` (no date filter) for the most recent 30-90 days. Save to ``news/recent.json``.
6. Call ``get_fundamentals`` / ``get_quote`` for current metrics via the us_stock__* or indian_stock__* namespaced tools.
7. For each historical document (either a filing excerpt or a news article body), call ``extract_forward_claims`` to pull structured claims.
8. For each extracted claim, build an "actuals context" (concatenate the most relevant recent news snippets + current fundamentals) and call ``compare_claim_to_reality``.
9. Save every verdict to ``verdicts/NN_claim.json`` so you can cite them in the final report.
10. Write the final markdown report per the output contract above.

## Which tool to use for which job

Tool names are prefixed with the MCP worker that owns them.

**Stock metrics** (pass a TICKER SYMBOL):
- ``us_stock__get_quote(ticker="NVDA")`` — price, 52w range, market cap
- ``us_stock__get_fundamentals(ticker="NVDA")`` — P/E, margins, ROE, revenue, etc.
- ``us_stock__get_defensive_metrics(ticker="NVDA")`` — Graham number, current ratio, dividend yield
- ``indian_stock__get_fundamentals(ticker="TCS")`` — same tools for NSE / BSE tickers (TCS, INFY, RELIANCE, HDFCBANK, ITC)

**News and research** (watch the parameter names carefully):
- ``research__search_news(ticker="NVDA", max_items=5)`` — recent news for a TICKER. Do NOT pass ``query=...``; use search_web for free text.
- ``research__search_historical_news(ticker="NVDA", start_date="2024-01-01", end_date="2024-06-30", max_items=8)`` — news in a specific date window (claim tracking)
- ``research__search_web(query="AI chip export controls 2025", max_items=5)`` — FREE-TEXT search. Use QUERY, NOT ticker.
- ``research__get_company_brief(ticker="NVDA")`` — one-paragraph company overview

**US SEC filings** (only for US-registered issuers; returns empty for Indian-only tickers):
- ``research__get_sec_filings(ticker="NVDA", form_types=["10-K", "10-Q", "8-K"], limit=5)`` — list recent filings
- ``research__fetch_sec_document(url="<report_url from get_sec_filings>", max_chars=40000, offset=0)`` — download + clean a filing document. ``offset`` lets you page through long documents without re-fetching.

**Indian NSE/BSE filings** (primary path for TCS, INFY, RELIANCE, HDFCBANK, ITC, WIPRO, HCLTECH, ICICIBANK, SBIN, …):

India has NO 10-K / 10-Q. Instead every listed company files under SEBI LODR:
- Quarterly Results (bare numbers, 3-5 pages)
- Annual Report (the REAL long-form MD&A equivalent, 200-400 pages)
- Earnings Call Transcripts (25-50 pages, published within 2 days of the call — the RICHEST claim-tracking source)
- Investor Presentations (20-40 slides with guidance tables)
- Continuous material-event disclosures (Reg 30)

Tools:
- ``research__get_screener_snapshot(ticker="TCS")`` — one-shot: name, BSE code, 9 ratios (P/E, ROCE, ROE, Dividend Yield, …), machine-generated Pros/Cons, direct URLs to Annual Reports + concall transcripts. **Call this FIRST** for any Indian ticker; it's the cheapest way to orient yourself.
- ``research__get_indian_concall_urls(ticker="TCS", limit=6)`` — shortcut to just the concall transcript URLs (BSE + Screener merged). Best claim-tracking starting point.
- ``research__get_indian_annual_reports(ticker="TCS", limit=3)`` — shortcut to just the Annual Report PDF URLs.
- ``research__get_indian_filings(ticker="TCS", source="bse", start_date="2024-01-01", end_date="2024-12-31", limit=20, categories=["transcript", "press release"])`` — full BSE announcement stream with date filter + category filter. Good when you want press releases or board-meeting outcomes beyond just transcripts.
- ``research__fetch_indian_document(url="<pdf_url>", max_chars=40000, offset=0)`` — download + parse a BSE or IR-page PDF (pdfplumber; layout-aware). Supports paging for 300-page Annual Reports.

Recommended Indian-ticker playbook:

1. ``get_screener_snapshot(ticker)`` → gives ratios + pros/cons + document URLs.
2. ``get_indian_concall_urls(ticker, limit=4)`` → most recent 4 quarters' transcripts.
3. For each transcript URL: ``fetch_indian_document(url)`` then ``extract_forward_claims`` on the text → save claims to ``claims/Qx.json``.
4. ``get_indian_filings(ticker, start_date="<3mo ago>", categories=["press release"])`` → pulls the most recent earnings press releases as "actuals".
5. For each claim: ``compare_claim_to_reality(...)`` using the recent press release text + current Screener ratios as the actuals context.
6. Write the final markdown report using the same output contract below.

**Claim tracking** (works for US and Indian tickers alike):
- ``research__extract_forward_claims(text="<document text>", source_label="TCS Q4 FY26 concall", max_claims=10)``
- ``research__compare_claim_to_reality(claim_text="...", claim_metric="FY26 revenue growth (CC)", claim_target_value="3-5%", claim_target_date="FY26", actuals_context="<recent press release + metrics>")``

Do NOT invent tool names. Do NOT pass ``query=`` to ``search_news`` (it wants ``ticker=``). For Indian tickers, SKIP ``get_sec_filings`` (it will return empty) and go straight to ``get_screener_snapshot``.

## Handling failures

* If ``get_sec_filings`` returns an empty list (non-US ticker), the ticker is Indian. Switch to the ``get_screener_snapshot`` / ``get_indian_concall_urls`` / ``get_indian_filings`` path instead.
* If ``get_screener_snapshot`` fails (Screener site change or a very small-cap company), fall back to ``get_indian_filings(source="bse")`` directly using the ticker's BSE scrip code lookup done by the tool.
* If ``fetch_indian_document`` fails on a specific PDF URL, try a different concall (older quarter) — some IR hosts briefly 404.
* If ``search_historical_news`` returns few results, widen the window OR fall back to news-less claim extraction on company press releases you find via ``search_web``.
* If ``extract_forward_claims`` returns ``{"claims": []}`` on a document, it means the document has no testable forward-looking language. That's a legitimate outcome - don't keep retrying the same text. Move to the next source.
* If ``compare_claim_to_reality`` returns ``verdict: "unknowable"`` you haven't fetched enough evidence. Try pulling more recent news snippets for the claim's specific topic (e.g. for an FSD claim, search "Tesla FSD autonomy level latest").
* If a specific tool call times out or errors, note it in your ``todos.md`` and continue. An incomplete report is better than no report.

## Style

* ALWAYS quote the original claim text and the evidence snippet. Your credibility comes from showing your work.
* Use full dates (YYYY-MM-DD) where possible.
* Use the ``thinking`` chunks sparingly - save verbose reasoning for the filesystem, not the final response.
"""


# ---------------------------------------------------------------------------
# Flow entry point
# ---------------------------------------------------------------------------
async def run(
    query: str,
    decision: Optional[RouteDecision] = None,
    user_id: str = "demo",
) -> AsyncIterator[PanelEvent]:
    """Stream the Deep Stock Research agent's work back to the user.

    Fix 3 layout: a brief italic chat status line, then the deep
    agent's planning + tool calls + final report stream directly to
    the chat (inline mode) or are wrapped in a ``:::artifact{}:::``
    block for LibreChat's side pane (artifact mode, set when the
    user used ``/report`` / ``/artifact`` or asked for a report
    in natural language).

    The previous H1 banner ``# 🔬 Deep Stock Research`` and the
    multi-paragraph "_This is the batch-mode deep-research flow.
    I'll autonomously plan a multi-step investigation..._" tour-guide
    have been removed - they were dev-style narration that told the
    user what was about to happen instead of just doing it.
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
    tickers = [t for t in tickers if t][:3]
    wants_artifact = bool(decision.get("wants_artifact"))

    # --- guardrail: no ticker → short inline help message -----------------
    if not tickers:
        yield chat_text(
            "Deep research needs at least one ticker. Try one of:\n\n"
            "- `Do a deep dive on NVDA`\n"
            "- `Claim tracking for Tesla`\n"
            "- `Did Microsoft deliver on its Azure AI guidance?`\n"
        )
        return

    label = ", ".join(tickers)
    started = time.time()
    max_secs_budget = _max_seconds()

    # --- one chat status line; rest streams to chat or artifact ------------
    yield status(
        f"Deep research on {label} — claim tracking + SEC + historical news "
        f"(ETA up to {int(max_secs_budget // 60)} min)..."
    )

    # Lazy artifact open. In artifact mode the wrapper opens on the
    # first content event from the deep agent. In inline mode it
    # never opens; everything flows to chat.
    artifact_title = f"Deep Research: {label}"
    artifact_id = safe_id(f"deep-research-{label}")
    artifact_open = False

    def _ensure_artifact_open() -> List[PanelEvent]:
        nonlocal artifact_open
        if artifact_open or not wants_artifact:
            return []
        artifact_open = True
        return [
            open_artifact(identifier=artifact_id, title=artifact_title),
            artifact_body(f"# {artifact_title}\n\n"),
        ]

    def _route(ev: PanelEvent) -> List[PanelEvent]:
        """Route an event - lazy-opens the artifact in artifact mode."""
        return _ensure_artifact_open() + [ev]

    # --- build the deep agent -----------------------------------------------
    try:
        from deepagents import create_deep_agent
    except ImportError as e:
        yield {
            "type": "error",
            "text": (
                "The `deepagents` package is not installed. "
                "Run `pip install deepagents>=0.5,<1` and rebuild the container. "
                f"(Import error: {e})"
            ),
        }
        return

    # Pull the full MCP tool set. The dispatcher has already called
    # get_tools() + install_tool_cache_wrappers + register_tools by now,
    # so this returns the shared cached list instantly.
    try:
        tools = await mcp_servers.get_tools()
    except Exception as e:
        yield {"type": "error", "text": f"Failed to load MCP tools: {e}"}
        return

    from src.agents.personas.base import build_chat_model

    # We use the NIM model by default; override with FINAI_DEEP_RESEARCH_MODEL
    # to point at a frontier model (e.g. "openai:gpt-4.1") for better planning
    # quality. The env variable takes precedence over build_chat_model's default.
    override_model = os.environ.get("FINAI_DEEP_RESEARCH_MODEL")
    if override_model:
        from langchain.chat_models import init_chat_model

        try:
            model = init_chat_model(override_model)
        except Exception as e:
            log.warning(
                "FINAI_DEEP_RESEARCH_MODEL=%r failed to init (%s); "
                "falling back to NIM default.",
                override_model,
                e,
            )
            model = build_chat_model(
                temperature=0.1, max_tokens=3000, streaming=True
            )
    else:
        model = build_chat_model(temperature=0.1, max_tokens=3000, streaming=True)

    try:
        agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=_DEEP_RESEARCH_SYSTEM,
        )
    except Exception as e:
        log.exception("Failed to create deep agent")
        yield {
            "type": "error",
            "text": f"Failed to create deep-research agent: {e}",
        }
        return

    # --- stream events -------------------------------------------------------
    user_turn = (
        f"Ticker(s) to research: {label}\n"
        f"Original user question: {query.strip()}\n\n"
        "Begin the investigation now. Follow the plan in the system prompt."
    )
    initial_state = {"messages": [{"role": "user", "content": user_turn}]}

    tool_call_count = 0
    final_text_parts: List[str] = []
    final_emitted = False

    # Bump LangGraph's default recursion_limit (25) - a full claim-tracking
    # run easily needs 60+ graph steps (plan + 6-10 tool calls + 6-10
    # intermediate LLM turns + final report). 500 matches the persona
    # recursion limit set in src.agents.personas.base.PERSONA_RECURSION_LIMIT.
    agent_config = {"recursion_limit": 500}

    max_secs = _max_seconds()
    deadline = started + max_secs
    timed_out = False

    try:
        # Debug counters - log event-kind distribution so we can tell at
        # a glance whether the agent is actually calling tools or just
        # looping in the planner.
        event_kinds: Dict[str, int] = {}
        async for event in agent.astream_events(
            initial_state, version="v2", config=agent_config
        ):
            # Wall-clock deadline: break the iteration and let the fallback
            # path emit whatever final state the agent has accumulated.
            if time.time() > deadline:
                timed_out = True
                log.warning(
                    "Deep-research run exceeded max_seconds=%.0f; stopping stream early",
                    max_secs,
                )
                for piece in _route({
                    "type": "text",
                    "text": (
                        f"\n\n> ⏱️ _Wall-clock budget of "
                        f"{max_secs:.0f}s reached; stopping the agent and "
                        f"assembling a report from what's been gathered so far._\n\n"
                    ),
                    "persona": "orchestrator",
                }):
                    yield piece
                break

            kind = event.get("event")
            event_kinds[kind] = event_kinds.get(kind, 0) + 1

            if kind == "on_tool_start":
                tool_call_count += 1
                tool_name = event.get("name") or "?"
                args = (event.get("data") or {}).get("input") or {}
                # Strip noisy / huge arg values (e.g. a 40k-char document text)
                # from the display - users don't need to see the full payload.
                args_for_display = _sanitize_tool_args(args)
                # Routed through _route so it lands in the right pane in
                # artifact mode. The dispatcher filters tool_call events
                # entirely when verbose_trace is off, so this only shows
                # for /trace runs in either mode.
                for piece in _route({
                    "type": "tool_call",
                    "persona": "deep_agent",
                    "persona_label": "Deep Research Agent",
                    "tool": tool_name,
                    "args": args_for_display,
                }):
                    yield piece

            elif kind == "on_tool_end":
                tool_name = event.get("name") or "?"
                output = (event.get("data") or {}).get("output")
                summary = _summarize_tool_output(tool_name, output)
                if summary:
                    # Emit as a `tool_result` event (not a free-form text
                    # event) so it's gated by the dispatcher's verbose_trace
                    # filter, matching the tool_call above.
                    for piece in _route({
                        "type": "tool_result",
                        "persona": "deep_agent",
                        "tool": tool_name,
                        "result_preview": summary,
                    }):
                        yield piece

            elif kind == "on_chat_model_stream":
                # Token-level stream for the agent's chat output. We only
                # surface tokens from the top-level run (not sub-agent
                # scratchpad reasoning) so the user sees ONE coherent
                # final answer rather than overlapping streams.
                metadata = event.get("metadata") or {}
                run_tags = metadata.get("tags") or []
                # The final report streams under the "langgraph_node" = "agent"
                # tag in deepagents' latest versions. Filter loosely to avoid
                # drowning the user in sub-agent tokens.
                if "agent" in run_tags or "supervisor" in run_tags or not run_tags:
                    chunk = (event.get("data") or {}).get("chunk")
                    content = getattr(chunk, "content", None) if chunk else None
                    if isinstance(content, str) and content:
                        if not final_emitted:
                            # Section break before the final report. The
                            # old "## 📄 Final Report" banner was Fix 3'd
                            # away because the streamed content speaks
                            # for itself.
                            for piece in _route({
                                "type": "text",
                                "text": "\n\n---\n\n",
                                "persona": "orchestrator",
                            }):
                                yield piece
                            final_emitted = True
                        final_text_parts.append(content)
                        for piece in _route({
                            "type": "text",
                            "text": content,
                            "persona": "deep_agent",
                        }):
                            yield piece

    except Exception as e:
        log.exception("Deep research agent crashed")
        yield {
            "type": "error",
            "text": f"Deep-research agent failed mid-run: {e}",
        }
        return

    # If streaming didn't surface a final report via on_chat_model_stream
    # (some model integrations skip it), fall back to the agent's final
    # assistant message in the returned state.
    if not final_emitted:
        try:
            final_state = await agent.ainvoke(initial_state, config=agent_config)
        except Exception as e:
            yield {
                "type": "error",
                "text": f"Could not retrieve final agent state: {e}",
            }
        else:
            final_msg = _extract_final_assistant_text(final_state)
            if final_msg:
                # Section break, no banner (Fix 3).
                for piece in _route({
                    "type": "text",
                    "text": "\n\n---\n\n",
                    "persona": "orchestrator",
                }):
                    yield piece
                for piece in _route({
                    "type": "text",
                    "text": final_msg,
                    "persona": "deep_agent",
                }):
                    yield piece

    # --- closing: log telemetry, close artifact (if open), chat summary ----
    duration = time.time() - started
    # Always log the event-kind distribution so we can debug stuck or
    # loopy agent runs by grepping ``docker logs finai-api``.
    log.info(
        "Deep-research run complete: duration=%.1fs tools=%d kinds=%s",
        duration,
        tool_call_count,
        event_kinds,
    )

    if artifact_open:
        yield close_artifact()

    # Final chat line. In artifact mode this is the only thing the
    # user sees in the chat after the streamed status; in inline mode
    # it just appends a single telemetry line below the report.
    if wants_artifact:
        yield chat_text(
            f"Deep research complete: {tool_call_count} tool calls in "
            f"{duration:.1f}s. Full report in the artifact pane →\n"
        )
    else:
        yield chat_text(
            f"\n\n_Deep research complete: {tool_call_count} tool calls "
            f"in {duration:.1f}s._\n"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_NOISY_ARG_KEYS = frozenset({"text", "actuals_context"})
_TRUNCATION_HINT_CHARS = 60


def _sanitize_tool_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Trim huge string arguments for display (e.g. a 40k-char document text).

    The underlying tool still receives the full value; this only affects
    what's rendered in the visible tool-call card.
    """
    if not isinstance(args, dict):
        return {"input": str(args)[:_TRUNCATION_HINT_CHARS]}
    out: Dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and (k in _NOISY_ARG_KEYS or len(v) > 200):
            out[k] = (
                v[:_TRUNCATION_HINT_CHARS].replace("\n", " ")
                + f"... [{len(v)} chars]"
            )
        else:
            out[k] = v
    return out


def _summarize_tool_output(tool_name: str, output: Any) -> str:
    """Produce a one-line label for the visible 'tool returned' banner."""
    if output is None:
        return ""
    # deepagents wraps tool results in a ToolMessage; unwrap the content.
    content = getattr(output, "content", output)
    if isinstance(content, list) and content and isinstance(content[0], dict):
        content = content[0].get("text") or content[0]
    if isinstance(content, str):
        return (content[:80] + "…") if len(content) > 80 else content
    if isinstance(content, dict):
        # Special-case a few of our tool shapes for nicer labels.
        if "filings" in content:
            return f"{len(content.get('filings') or [])} filings"
        if "items" in content:
            return f"{len(content.get('items') or [])} items"
        if "claims" in content:
            return f"{content.get('count', len(content.get('claims') or []))} claims"
        if "verdict" in content:
            return f"verdict: {content.get('verdict')}"
        if "chars_returned" in content:
            return f"{content['chars_returned']:,} chars"
    return ""


def _extract_final_assistant_text(state: Dict[str, Any]) -> str:
    """Pull the last assistant message from a deepagents final state.

    Used only if streaming didn't emit tokens via on_chat_model_stream
    (some model integrations don't fire that event reliably).
    """
    if not isinstance(state, dict):
        return ""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        # LangChain message objects: AIMessage.content is a string
        content = getattr(msg, "content", None)
        role = getattr(msg, "type", None) or getattr(msg, "role", None) or ""
        if role in ("ai", "assistant") and isinstance(content, str) and content.strip():
            return content
    return ""
