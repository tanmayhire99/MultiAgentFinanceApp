"""Multi-round investor-panel debate with a shared scratchpad.

This is the tier-5 upgrade over the prior parallel-persona panel:

* **Sequential** — personas speak in a fixed order (Buffett -> Wood ->
  Graham) so every subsequent speaker sees the earlier speakers' text.
* **Shared scratchpad** — every persona's contribution is appended to a
  single :class:`PanelScratchpad`. On each turn we render the scratchpad
  into the persona's user message, so the LLM genuinely reasons over
  what the others have said (not just over the portfolio data).
* **Round 1** — "Opening" statements. Each persona still runs their
  full ReAct loop with MCP tool access so the debate is grounded in
  live data gathered at the start.
* **Round 2** — "Rebuttal". Each persona must explicitly name a claim
  from another persona and agree / challenge / refine it. No tool
  access (we already have the data); a plain streamed LLM call over the
  accumulated scratchpad keeps this phase tight.
* **Round 3** — "Final Position". Runs only if stances have shifted
  between rounds 1 and 2 (i.e. the debate hasn't converged). Each
  persona may reaffirm or move their position. Same no-tools format as
  round 2.
* **Convergence** — after each round past the first, we compare the
  persona stances to the previous round. If nobody moved, the debate
  has converged and we stop early.

The output is rendered as a linear stream of :class:`PanelEvent` dicts
so the existing SSE renderer picks it up without changes.
"""
from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.personas.base import (
    PersonaDef,
    build_chat_model,
    parse_verdict,
)
from src.core.cache import CachedResponse, ResponseCache, get_cache
from src.core.panel import (
    PERSONA_ORDER,
    PanelEvent,
    PortfolioContext,
    _stream_persona_events,
)


log = logging.getLogger("finai.debate")


# ---------------------------------------------------------------------------
# Scratchpad
# ---------------------------------------------------------------------------
@dataclass
class ScratchpadEntry:
    """A single persona's contribution in a single round."""

    persona: str  # "buffett" / "wood" / "graham"
    persona_title: str  # display label
    round: int  # 1..MAX_ROUNDS
    content: str  # rationale / body text (verdict block stripped)
    stance: str = "neutral"  # "bullish" / "neutral" / "cautious" / "bearish"
    one_liner: str = ""
    confidence: str = "low"  # "low" / "medium" / "high"
    tools_used: List[str] = field(default_factory=list)


@dataclass
class PanelScratchpad:
    """Shared blackboard the debate loop reads from and appends to."""

    query: str
    portfolio_ctx: Optional[PortfolioContext] = None
    entries: List[ScratchpadEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def entries_for_round(self, round_num: int) -> List[ScratchpadEntry]:
        return [e for e in self.entries if e.round == round_num]

    def latest_entry(self, persona_name: str) -> Optional[ScratchpadEntry]:
        for entry in reversed(self.entries):
            if entry.persona == persona_name:
                return entry
        return None

    def stance_at_round(self, persona_name: str, round_num: int) -> Optional[str]:
        for entry in self.entries:
            if entry.persona == persona_name and entry.round == round_num:
                return entry.stance
        return None

    def has_converged(self, current_round: int) -> bool:
        """True iff all personas reached **consensus** on the same stance.

        The earlier definition was "stances stable across rounds" — i.e.
        every persona kept the SAME stance they held in the previous
        round. That made the panel terminate after round 2 in almost
        every run, because the personas (modelled on real investors)
        rarely flip stance just from one rebuttal.

        The current definition is true consensus: every persona arrives
        at the same stance label (e.g. all three "cautious", or all
        three "bullish"). Stances of e.g. ``bullish / cautious /
        bearish`` no longer trigger convergence — those debates run
        the full ``MAX_ROUNDS`` so the audience sees the personas
        actually engage with each other.

        Convergence still cannot fire on round 1: at minimum two rounds
        are required so the audience sees the rebuttal phase before
        the panel can short-circuit.
        """
        if current_round < 2:
            return False
        stances: List[str] = []
        for persona in PERSONA_ORDER:
            curr = self.stance_at_round(persona.name, current_round)
            if curr is None:
                # Missing data — can't declare consensus.
                return False
            stances.append(curr)
        return len(set(stances)) == 1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_for_persona(
        self, current_persona: str, current_round: int
    ) -> str:
        """Render the scratchpad as a prompt block for the next persona's turn.

        Includes EVERY prior contribution (all rounds, all personas up to
        the current position in the current round), so the speaker sees
        exactly what the room has heard.
        """
        if not self.entries:
            return ""
        lines: List[str] = ["## Debate transcript so far"]
        for entry in self.entries:
            lines.append(
                f"\n### {entry.persona_title} — Round {entry.round}"
                f"  (stance: **{entry.stance}**, confidence: {entry.confidence})"
            )
            body = entry.content.strip() or "(no content)"
            lines.append(body)
            if entry.one_liner:
                lines.append(f"\n_Verdict one-liner: {entry.one_liner}_")
        return "\n".join(lines)

    def stance_evolution_md(self) -> str:
        """Markdown table summarising how each persona's stance evolved."""
        rounds_used = sorted({e.round for e in self.entries})
        if not rounds_used:
            return ""
        headers = ["| Persona |"] + [f" Round {r} |" for r in rounds_used]
        sep = ["|---|"] + ["---|" for _ in rounds_used]
        rows: List[str] = ["".join(headers), "".join(sep)]
        stance_icon = {
            "bullish": "🟢",
            "neutral": "⚪",
            "cautious": "🟡",
            "bearish": "🔴",
        }
        for persona in PERSONA_ORDER:
            row = [f"| **{persona.title}** |"]
            prev_stance: Optional[str] = None
            for r in rounds_used:
                stance = self.stance_at_round(persona.name, r)
                if stance is None:
                    row.append(" — |")
                    continue
                icon = stance_icon.get(stance, "•")
                marker = ""
                if prev_stance is not None and stance != prev_stance:
                    marker = " ↻"  # stance shifted from previous round
                row.append(f" {icon} {stance.title()}{marker} |")
                prev_stance = stance
            rows.append("".join(row))
        return "\n".join(rows)

    def final_verdicts(self) -> List[ScratchpadEntry]:
        """Latest entry per persona in scratchpad order — used by moderator synthesis."""
        seen: Dict[str, ScratchpadEntry] = {}
        for entry in self.entries:
            seen[entry.persona] = entry
        return [seen[p.name] for p in PERSONA_ORDER if p.name in seen]


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------
_ROUND_1_INSTRUCTIONS = (
    "This is **Round 1: Opening Statements** of a multi-round panel debate.\n"
    "- The moderator just opened the debate. Earlier speakers may already "
    "  have spoken; if so, you may reference their points when it's natural.\n"
    "- Produce your own analysis grounded in the data. You may use MCP "
    "  tools to fetch any numbers you want to cite.\n"
    "- Aim for 200-280 words before the VERDICT block.\n"
    "- End with the standard VERDICT / STANCE / CONFIDENCE trailer."
)

_ROUND_2_INSTRUCTIONS = (
    "This is **Round 2: Rebuttal** of a multi-round panel debate.\n"
    "- You have the full Round 1 transcript above.\n"
    "- Pick 1-2 specific claims from another panelist (name them) and "
    "  either AGREE, CHALLENGE, or REFINE them. Cite the panelist by name "
    "  (e.g. \"Buffett says X; I disagree because...\").\n"
    "- If a fellow panelist's argument has genuine merit, say so explicitly. "
    "  You may refine or update your stance — partial concessions are honest.\n"
    "- Do NOT fetch more data — reason over the numbers already in the "
    "  transcript and the portfolio snapshot.\n"
    "- Aim for 120-180 words before the VERDICT block.\n"
    "- End with the standard VERDICT / STANCE / CONFIDENCE trailer."
)

_ROUND_3_INSTRUCTIONS = (
    "This is **Round 3: Reconsideration** of a multi-round panel debate.\n"
    "- The panel did not yet reach consensus. Take a step back and "
    "  reconsider your position.\n"
    "- Pick the OPPOSING panelist whose argument is strongest against your "
    "  view. **Steel-man** their case in 2-3 sentences — articulate the "
    "  best version of their argument as they would put it themselves, "
    "  using their numbers and framing.\n"
    "- Then ask yourself honestly: does that steel-manned argument change "
    "  your evaluation? If yes, update your stance and explain what "
    "  changed your mind. If no, explain why the steel-manned version "
    "  is still not enough to move you.\n"
    "- No new tool calls. Aim for 100-160 words before the VERDICT block.\n"
    "- End with the standard VERDICT / STANCE / CONFIDENCE trailer."
)

_ROUND_4_INSTRUCTIONS = (
    "This is **Round 4: Bridge-Building** of a multi-round panel debate.\n"
    "- The panel still has not reached consensus. The goal of this round "
    "  is to find common ground — not to argue harder.\n"
    "- Identify at least one specific point of GENUINE agreement with "
    "  each of the other two panelists. Name the panelist and the point "
    "  (e.g. \"I agree with Wood that the AI tailwind is real; my caution "
    "  is about the entry multiple, not the underlying business\").\n"
    "- After acknowledging the agreement, restate where your stance still "
    "  differs and why. Be precise about the irreducible disagreement.\n"
    "- No new tool calls. Aim for 100-150 words before the VERDICT block.\n"
    "- End with the standard VERDICT / STANCE / CONFIDENCE trailer."
)

_ROUND_5_INSTRUCTIONS = (
    "This is **Round 5: Final Position** — the closing round.\n"
    "- This is the last round. State your closing position.\n"
    "- If the panel converged in spirit (you all see the same picture even "
    "  if your stance labels differ slightly), say so.\n"
    "- If you genuinely cannot bridge the disagreement, say so explicitly: "
    "  \"We agree to disagree on X because [one-sentence summary of the "
    "  irreducible difference]\". This is a legitimate outcome.\n"
    "- Stay short: 80-120 words before the VERDICT block.\n"
    "- End with the standard VERDICT / STANCE / CONFIDENCE trailer."
)


_ROUND_INSTRUCTIONS: Dict[int, str] = {
    1: _ROUND_1_INSTRUCTIONS,
    2: _ROUND_2_INSTRUCTIONS,
    3: _ROUND_3_INSTRUCTIONS,
    4: _ROUND_4_INSTRUCTIONS,
    5: _ROUND_5_INSTRUCTIONS,
}


def _compose_round_user_message(
    persona: PersonaDef,
    query: str,
    scratchpad: PanelScratchpad,
    round_num: int,
) -> str:
    """Build the user message for a specific persona's turn in a specific round."""
    parts: List[str] = [f"User's question: {query}"]
    ctx = scratchpad.portfolio_ctx
    if ctx and ctx.has_data():
        parts.append("")
        parts.append(ctx.persona_context_block())
    transcript = scratchpad.render_for_persona(persona.name, round_num)
    if transcript:
        parts.append("")
        parts.append(transcript)
    parts.append("")
    parts.append(
        _ROUND_INSTRUCTIONS.get(round_num, _ROUND_1_INSTRUCTIONS)
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Rebuttal executor (plain streamed LLM call, no ReAct loop, no tools)
# ---------------------------------------------------------------------------
from src.personas.base import PERSONA_API_KEY_SLOT


async def _stream_rebuttal(
    persona: PersonaDef,
    user_message: str,
    round_num: int,
) -> AsyncIterator[PanelEvent]:
    """Stream a persona's rebuttal or final-position turn.

    Uses a plain :func:`build_chat_model` streamed call (no tools, no
    ReAct loop) — rebuttals are pure reasoning over the accumulated
    scratchpad, so the ReAct machinery is unnecessary overhead.
    """
    from src.core.panel import VerdictTrimFilter

    slot = PERSONA_API_KEY_SLOT.get(persona.name, "primary")
    llm = build_chat_model(
        temperature=0.3,
        max_tokens=700,
        streaming=True,
        api_key_slot=slot,
    )

    system_prompt = (
        f"{persona.system_prompt.strip()}\n\n"
        "### Output contract (rebuttal)\n"
        f"This is round {round_num} of a panel debate. Reason over the "
        "debate transcript you are shown and respond in your own voice. "
        "Do NOT call any tools. Finish your message with EXACTLY this "
        "final block, on its own lines, and nothing after it:\n"
        "VERDICT: <one sentence, <=25 words>\n"
        "STANCE: <bullish|neutral|cautious|bearish>\n"
        "CONFIDENCE: <low|medium|high>"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    verdict_filter = VerdictTrimFilter()
    yield {
        "type": "text",
        "text": f"\n\n_{persona.title} is responding…_\n\n",
        "persona": persona.name,
    }
    final_parts: List[str] = []
    try:
        async for chunk in llm.astream(messages):
            content = getattr(chunk, "content", None)
            if not content:
                continue
            final_parts.append(content)
            visible = verdict_filter.push(content)
            if visible:
                yield {
                    "type": "text",
                    "text": visible,
                    "persona": persona.name,
                }
    except Exception as e:
        log.exception("Rebuttal stream failed for %s", persona.name)
        yield {
            "type": "error",
            "text": f"{persona.title} rebuttal failed: {e}",
        }
        return

    if not verdict_filter.closed:
        tail = verdict_filter.flush()
        if tail:
            yield {"type": "text", "text": tail, "persona": persona.name}

    final_text = "".join(final_parts)
    verdict = parse_verdict(final_text, persona)
    yield {
        "type": "persona_verdict",
        "persona": persona.name,
        "title": persona.title,
        "stance": verdict.get("stance", "neutral"),
        "one_liner": verdict.get("one_liner", ""),
        "confidence": verdict.get("confidence", "low"),
        "tools_used": [],
    }


# ---------------------------------------------------------------------------
# Round 1 executor (full ReAct persona with tool access + scratchpad-aware
# user message)
# ---------------------------------------------------------------------------
async def _stream_round_one_persona(
    persona: PersonaDef,
    query: str,
    scratchpad: PanelScratchpad,
) -> AsyncIterator[PanelEvent]:
    """Run the full ReAct persona for round 1 with the scratchpad injected.

    Round 1 keeps full MCP tool access so the panel starts grounded in
    live numbers. The custom user message - which includes earlier
    Round-1 speakers' text if any - is fed via the
    ``user_message_override`` parameter on
    :func:`src.core.panel._stream_persona_events`.
    """
    user_message = _compose_round_user_message(persona, query, scratchpad, 1)
    async for ev in _stream_persona_events(
        persona,
        query,
        scratchpad.portfolio_ctx,
        user_message_override=user_message,
    ):
        yield ev


# ---------------------------------------------------------------------------
# Debate loop
# ---------------------------------------------------------------------------
# Max rounds the panel will run before stopping unconditionally. The
# loop short-circuits early via :meth:`PanelScratchpad.has_converged`
# the moment all three personas agree on the same stance label, so
# typical runs are 2-4 rounds. The cap is in place so a genuinely
# divergent panel (e.g. bullish / cautious / bearish across the three
# personas) doesn't run indefinitely.
MAX_ROUNDS = 5


_STANCE_ICON = {
    "bullish": "🟢",
    "neutral": "⚪",
    "cautious": "🟡",
    "bearish": "🔴",
}


async def _run_persona_turn_with_cache(
    *,
    persona: PersonaDef,
    query: str,
    scratchpad: "PanelScratchpad",
    round_num: int,
    user_id: str,
    flow_name: str,
    cache: ResponseCache,
) -> AsyncIterator[PanelEvent]:
    """Run a single persona turn with cache-aware fallback.

    Happy path
    ----------
    1. Stream live events from the ReAct loop (Round 1) or the plain
       streamed LLM call (Rounds 2+) to the caller as they arrive.
    2. On successful completion, persist ``content`` + ``verdict`` to
       the :class:`ResponseCache` so the next run has a warm fallback.

    Failure path
    ------------
    If any exception escapes the live stream (typically an
    ``httpx.RemoteProtocolError`` from NIM dropping the connection),
    we:

    * Look up the cache with the exact ``(user_id, query, flow, persona, round)`` key.
    * If a cache entry exists, yield a visible "⚠️ live failed, using cached"
      banner followed by the cached text + verdict card.
    * If no cache exists (cold cache), yield an error event and skip the turn.

    The final yield is a private ``_scratchpad_entry`` event the caller
    consumes to append to the shared :class:`PanelScratchpad`. That way
    downstream speakers still see the (cached) content in their
    transcript, and convergence detection still works even when we're
    serving from cache.
    """
    rationale_parts: List[str] = []
    verdict_info: Dict[str, object] = {}
    tools_used: List[str] = []
    live_succeeded = False
    live_exception: Optional[BaseException] = None

    if round_num == 1:
        stream = _stream_round_one_persona(persona, query, scratchpad)
    else:
        user_msg = _compose_round_user_message(
            persona, query, scratchpad, round_num
        )
        stream = _stream_rebuttal(persona, user_msg, round_num)

    try:
        async for ev in stream:
            etype = ev.get("type")
            if etype == "text" and ev.get("persona") == persona.name:
                text = ev.get("text", "")
                if text:
                    rationale_parts.append(text)
            elif etype == "persona_verdict":
                verdict_info = dict(ev)
                tools_used = list(ev.get("tools_used") or [])
            yield ev
        live_succeeded = True
    except BaseException as e:  # catches asyncio.CancelledError too
        live_exception = e
        log.exception(
            "Persona %s (round %d) live stream failed: %s", persona.name, round_num, e
        )

    # ------------------------------------------------------------------
    # Success path: update cache and emit the scratchpad entry.
    # ------------------------------------------------------------------
    if live_succeeded:
        content = "".join(rationale_parts).strip()
        verdict_for_cache = {
            "stance": str(verdict_info.get("stance") or "neutral"),
            "one_liner": str(verdict_info.get("one_liner") or ""),
            "confidence": str(verdict_info.get("confidence") or "low"),
            "tools_used": tools_used,
        }
        try:
            cache.put(
                user_id=user_id,
                query=query,
                flow=flow_name,
                agent=persona.name,
                agent_title=persona.title,
                content=content,
                verdict=verdict_for_cache,
                round=round_num,
            )
        except Exception as e:  # pragma: no cover - defensive
            log.warning("Failed to write cache for %s r%d: %s", persona.name, round_num, e)

        yield {
            "type": "_scratchpad_entry",
            "entry": ScratchpadEntry(
                persona=persona.name,
                persona_title=persona.title,
                round=round_num,
                content=content,
                stance=verdict_for_cache["stance"],
                one_liner=verdict_for_cache["one_liner"],
                confidence=verdict_for_cache["confidence"],
                tools_used=tools_used,
            ),
        }
        return

    # ------------------------------------------------------------------
    # Failure path: try cache, degrade gracefully.
    # ------------------------------------------------------------------
    cached = cache.get(
        user_id=user_id,
        query=query,
        flow=flow_name,
        agent=persona.name,
        round=round_num,
    )
    if cached is None:
        # Cold cache - best we can do is emit an error and keep the
        # debate going with what we have.
        yield {
            "type": "text",
            "text": (
                f"\n\n> ⚠️ **Live call failed for {persona.title} in round "
                f"{round_num} and no cached response is available.** The "
                f"panel continues with the other analysts.\n\n"
                f"> _Error: {live_exception}_\n\n"
            ),
            "persona": persona.name,
        }
        yield {
            "type": "_scratchpad_entry",
            "entry": ScratchpadEntry(
                persona=persona.name,
                persona_title=persona.title,
                round=round_num,
                content="(live call failed; no cached response available)",
                stance="neutral",
                one_liner=f"{persona.title} was unavailable in round {round_num}.",
                confidence="low",
                tools_used=[],
            ),
        }
        return

    # Warm cache hit — replay cached content with a visible banner.
    cached_age_s = max(0.0, time.time() - cached.cached_at)
    try:
        cached_when = datetime.datetime.fromtimestamp(
            cached.cached_at
        ).strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError):
        cached_when = "unknown time"

    yield {
        "type": "text",
        "text": (
            f"\n\n> ⚠️ **Live call failed mid-stream — serving cached "
            f"response** (cached {cached_when}, {int(cached_age_s)}s ago). "
            f"This is the demo safety net; the next successful run will "
            f"refresh the cache.\n\n"
        ),
        "persona": persona.name,
    }
    yield {
        "type": "text",
        "text": cached.content + "\n\n",
        "persona": persona.name,
    }
    cached_verdict = cached.verdict or {}
    yield {
        "type": "persona_verdict",
        "persona": persona.name,
        "title": persona.title,
        "stance": cached_verdict.get("stance", "neutral"),
        "one_liner": cached_verdict.get("one_liner", ""),
        "confidence": cached_verdict.get("confidence", "low"),
        "tools_used": list(cached_verdict.get("tools_used") or []),
    }
    yield {
        "type": "_scratchpad_entry",
        "entry": ScratchpadEntry(
            persona=persona.name,
            persona_title=persona.title,
            round=round_num,
            content=cached.content,
            stance=str(cached_verdict.get("stance") or "neutral"),
            one_liner=str(cached_verdict.get("one_liner") or ""),
            confidence=str(cached_verdict.get("confidence") or "low"),
            tools_used=list(cached_verdict.get("tools_used") or []),
        ),
    }


async def run_debate_loop(
    query: str,
    portfolio_ctx: Optional[PortfolioContext] = None,
    max_rounds: int = MAX_ROUNDS,
    *,
    user_id: str = "demo",
    flow_name: str = "panel",
) -> AsyncIterator[PanelEvent]:
    """Run a sequential multi-round debate with a shared scratchpad.

    Each persona turn is wrapped in cache-aware fallback logic. If the
    live NVIDIA NIM streaming call fails mid-response (for example with
    ``peer closed connection without sending complete message body``),
    we transparently serve the most recent cached response for that
    exact ``(user_id, query, flow, persona, round)`` key with a visible
    banner. On success, the cache is overwritten with the fresh content
    so the next demo run has the latest known-good fallback.

    The final yield is a special
    ``{"type": "_debate_done", "scratchpad": PanelScratchpad}`` payload
    the caller consumes to feed the moderator synthesis.
    """
    scratchpad = PanelScratchpad(query=query, portfolio_ctx=portfolio_ctx)
    cache = get_cache()

    converged_round: Optional[int] = None
    for round_num in range(1, max_rounds + 1):
        yield {
            "type": "header",
            "text": _round_heading(round_num),
        }
        yield {
            "type": "text",
            "text": _round_preamble(round_num),
            "persona": "moderator",
        }

        for persona in PERSONA_ORDER:
            yield {
                "type": "header",
                "text": f"\n#### {persona.title}  \n",
            }

            entry = None
            async for ev in _run_persona_turn_with_cache(
                persona=persona,
                query=query,
                scratchpad=scratchpad,
                round_num=round_num,
                user_id=user_id,
                flow_name=flow_name,
                cache=cache,
            ):
                if ev.get("type") == "_scratchpad_entry":
                    entry = ev.get("entry")  # type: ignore[assignment]
                    continue
                yield ev

            if entry is not None:
                scratchpad.entries.append(entry)

        # Consensus check (skipped after round 1; needs at least 2 rounds
        # so the audience always sees the rebuttal phase before the panel
        # can short-circuit).
        if round_num >= 2 and scratchpad.has_converged(round_num):
            converged_round = round_num
            consensus_stance = scratchpad.stance_at_round(
                PERSONA_ORDER[0].name, round_num
            ) or ""
            stance_label = consensus_stance.title() if consensus_stance else ""
            yield {
                "type": "header",
                "text": (
                    f"\n### ✅ Panel Reached Consensus After Round {round_num}\n\n"
                ),
            }
            remaining = max_rounds - round_num
            skipped_clause = (
                f" rounds {round_num + 1}–{max_rounds} are skipped"
                if remaining > 0 else ""
            )
            stance_clause = (
                f" on **{stance_label}**" if stance_label else ""
            )
            yield {
                "type": "text",
                "text": (
                    f"_All three panelists converged{stance_clause} "
                    f"in round {round_num}{skipped_clause}. The moderator "
                    "will synthesise the agreed view below._\n\n"
                ),
                "persona": "moderator",
            }
            break

    # If we exhausted max_rounds without consensus, say so explicitly.
    if converged_round is None:
        yield {
            "type": "header",
            "text": f"\n### ⚖️ Max Rounds Reached — Panel Remains Divergent\n\n",
        }
        yield {
            "type": "text",
            "text": (
                "_After {r} rounds the panelists still hold different "
                "stances. This is itself a useful signal — the moderator "
                "will synthesise the irreducible disagreement below._\n\n"
            ).format(r=max_rounds),
            "persona": "moderator",
        }

    # Stance-evolution summary
    evolution_md = scratchpad.stance_evolution_md()
    if evolution_md:
        yield {
            "type": "text",
            "text": (
                "\n#### Stance Evolution\n\n"
                + evolution_md
                + "\n\n_↻ marks a round in which the persona shifted stance._\n\n"
            ),
            "persona": "moderator",
        }

    # Hand off to caller (moderator synthesis feeds on the scratchpad).
    yield {"type": "_debate_done", "scratchpad": scratchpad}


# ---------------------------------------------------------------------------
# Copy for per-round headers / preambles
# ---------------------------------------------------------------------------
def _round_heading(round_num: int) -> str:
    titles = {
        1: "\n### 🟦 Round 1 — Opening Statements\n\n",
        2: "\n### 🟨 Round 2 — Rebuttal\n\n",
        3: "\n### 🟧 Round 3 — Reconsideration (steel-man)\n\n",
        4: "\n### 🟪 Round 4 — Bridge-Building\n\n",
        5: "\n### 🟥 Round 5 — Final Position\n\n",
    }
    return titles.get(round_num, f"\n### Round {round_num}\n\n")


def _round_preamble(round_num: int) -> str:
    preambles = {
        1: (
            "_Each analyst opens with their independent read of the "
            "portfolio. Later speakers may already reference earlier "
            "openings — that's a feature, not a bug: the debate starts "
            "collaborating immediately._\n\n"
        ),
        2: (
            "_Each analyst now has the full Round 1 transcript above. "
            "Rebuttals must cite at least one specific claim from another "
            "panelist. No new tool calls — this round is pure debate._\n\n"
        ),
        3: (
            "_The panel did not yet reach consensus. Each analyst now "
            "**steel-mans** the strongest argument against their own view "
            "— articulating it as the opposing panelist would, then "
            "deciding honestly whether it changes their evaluation._\n\n"
        ),
        4: (
            "_Still no consensus. The panel pivots from arguing to "
            "**bridge-building** — each analyst names at least one point "
            "of genuine agreement with each of the other two panelists "
            "before restating where they still differ._\n\n"
        ),
        5: (
            "_This is the closing round. Each panelist states their "
            "final position; if the panel still cannot bridge the "
            "disagreement, they explicitly flag the irreducible "
            "difference for the moderator's synthesis._\n\n"
        ),
    }
    return preambles.get(round_num, "")


# ---------------------------------------------------------------------------
# Moderator synthesis prompt + scratchpad formatter (shared by planner
# pipeline and static flows)
# ---------------------------------------------------------------------------
DEBATE_SYNTH_SYSTEM = (
    "You are the moderator of the FinAI Investor Panel. The three "
    "analyst personas have just completed a multi-round sequential "
    "debate on the user's portfolio, with a shared scratchpad visible "
    "to every speaker. Your job is to write the **Closing Brief** "
    "section, grounded in both the live portfolio snapshot and the "
    "full debate transcript you will be shown.\n\n"
    "Follow this structure exactly:\n\n"
    "**Where the panel converged:** 1-2 bullets on points the three "
    "analysts now agree on. If the panel never converged, say so.\n"
    "**Where the panel remained divergent:** 1-2 bullets naming who is "
    "on each side and citing a specific claim from the transcript.\n"
    "**How stances evolved:** 2-3 sentences summarising movements "
    "across rounds. Call out any persona who shifted stance and the "
    "argument that moved them.\n"
    "**What it means for your portfolio:** 2-3 sentences connecting "
    "the (resolved or persisting) disagreement back to specific "
    "holdings or concentration flags from the snapshot.\n"
    "**What to watch next:** 2-4 bullets with educational indicators "
    "the user should monitor. No buy/sell calls, no price targets.\n\n"
    "Rules:\n"
    "- Aim for 260-380 words total.\n"
    "- Cite panelists by name when referencing a claim.\n"
    "- Quote specific numbers from the portfolio snapshot where useful.\n"
    "- Do NOT invent new metrics.\n"
    "- Do NOT include a caveat - a standalone disclaimer follows."
)


def format_scratchpad_for_moderator(scratchpad: PanelScratchpad) -> str:
    """Render the full debate transcript for the moderator synthesis prompt."""
    lines: list[str] = [f"Query: {scratchpad.query}", ""]
    rounds_used = sorted({e.round for e in scratchpad.entries})
    for r in rounds_used:
        lines.append(f"=== Round {r} ===")
        for entry in scratchpad.entries_for_round(r):
            lines.append(
                f"\n### {entry.persona_title} — stance: {entry.stance} "
                f"({entry.confidence} confidence)"
            )
            if entry.one_liner:
                lines.append(f"One-liner: {entry.one_liner}")
            if entry.content:
                lines.append(entry.content)
            lines.append("")
    evolution = scratchpad.stance_evolution_md()
    if evolution:
        lines.append("Stance evolution:")
        lines.append(evolution)
    return "\n".join(lines)
