"""Retry-and-cache wrapper for streaming LLM calls (moderator + analyst).

Motivation
----------
Persona turns already degrade gracefully when NVIDIA NIM drops the
connection (see :func:`src.core.debate._run_persona_turn_with_cache`)
but the three other streaming LLM calls in the portfolio / stock-research
flows — the moderator opening, the moderator's closing synthesis, and
the no-panel analyst summary — were raw ``async for chunk in llm.astream(...)``
loops with only a bare ``except: yield error; return`` at the bottom. A
single connection blip on any of those three would abort the rest of
the flow (we saw exactly this happen in a demo run:
``Moderator opening failed: Connection error.``).

This module provides :func:`stream_llm_resilient`, a drop-in wrapper
with three behaviours:

1. **Retry on cold failure.** If the underlying stream raises before
   yielding any tokens, retry ``retries`` times (default 1) with a
   small delay. Most NIM blips recover on the second attempt.

2. **Cache on success.** When a stream completes cleanly, its full text
   is written to :class:`src.core.cache.ResponseCache` keyed by
   ``(user_id, query, flow, agent, round=0)``. The next run with the
   same inputs will have a warm fallback.

3. **Replay cache on failure.** If all retries fail AND we have a
   cached entry for this exact key, stream the cached content with a
   visible "⚠️ live failed → serving cached" banner. If the cache is
   cold, emit a soft error banner and return — the surrounding flow
   keeps rendering downstream sections (the panel carries on with the
   subsequent personas / closing brief).

Mid-stream failures (after at least one token) are **not** retried
because retrying would duplicate tokens in the visible output. We log
the partial failure and return early; the caller sees a truncated
section but the rest of the flow still runs.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import AsyncIterator, Awaitable, Callable, Optional

from src.core.cache import ResponseCache, get_cache


log = logging.getLogger("finai.resilient_stream")


# Factory signature: a zero-arg callable that returns a fresh async
# iterator of string chunks. Passing the *function* (not the iterator)
# is critical because retries need a new iterator each attempt.
StreamFactory = Callable[[], AsyncIterator[str]]


async def stream_llm_resilient(
    *,
    stream_factory: StreamFactory,
    user_id: str,
    query: str,
    flow_name: str,
    cache_agent: str,
    cache_agent_title: str,
    retries: int = 1,
    retry_delay: float = 0.5,
    error_label: str = "LLM call",
    cache: Optional[ResponseCache] = None,
) -> AsyncIterator[str]:
    """Stream an LLM response with retry + cache fallback.

    Args:
        stream_factory: zero-arg async function that returns a fresh
            async iterator of string chunks. Called anew on each retry.
        user_id / query / flow_name / cache_agent: cache key parts.
        cache_agent_title: human-readable label stored with the cache
            entry (shown nowhere in the UI right now but handy for
            ``/cache`` debugging endpoints).
        retries: number of retries after a cold failure (default 1,
            i.e. 2 total attempts).
        retry_delay: seconds to wait between retries.
        error_label: used in banner text ("Moderator opening failed",
            "Analyst summary failed", etc.).
        cache: optional override; defaults to the shared singleton.

    Yields:
        String chunks. Never raises — always returns cleanly, either
        with the live / cached content or with an error banner.
    """
    cache = cache or get_cache()

    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        buf: list[str] = []
        tokens_yielded = 0
        try:
            async for chunk in stream_factory():
                if not chunk:
                    continue
                buf.append(chunk)
                tokens_yielded += 1
                yield chunk
        except BaseException as e:  # noqa: BLE001 - catches asyncio.CancelledError too
            last_exc = e
            if tokens_yielded > 0:
                # Mid-stream failure: retrying would double-emit the
                # prefix that already made it to the user, so just log
                # and return. The section ends truncated; the rest of
                # the flow keeps going.
                log.warning(
                    "%s partial-stream failure after %d chunks: %s",
                    error_label,
                    tokens_yielded,
                    e,
                )
                return
            if attempt < retries:
                log.warning(
                    "%s cold failure on attempt %d/%d: %s — retrying in %.1fs",
                    error_label,
                    attempt + 1,
                    retries + 1,
                    e,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                continue
            # All retries exhausted with zero tokens emitted — break
            # out of the attempt loop and go into the cache-replay
            # branch below.
            break
        else:
            # Stream completed cleanly.
            content = "".join(buf).strip()
            if content:
                try:
                    cache.put(
                        user_id=user_id,
                        query=query,
                        flow=flow_name,
                        agent=cache_agent,
                        agent_title=cache_agent_title,
                        content=content,
                        verdict={},
                        round=0,
                    )
                except Exception as e:  # pragma: no cover - defensive
                    log.warning(
                        "Failed to write cache for %s: %s", cache_agent, e
                    )
            return

    # --------------------------------------------------------------
    # Cold failure after all retries — try cache.
    # --------------------------------------------------------------
    cached = cache.get(
        user_id=user_id,
        query=query,
        flow=flow_name,
        agent=cache_agent,
        round=0,
    )
    if cached is not None and cached.content.strip():
        try:
            cached_when = datetime.datetime.fromtimestamp(
                cached.cached_at
            ).strftime("%Y-%m-%d %H:%M")
        except (OSError, ValueError):
            cached_when = "unknown time"
        yield (
            f"\n\n> ⚠️ **Live {error_label} failed after {retries + 1} "
            f"attempt(s) — serving cached response** (cached {cached_when}). "
            "This is the demo safety net; the next successful run will "
            "refresh the cache.\n\n"
        )
        yield cached.content
        yield "\n\n"
        return

    # Cold cache — emit a soft error banner. The surrounding flow
    # keeps rendering subsequent sections.
    yield (
        f"\n\n> ⚠️ **{error_label.capitalize()} failed and no cached "
        f"response is available.** The flow will continue with the "
        f"content already rendered above.\n\n"
        f"> _Error: {last_exc}_\n\n"
    )
