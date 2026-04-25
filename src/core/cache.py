"""Disk-backed response cache for demo reliability.

Motivation
----------
NVIDIA NIM occasionally drops streaming responses mid-flight ("peer closed
connection without sending complete message body"). Most of the time this
is rate-limiting on the free tier; sometimes it's a transient server-side
close. Either way, a single dropped connection tanks an entire
multi-round persona debate - very bad in a live demo.

Strategy
--------
Every persona turn (and every moderator turn) is cached on disk keyed on
``(user_id, query, flow, agent, round)``. When a live call fails, the
caller falls back to the most recent cached value for that exact key and
streams it back to the user with a visible "⚠️ served from cache" note.
When a live call *succeeds*, the cache entry is overwritten with the
fresh content. So the cache is always "latest known-good".

This is explicitly a demo safety net, not a production caching layer:

* Cold cache + live failure = persona is skipped. Audience sees the
  error; debate continues with fewer personas.
* Warm cache + live failure = audience sees cached content + banner.
* Warm cache + live success = cache refreshed, audience sees live output.

Storage
-------
JSON files in ``data/response_cache/`` (bind-mounted into the container
so cache survives ``docker compose up --build``). One file per cache
entry, named by a short hash of the cache key.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


log = logging.getLogger("finai.cache")


# Default cache root. Can be overridden via ``FINAI_RESPONSE_CACHE_DIR``.
_DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "response_cache",
)


@dataclass
class CachedResponse:
    """Shape of what we store per persona turn / moderator turn."""

    agent: str  # persona name or "moderator_opening" / "moderator_synthesis" / ...
    agent_title: str  # display label
    round: int  # 1..N for debate turns; 0 for non-debate agents (moderator, analysts)
    user_id: str
    query: str  # original query text (for audit / debugging)
    content: str  # the rationale text the agent produced
    verdict: Dict[str, Any]  # stance / one_liner / confidence / tools_used (may be empty)
    cached_at: float  # unix timestamp


class ResponseCache:
    """Filesystem-backed latest-known-good cache for agent responses."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root or os.environ.get("FINAI_RESPONSE_CACHE_DIR") or _DEFAULT_DIR
        try:
            os.makedirs(self.root, exist_ok=True)
        except OSError as e:  # pragma: no cover - defensive
            log.warning("Could not create cache dir %s: %s", self.root, e)

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------
    @staticmethod
    def key(
        *,
        user_id: str,
        query: str,
        flow: str,
        agent: str,
        round: int = 0,
    ) -> str:
        """Stable hash of the cache key. Normalises query whitespace + case.

        We intentionally DO NOT fold over model / temperature - those are
        effectively fixed per deployment and changing them is a
        deliberate "blow the cache" signal (just ``rm -rf data/response_cache/``).
        """
        raw = "|".join(
            [
                user_id.strip() or "anonymous",
                " ".join(query.lower().split()),
                flow,
                agent,
                str(round),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _path_for(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(
        self,
        *,
        user_id: str,
        query: str,
        flow: str,
        agent: str,
        round: int = 0,
    ) -> Optional[CachedResponse]:
        """Return the latest cached entry for this key, or ``None``."""
        k = self.key(
            user_id=user_id, query=query, flow=flow, agent=agent, round=round
        )
        path = self._path_for(k)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning("Failed to read cache file %s: %s", path, e)
            return None
        try:
            return CachedResponse(
                agent=data["agent"],
                agent_title=data.get("agent_title") or data["agent"],
                round=int(data.get("round", 0)),
                user_id=data.get("user_id", ""),
                query=data.get("query", ""),
                content=data.get("content", ""),
                verdict=data.get("verdict") or {},
                cached_at=float(data.get("cached_at", 0)),
            )
        except (KeyError, TypeError, ValueError) as e:
            log.warning("Cache file %s has unexpected shape: %s", path, e)
            return None

    def put(
        self,
        *,
        user_id: str,
        query: str,
        flow: str,
        agent: str,
        agent_title: str,
        content: str,
        verdict: Optional[Dict[str, Any]] = None,
        round: int = 0,
    ) -> None:
        """Overwrite the cache entry for this key with the new content."""
        if not content.strip():
            # Don't cache empty outputs; they're not useful fallbacks.
            return
        k = self.key(
            user_id=user_id, query=query, flow=flow, agent=agent, round=round
        )
        path = self._path_for(k)
        try:
            payload = {
                "agent": agent,
                "agent_title": agent_title,
                "round": round,
                "user_id": user_id,
                "query": query,
                "flow": flow,
                "content": content,
                "verdict": verdict or {},
                "cached_at": time.time(),
            }
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            # Atomic replace so partially-written files don't poison the cache
            os.replace(tmp_path, path)
            log.info(
                "Cached %s/%s/round=%d for query %r (%d chars)",
                flow,
                agent,
                round,
                query[:60],
                len(content),
            )
        except OSError as e:  # pragma: no cover - defensive
            log.warning("Failed to write cache file %s: %s", path, e)

    def stats(self) -> Dict[str, Any]:
        """Return a small dict describing the cache state - for /health."""
        try:
            files = [f for f in os.listdir(self.root) if f.endswith(".json")]
        except OSError:
            files = []
        return {"dir": self.root, "entries": len(files)}

    def list_entries(self) -> List[Dict[str, Any]]:
        """Return a summary of every cache entry (for debugging / demo pre-warm)."""
        rows: List[Dict[str, Any]] = []
        try:
            names = sorted(os.listdir(self.root))
        except OSError:
            return rows
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.root, name), "r", encoding="utf-8") as f:
                    d = json.load(f)
                rows.append(
                    {
                        "key": name.removesuffix(".json"),
                        "flow": d.get("flow"),
                        "agent": d.get("agent"),
                        "round": d.get("round"),
                        "query": (d.get("query") or "")[:80],
                        "cached_at": d.get("cached_at"),
                        "content_chars": len(d.get("content") or ""),
                        "stance": (d.get("verdict") or {}).get("stance"),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
        return rows


# Shared singleton so all flows use the same directory.
_SHARED: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    global _SHARED
    if _SHARED is None:
        _SHARED = ResponseCache()
    return _SHARED
