"""Persistent per-user memory (the #1 demo→product gap, FROM_DEMO_TO_PRODUCT §4).

Without this, every session is cold — the assistant re-learns who the user is on
every turn. This module is the system's long-term memory, keyed by ``user_id``:

* a small structured **profile** (risk tolerance, horizon, goals), and
* a rolling list of **notes** (salient facts + recent topics).

It is read at prompt-build time (so every agent — planner, research, panel,
synthesizer — can personalise) and written once per turn via :func:`observe`.

Design choices:

* **SQLite** (stdlib, zero-dependency, single file): durable, restart-safe, and
  trivially queryable. Path overridable via ``FINAI_MEMORY_DB`` (tests isolate it).
* Extraction in :func:`observe` is **deterministic and conservative** (no extra
  LLM call): it only records signals it is confident about, plus a topic note.
  :func:`_extract_profile_signals` is isolated so an LLM-based extractor can be
  swapped in later without touching the store or call sites.
* Memory is only active for an **authenticated** user — never the shared ``demo``
  / ``anonymous`` identities — so the demo portfolio user doesn't accumulate state.
* Every public function is wrapped to **never raise into the request path**:
  memory is an enhancement, and a memory failure must not break an answer.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("finai.memory")

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "finai_memory.db",
)

_RESERVED_USERS = {"", "demo", "anonymous", "guest"}
_PROFILE_FIELDS = ("risk_tolerance", "horizon", "goals")
_MAX_NOTE_LEN = 200


def _path() -> str:
    return os.environ.get("FINAI_MEMORY_DB", _DEFAULT_PATH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_real_user(user_id: Optional[str]) -> bool:
    """Memory is only kept for an authenticated, non-shared identity."""
    return bool(user_id) and user_id.strip().lower() not in _RESERVED_USERS


def _connect() -> sqlite3.Connection:
    p = Path(_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=3000")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS profile (
            user_id TEXT PRIMARY KEY,
            risk_tolerance TEXT, horizon TEXT, goals TEXT, updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS note (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL, kind TEXT NOT NULL, content TEXT NOT NULL,
            created_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_note_user ON note(user_id, id);
        """
    )
    return con


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def get_profile(user_id: str) -> Dict[str, str]:
    if not is_real_user(user_id):
        return {}
    try:
        with _connect() as con:
            row = con.execute("SELECT * FROM profile WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return {}
        return {k: row[k] for k in _PROFILE_FIELDS if row[k]}
    except Exception:
        log.exception("memory.get_profile failed")
        return {}


def set_profile(user_id: str, **fields: Optional[str]) -> None:
    """Upsert the non-empty profile fields for a user (merges with existing)."""
    if not is_real_user(user_id):
        return
    updates = {k: v.strip() for k, v in fields.items()
               if k in _PROFILE_FIELDS and isinstance(v, str) and v.strip()}
    if not updates:
        return
    try:
        with _connect() as con:
            existing = con.execute(
                "SELECT * FROM profile WHERE user_id=?", (user_id,)
            ).fetchone()
            merged = {k: (existing[k] if existing else None) for k in _PROFILE_FIELDS}
            merged.update(updates)
            con.execute(
                """INSERT INTO profile (user_id, risk_tolerance, horizon, goals, updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                     risk_tolerance=excluded.risk_tolerance, horizon=excluded.horizon,
                     goals=excluded.goals, updated_at=excluded.updated_at""",
                (user_id, merged["risk_tolerance"], merged["horizon"],
                 merged["goals"], _now()),
            )
    except Exception:
        log.exception("memory.set_profile failed")


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
def remember(user_id: str, content: str, kind: str = "fact") -> None:
    """Append a note, skipping an exact duplicate of the most recent same-kind note."""
    if not is_real_user(user_id) or not content or not content.strip():
        return
    content = content.strip()[:_MAX_NOTE_LEN]
    try:
        with _connect() as con:
            last = con.execute(
                "SELECT content FROM note WHERE user_id=? AND kind=? ORDER BY id DESC LIMIT 1",
                (user_id, kind),
            ).fetchone()
            if last and last["content"] == content:
                return
            con.execute(
                "INSERT INTO note (user_id, kind, content, created_at) VALUES (?,?,?,?)",
                (user_id, kind, content, _now()),
            )
    except Exception:
        log.exception("memory.remember failed")


def _notes(user_id: str, limit: int, query: Optional[str]) -> List[str]:
    with _connect() as con:
        rows = con.execute(
            "SELECT content FROM note WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, max(limit * 3, limit)),
        ).fetchall()
    items = [r["content"] for r in rows]
    if query:  # float notes that share a meaningful word with the query to the front
        qwords = {w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())}
        items.sort(key=lambda c: 0 if qwords & set(re.findall(r"[a-zA-Z]{4,}", c.lower())) else 1)
    return items[:limit]


# ---------------------------------------------------------------------------
# Recall / observe — the two call sites the pipeline uses
# ---------------------------------------------------------------------------
def recall(user_id: str, query: Optional[str] = None, max_notes: int = 6) -> str:
    """A compact markdown block of what we know about ``user_id`` (or "").

    Safe to call for anyone — returns "" for unauthenticated/empty users so
    callers can unconditionally inject the result.
    """
    if not is_real_user(user_id):
        return ""
    try:
        profile = get_profile(user_id)
        notes = _notes(user_id, max_notes, query)
    except Exception:
        log.exception("memory.recall failed")
        return ""
    if not profile and not notes:
        return ""
    lines = ["### What we know about this user (persistent memory)"]
    label = {"risk_tolerance": "Risk tolerance", "horizon": "Horizon", "goals": "Goals"}
    for k in _PROFILE_FIELDS:
        if profile.get(k):
            lines.append(f"- {label[k]}: {profile[k]}")
    if notes:
        lines.append("Recent things this user has asked about / told us:")
        lines.extend(f"- {n}" for n in notes)
    lines.append(
        "Use this to personalise tone and emphasis. Do not invent facts beyond it; "
        "if it conflicts with the current query, prefer the current query."
    )
    return "\n".join(lines)


# Conservative keyword signals. Each maps a normalized value to trigger phrases.
_RISK_SIGNALS = {
    "conservative": ("conservative", "risk-averse", "risk averse", "low risk",
                     "low-risk", "capital preservation", "preserve capital", "play it safe"),
    "aggressive": ("aggressive", "high risk", "high-risk", "risk tolerant",
                   "risk-tolerant", "speculative", "high growth", "swing for"),
    "moderate": ("moderate risk", "balanced portfolio", "balanced approach", "medium risk"),
}
_HORIZON_SIGNALS = {
    "long-term": ("long-term", "long term", "retirement", "retire", "decades",
                  "buy and hold", "buy-and-hold", "hold for years"),
    "short-term": ("short-term", "short term", "day trad", "swing trad",
                   "next month", "this week", "quick profit", "quick gains"),
}
_GOAL_SIGNALS = {
    "retirement planning": ("retire", "retirement", "401k", "pension", "nest egg"),
    "buying a home": ("buy a house", "buy a home", "down payment", "mortgage", "first home"),
    "education funding": ("college", "tuition", "education fund", "child's education", "529"),
    "passive income": ("dividend", "passive income", "income stream", "monthly income"),
    "wealth building": ("build wealth", "grow my wealth", "long-term wealth", "financial freedom"),
}


def _match(text: str, signals: dict) -> Optional[str]:
    for value, phrases in signals.items():
        if any(p in text for p in phrases):
            return value
    return None


def _extract_profile_signals(text: str) -> Dict[str, str]:
    """Deterministic, conservative extraction of profile signals from user text.

    Isolated so an LLM-based extractor can replace it later without touching the
    store or the pipeline call sites. Only returns fields it is confident about.
    """
    t = (text or "").lower()
    out: Dict[str, str] = {}
    for field, signals in (("risk_tolerance", _RISK_SIGNALS),
                           ("horizon", _HORIZON_SIGNALS), ("goals", _GOAL_SIGNALS)):
        val = _match(t, signals)
        if val:
            out[field] = val
    return out


def observe(user_id: str, query: str, answer: str = "") -> None:
    """Record what we learned this turn: profile signals + a topic note.

    Extraction is intentionally conservative and reads the *user's* words
    (the query), not the assistant's answer. Never raises.
    """
    if not is_real_user(user_id) or not query or not query.strip():
        return
    try:
        signals = _extract_profile_signals(query)
        if signals:
            set_profile(user_id, **signals)
        topic = " ".join(query.strip().split())
        remember(user_id, topic, kind="topic")
    except Exception:
        log.exception("memory.observe failed")


def forget(user_id: str) -> None:
    """Erase a user's memory (right-to-be-forgotten / test helper)."""
    try:
        with _connect() as con:
            con.execute("DELETE FROM profile WHERE user_id=?", (user_id,))
            con.execute("DELETE FROM note WHERE user_id=?", (user_id,))
    except Exception:
        log.exception("memory.forget failed")


def list_real_users() -> tuple[str, ...]:
    """Return every user_id that has a profile row (excludes demo/anonymous/guest).

    The scheduler reads this to know *who* to scan; memory.gate ensures only
    real (authed) users ever get a row here. Safe for a hot-loop caller — the
    profile table is small.
    """
    try:
        with _connect() as con:
            rows = con.execute("SELECT user_id FROM profile ORDER BY user_id").fetchall()
        return tuple(r["user_id"] for r in rows if is_real_user(r["user_id"]))
    except Exception:
        log.exception("memory.list_real_users failed")
        return ()


__all__ = [
    "recall", "observe", "remember", "get_profile", "set_profile",
    "forget", "is_real_user", "list_real_users",
]
