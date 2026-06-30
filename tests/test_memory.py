"""Tests for persistent per-user memory (src.core.memory) and its integration
into the planner user message + every ScopedAgent's system prompt.

Deterministic, offline, no LLM/network. Each test isolates the SQLite store via
the ``FINAI_MEMORY_DB`` env var so nothing touches a real memory file.
"""
from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from src.core import memory
from src.core.agents._base import ScopedAgent
from src.core.planner import _build_user_message
from src.core.types import KNOWN_INTENT_FLAGS, PlanStep, Scratchpad


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_scoped_agent.py so we can build a real agent)
# ---------------------------------------------------------------------------
class _BindableFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


def _fake_model() -> _BindableFakeModel:
    return _BindableFakeModel(responses=[AIMessage(content="ok")])


def _flags(**ov: bool) -> dict:
    base = {f: False for f in KNOWN_INTENT_FLAGS}
    base.update(ov)
    return base


def _tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(func=lambda **k: "ok", name=name, description="stub")


@pytest.fixture
def mem_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FINAI_MEMORY_DB", str(tmp_path / "mem.db"))
    yield


# ---------------------------------------------------------------------------
# is_real_user gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("uid", ["", "demo", "anonymous", "guest", "DEMO", " demo "])
def test_reserved_users_are_not_real(uid):
    assert memory.is_real_user(uid) is False


@pytest.mark.parametrize("uid", ["alice", "user-123", "tanmay"])
def test_named_users_are_real(uid):
    assert memory.is_real_user(uid) is True


# ---------------------------------------------------------------------------
# recall / observe basics
# ---------------------------------------------------------------------------
def test_recall_empty_returns_blank(mem_db):
    assert memory.recall("alice", "anything") == ""


def test_observe_then_recall_includes_topic(mem_db):
    memory.observe("alice", "What is NVDA's current valuation?")
    block = memory.recall("alice")
    assert "persistent memory" in block.lower()
    assert "NVDA" in block


def test_demo_user_is_never_stored(mem_db):
    memory.observe("demo", "I am a conservative long-term investor")
    assert memory.recall("demo") == ""
    assert memory.get_profile("demo") == {}


# ---------------------------------------------------------------------------
# Profile extraction (deterministic, conservative)
# ---------------------------------------------------------------------------
def test_observe_extracts_profile_signals(mem_db):
    memory.observe(
        "bob",
        "As a conservative investor saving for retirement, I hold for the long term.",
    )
    prof = memory.get_profile("bob")
    assert prof.get("risk_tolerance") == "conservative"
    assert prof.get("horizon") == "long-term"
    assert prof.get("goals") == "retirement planning"
    block = memory.recall("bob")
    assert "conservative" in block and "retirement planning" in block


def test_extract_profile_signals_direct():
    s = memory._extract_profile_signals("I want aggressive high-risk short-term speculative trades")
    assert s["risk_tolerance"] == "aggressive"
    assert s["horizon"] == "short-term"


def test_extract_returns_nothing_for_neutral_text():
    assert memory._extract_profile_signals("What time does the market open?") == {}


def test_set_profile_merges(mem_db):
    memory.set_profile("carol", risk_tolerance="moderate")
    memory.set_profile("carol", horizon="long-term")
    prof = memory.get_profile("carol")
    assert prof["risk_tolerance"] == "moderate" and prof["horizon"] == "long-term"


# ---------------------------------------------------------------------------
# Notes: dedup, query-relevance ordering, forget
# ---------------------------------------------------------------------------
def test_remember_dedups_consecutive(mem_db):
    memory.remember("dan", "likes dividend stocks", kind="fact")
    memory.remember("dan", "likes dividend stocks", kind="fact")
    assert memory.recall("dan").count("likes dividend stocks") == 1


def test_recall_orders_query_relevant_first(mem_db):
    memory.observe("eve", "Tell me about Tesla")
    memory.observe("eve", "Analyze Infosys fundamentals")
    block = memory.recall("eve", query="latest Tesla earnings")
    assert block.index("Tesla") < block.index("Infosys")


def test_forget_clears(mem_db):
    memory.observe("frank", "I am aggressive")
    assert memory.recall("frank") != ""
    memory.forget("frank")
    assert memory.recall("frank") == ""
    assert memory.get_profile("frank") == {}


# ---------------------------------------------------------------------------
# Integration: planner user message
# ---------------------------------------------------------------------------
def test_planner_user_message_includes_memory():
    block = ("### What we know about this user (persistent memory)\n"
             "- Risk tolerance: conservative")
    msg = _build_user_message("Should I buy NVDA?", None, block)
    assert "persistent memory" in msg and "conservative" in msg


def test_planner_user_message_without_memory_unchanged():
    msg = _build_user_message("Should I buy NVDA?", None, None)
    assert "persistent memory" not in msg


# ---------------------------------------------------------------------------
# Integration: ScopedAgent system prompt (covers every agent incl. synthesizer)
# ---------------------------------------------------------------------------
def _scoped(user_id: str, query: str = "news on Apple") -> ScopedAgent:
    return ScopedAgent(
        step=PlanStep(id=1, description="Look up news.", agent="research_agent",
                      tool_subset=["research__search_news"], depends_on=[]),
        scratchpad=Scratchpad(query=query),
        all_mcp_tools=[_tool("research__search_news")],
        model=_fake_model(),
        intent_flags=_flags(),
        user_id=user_id,
    )


def test_scoped_agent_prompt_includes_memory_for_authenticated_user(mem_db):
    memory.observe("grace", "I am a conservative long-term investor")
    sa = _scoped("grace")
    assert "persistent memory" in sa.system_prompt.lower()
    assert "conservative" in sa.system_prompt


def test_scoped_agent_prompt_excludes_memory_for_demo(mem_db):
    memory.observe("grace", "I am a conservative long-term investor")
    sa = _scoped("demo")
    assert "persistent memory" not in sa.system_prompt.lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
