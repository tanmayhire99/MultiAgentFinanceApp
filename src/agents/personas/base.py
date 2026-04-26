"""Shared machinery for persona agents.

All personas share the same:

* model configuration (GLM-5.1 on NVIDIA NIM, reasoning off, temperature 0.3)
* MCP tool pool (populated lazily by :mod:`src.config.mcp_servers`)
* graph shape (ReAct loop via :func:`langgraph.prebuilt.create_react_agent`)
* output contract: a :class:`PersonaVerdict` dict on the final turn

Each concrete persona only needs to supply its *name*, *title* (display label),
and *system prompt*.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


load_dotenv()

# Default model for all persona + moderator LLM calls.
#
# The demo originally defaulted to ``z-ai/glm-5.1``, but a head-to-head
# NIM benchmark showed ~60-90s time-to-first-token for GLM vs ~0.2-1s
# for the fast non-reasoning chat models, which was the single biggest
# reason the panel felt slow. We now default to a fast production-grade
# chat model; set ``NVIDIA_MODEL`` in the environment to override.
#
# Benchmark (Apr 2026, same API key, short plain-chat + Buffett-persona test):
#
#   model                                 TTFT    tok/s    persona voice
#   openai/gpt-oss-120b                   0.9 s   ~820     ** excellent ** (default)
#   meta/llama-3.3-70b-instruct           0.25s   ~280     ok, sometimes too bullish
#   mistralai/ministral-14b-instruct-..   0.3 s   ~310     strongest voice but slow end-to-end
#   meta/llama-3.1-8b-instruct            0.22s   ~850     decent, a bit generic
#   nvidia/llama-3.1-nemotron-nano-8b-v1  0.22s   ~385     terse, weak persona embodiment
#   z-ai/glm-5.1                          ~60s    ~200     excellent voice but 60-90s TTFT
#
# GPT-OSS 120B gives us strong Buffett-style voice (it correctly identified NVDA
# as "a wonderful company at a very rich price") together with sub-second TTFT
# and very high tokens-per-second, so it dominates the quality/speed frontier
# for this demo.
#
# ``os.getenv(..., default)`` only uses the default when the var is *unset*;
# docker-compose passes an empty string when the user hasn't set the override,
# so we need ``or`` to fall through on empty values too.
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL") or "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _extra_body_for(model: str) -> Dict[str, Any]:
    """Return model-specific ``extra_body`` options.

    NIM only accepts ``chat_template_kwargs`` for reasoning-capable models
    that implement the hybrid-thinking template (GLM and Qwen3-thinking
    variants). Passing it to Llama / Mistral / Gemma / GPT-OSS raises a
    400/422 at the NIM layer.
    """
    ml = model.lower()
    if "glm" in ml or "qwen" in ml:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


# ---------------------------------------------------------------------------
# API key pool
# ---------------------------------------------------------------------------
# We read one "primary" key from ``NVIDIA_API_KEY`` and up to four extra keys
# from ``NVIDIA_API_KEY_1`` .. ``NVIDIA_API_KEY_4``. Each persona is pinned
# to its own slot so that concurrent persona LLM calls hit different keys -
# NIM serialises / rate-limits per key, so this removes the main bottleneck.
_API_KEY_SLOTS: Dict[str, str] = {}


def _load_api_keys() -> None:
    """Populate :data:`_API_KEY_SLOTS` from the process environment."""
    primary = os.getenv("NVIDIA_API_KEY", "").strip()
    if not primary:
        raise ValueError("NVIDIA_API_KEY environment variable is missing")
    _API_KEY_SLOTS["primary"] = primary
    for slot in ("1", "2", "3", "4", "5", "6"):
        key = (os.getenv(f"NVIDIA_API_KEY_{slot}") or "").strip()
        if key:
            _API_KEY_SLOTS[slot] = key


_load_api_keys()


def get_api_key(slot: str = "primary") -> str:
    """Return the key for ``slot``, falling back to the primary key."""
    return _API_KEY_SLOTS.get(slot) or _API_KEY_SLOTS["primary"]


def list_configured_slots() -> List[str]:
    """Return the names of every slot that has a key configured."""
    return list(_API_KEY_SLOTS.keys())


# Persona -> key slot mapping. Moderator uses primary; each persona gets a
# distinct slot. If a slot is missing we transparently fall back to primary,
# but the demo is much faster when all four numbered keys are populated.
PERSONA_API_KEY_SLOT = {
    "buffett": "1",
    "wood": "2",
    "graham": "3",
    # "moderator" is implicit: uses "primary".
}


class PersonaVerdict(TypedDict, total=False):
    """Structured summary of a single persona's view.

    Only ``persona``, ``stance`` and ``one_liner`` are required; callers should
    fall back gracefully when optional fields are missing.
    """

    persona: str
    title: str
    stance: str  # "bullish" | "neutral" | "cautious" | "bearish"
    one_liner: str
    rationale: str
    key_metrics_cited: List[str]
    risk_caveats: List[str]
    tools_used: List[str]
    confidence: str  # "low" | "medium" | "high"


@dataclass(frozen=True)
class PersonaDef:
    """Static configuration for a persona agent."""

    name: str  # machine identifier, e.g. "buffett"
    title: str  # display label, e.g. "Warren Buffett (Value)"
    system_prompt: str


def build_chat_model(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1500,
    streaming: bool = True,
    api_key: Optional[str] = None,
    api_key_slot: str = "primary",
    response_format: Optional[Dict[str, Any]] = None,
) -> ChatOpenAI:
    """Construct a ``ChatOpenAI`` bound to NVIDIA NIM with thinking disabled.

    The ``extra_body.chat_template_kwargs.enable_thinking=False`` toggle is
    the single most important piece of configuration for GLM-style
    hybrid-thinking models: without it the model spends its token budget on
    reasoning tokens and leaves ``content`` empty.

    Multi-key routing
    -----------------
    The caller can specify ``api_key_slot`` (e.g. ``"1"``, ``"2"``, ``"3"``)
    to pin this model to a specific key in the pool. NIM serialises
    concurrent streaming requests *per API key*, so splitting the panel's
    four parallel LLM calls across four keys is what makes true real-time
    streaming (without dropped connections) reliable.

    ``api_key`` overrides ``api_key_slot`` when provided.

    Structured output
    -----------------
    Pass ``response_format={"type": "json_object"}`` to force the model to
    return a JSON object (used by :mod:`src.core.router`). Forwarded via
    ``model_kwargs`` so it reaches the OpenAI-compat request body
    regardless of the langchain-openai version in use.
    """
    resolved_key = api_key or get_api_key(api_key_slot)
    kwargs: Dict[str, Any] = dict(
        model=model,
        base_url=DEFAULT_BASE_URL,
        api_key=resolved_key,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        extra_body=_extra_body_for(model),
    )
    if response_format is not None:
        kwargs["model_kwargs"] = {"response_format": response_format}
    return ChatOpenAI(**kwargs)


def _format_system_prompt(persona: PersonaDef) -> str:
    """Append the shared output contract + debate-style block to every persona's system prompt."""
    return (
        f"{persona.system_prompt.strip()}\n\n"
        "### Debate stance: intellectual honesty over winning\n"
        "You are participating in a panel debate, not a televised "
        "argument. The goal is to arrive at the most honest assessment "
        "of the question — together, across multiple rounds — not to "
        "win against the other panelists.\n"
        "\n"
        "* When another panelist presents data or framing that genuinely "
        "shifts your evaluation, acknowledge it explicitly: \"X's point "
        "about ... updates my view because ...\". Crediting a good "
        "argument is a sign of strength, not weakness.\n"
        "* You may concede a specific point and still hold your overall "
        "stance — partial agreement is honest. Or you may update your "
        "stance entirely if the evidence warrants. Both are legitimate.\n"
        "* Do NOT adopt a position you don't hold just to manufacture "
        "agreement; do NOT dig in on a position to manufacture "
        "disagreement. Mean what you say.\n"
        "* The four stance labels (bullish/neutral/cautious/bearish) are "
        "rough buckets. Use them honestly — if a fellow panelist's "
        "argument moved you from \"bullish\" to \"I now see structural "
        "risks I hadn't considered\", drop to \"neutral\" or \"cautious\". "
        "Don't artificially stay bullish to avoid looking persuaded.\n"
        "\n"
        "### Output contract\n"
        "1. Use the available MCP tools to fetch any data you need. "
        "Prefer 2-4 focused tool calls over one broad fetch. (Subsequent "
        "rounds will tell you not to fetch more data — follow that "
        "instruction in those rounds.)\n"
        "2. After gathering the facts, write a short, first-person analysis in "
        "your own voice (length specified by the per-round instruction). "
        "Reference specific numbers from the tools you called or from "
        "the transcript above.\n"
        "3. Finish your message with EXACTLY this final block, on its own lines, "
        "and nothing after it:\n"
        "VERDICT: <one sentence, <=25 words>\n"
        "STANCE: <bullish|neutral|cautious|bearish>\n"
        "CONFIDENCE: <low|medium|high>\n"
        "Do not wrap the verdict block in markdown, code fences, or JSON."
    )


# Lazy module-level registry of MCP tools; populated by the panel supervisor
# at request time via :func:`src.config.mcp_servers.get_tools`.
_TOOL_REGISTRY: Dict[str, BaseTool] = {}


def register_tools(tools: List[BaseTool]) -> None:
    """Replace the in-process tool pool used by :func:`build_persona_agent`."""
    _TOOL_REGISTRY.clear()
    for t in tools:
        _TOOL_REGISTRY[t.name] = t


def available_tools() -> List[BaseTool]:
    """Return the current list of registered tools (order preserved by name)."""
    return list(_TOOL_REGISTRY.values())


def build_persona_agent(
    persona: PersonaDef,
    *,
    tools: Optional[List[BaseTool]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    """Return a coroutine that runs the persona's ReAct loop.

    The returned callable expects ``{"messages": [...]}`` and resolves to the
    final LangGraph state dict (containing the full message trajectory).

    Each persona is routed to its dedicated API key slot
    (see :data:`PERSONA_API_KEY_SLOT`). Because every persona uses a
    different key, all three can stream concurrently from NIM without
    hitting the per-key stream-concurrency limit, so we keep
    ``streaming=True`` for true real-time token delivery.
    """
    slot = PERSONA_API_KEY_SLOT.get(persona.name, "primary")
    llm = build_chat_model(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
        api_key_slot=slot,
    )
    candidate_tools = tools if tools is not None else available_tools()
    # Personas receive the portfolio snapshot in their user message, so they
    # do not need to call Portfolio Agent tools themselves (those are for
    # the orchestrator). Exclude them from the persona's tool pool.
    persona_tools = [
        t for t in candidate_tools if not t.name.startswith("portfolio__")
    ]

    # We keep the compiled graph so subsequent invocations reuse it.
    compiled = create_react_agent(
        model=llm,
        tools=persona_tools,
        prompt=_format_system_prompt(persona),
    )

    # ``create_react_agent`` defaults recursion_limit=25, which caps the
    # ReAct loop at ~10 tool calls. Defensive personas with 10-ticker
    # portfolios blow through that; see ``PERSONA_RECURSION_LIMIT`` in
    # :mod:`src.core.panel` for rationale. Keep imports local to avoid
    # a circular dependency between this module and ``src.core.panel``.
    async def _run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        from src.core.panel import PERSONA_RECURSION_LIMIT

        return await compiled.ainvoke(
            inputs, config={"recursion_limit": PERSONA_RECURSION_LIMIT}
        )

    _run.persona = persona  # type: ignore[attr-defined]
    _run.graph = compiled  # type: ignore[attr-defined]
    return _run


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------
import re

VERDICT_RE = re.compile(
    r"VERDICT:\s*(?P<verdict>.+?)\s*\n\s*STANCE:\s*(?P<stance>\w+)\s*\n\s*CONFIDENCE:\s*(?P<confidence>\w+)",
    re.IGNORECASE,
)


def strip_verdict_block(text: str) -> str:
    """Return the text with the trailing VERDICT/STANCE/CONFIDENCE block removed.

    Used by the streaming renderer so the raw machine-readable verdict is
    not shown twice (once in free-form text, then again in the structured
    verdict event).
    """
    match = VERDICT_RE.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip()


def parse_verdict(text: str, persona: PersonaDef) -> PersonaVerdict:
    """Extract the trailing VERDICT/STANCE/CONFIDENCE block from a persona reply."""
    result: PersonaVerdict = {
        "persona": persona.name,
        "title": persona.title,
        "rationale": text.strip(),
    }
    match = VERDICT_RE.search(text)
    if match:
        result["one_liner"] = match.group("verdict").strip()
        result["stance"] = match.group("stance").strip().lower()
        result["confidence"] = match.group("confidence").strip().lower()
        # The rationale should be everything *before* the verdict block
        result["rationale"] = text[: match.start()].strip()
    else:
        # Best-effort fallback: last non-empty line as the one-liner
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        result["one_liner"] = lines[-1] if lines else "(no summary)"
        result["stance"] = "neutral"
        result["confidence"] = "low"
    return result


def collect_tools_used(messages: List[Any]) -> List[str]:
    """Scan a message trajectory and return the names of every tool invoked."""
    used: List[str] = []
    seen = set()
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            # tool_calls may be dicts or objects depending on the LC version
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name and name not in seen:
                used.append(name)
                seen.add(name)
    return used
